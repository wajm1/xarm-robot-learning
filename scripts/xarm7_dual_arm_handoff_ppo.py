#!/usr/bin/env python3
"""GPU-parallel two-xArm7 cube handoff with Isaac Lab PhysX and RSL-RL PPO.

Task: a left xArm7 picks up a cube, brings it to the handoff zone, and a
right mirrored xArm7 receives it. Success requires the cube to be near the
receiver while separated from the giver.
Isaac RTX cameras are not used. Milestone MP4s are MuJoCo CPU-offscreen,
labeled non-RTX, written to videos/.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
os.environ.setdefault("ACCEPT_EULA", "Y")

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Train two xArm7 robots to hand off a cube with PPO.")
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--max_iterations", type=int, default=3000)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--play", action="store_true")
parser.add_argument("--smoke", action="store_true", help="Short launch test; does not train.")
parser.add_argument("--motion_test", action="store_true", help="Record one scripted PhysX handoff attempt; no training.")
parser.add_argument("--motion_probe", action="store_true", help="Check the PhysX motion without encoding a video.")
parser.add_argument("--geometry_probe", action="store_true", help="Print Isaac geometry without recording video.")
parser.add_argument("--video_only", action="store_true", help="Write one MuJoCo MP4 from a checkpoint; no training.")
parser.add_argument("--video_iteration", type=int, default=None, help="Iteration label for --video_only filenames.")
parser.add_argument("--checkpoint", type=str, default=None)
parser.add_argument("--eval_episodes", type=int, default=20)
parser.add_argument("--video_milestones", type=str, default="500,1000")
parser.add_argument("--log_dir", type=str, default="logs/xarm7_dual_handoff")
parser.add_argument("--video_dir", type=str, default="videos")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
# This Isaac Lab build has no --headless CLI flag; always run headless on the pod.
args.headless = True
if args.smoke or args.video_only or args.motion_test or args.motion_probe or args.geometry_probe:
    args.num_envs = min(args.num_envs, 4)
# Dual-arm PhysX is heavier; keep a safer default unless the caller overrides.
if not args.smoke and not args.video_only and args.num_envs == 256:
    args.num_envs = 128
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import importlib.metadata as metadata

import torch

import isaaclab.sim as sim_utils
from isaaclab import cloner
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.configclass import configclass
from isaaclab.utils.math import quat_apply
from isaaclab_rl.rsl_rl import (
    RslRlMLPModelCfg,
    RslRlOnPolicyRunnerCfg,
    RslRlPpoAlgorithmCfg,
    RslRlVecEnvWrapper,
    handle_deprecated_rsl_rl_cfg,
)
from rsl_rl.runners import OnPolicyRunner

try:
    from isaaclab_physx.physics import PhysxCfg

    _PHYSICS = PhysxCfg()
except Exception:
    _PHYSICS = None

XARM7_USD = f"{ISAAC_NUCLEUS_DIR}/Robots/Ufactory/xarm7/xarm7.usd"
TABLE_Z = 0.44
CUBE_SIZE = 0.055
LIFT_HEIGHT = 0.10
GRASP_DIST = 0.07
# Bases ~1.0 m apart so workspaces meet at the handoff zone without crowding.
GIVER_BASE = (0.0, 0.0, TABLE_Z)
RECV_BASE = (1.10, 0.0, TABLE_Z)
TABLE_CENTER = (0.55, 0.0, TABLE_Z - 0.04)
TABLE_SIZE = (1.50, 0.80, 0.08)
HANDOFF_POS = (0.55, 0.0, TABLE_Z + 0.18)
RECEIVER_READY_POS = (0.68, 0.0, TABLE_Z + 0.18)
# The UFACTORY gripper's link_tcp is 0.172 m along gripper-base local +Z.
TCP_OFFSET = (0.0, 0.0, 0.172)
# Menagerie / UFACTORY home — keeps the gripper well above the tabletop.
HOME_Q = (0.0, -0.247, 0.0, 0.909, 0.0, 1.15644, 0.0)
# Keep the receiver's wrist square to the approaching cube; the 90-degree
# tool-axis rotation made its jaws face the wrong way in the motion test.
RECEIVER_HOME_Q = (0.0, -0.12, 0.0, 0.909, 0.0, 1.025, 0.0)
# Exponential smoothing on joint targets (1 = no smoothing).
ACTION_SMOOTH = 0.18
PROJECT_ROOT = Path(__file__).resolve().parents[1]
VIDEO_SCRIPT = PROJECT_ROOT / "scripts" / "xarm7_dual_mujoco_video.py"
SYSTEM_PYTHON = "/usr/bin/python3"


def _t(value):
    """Isaac Lab 3 wraps tensors; older APIs returned them directly."""
    return value.torch if hasattr(value, "torch") else value


def _set_q_target(robot, q, joint_ids):
    if hasattr(robot, "set_joint_position_target_index"):
        robot.set_joint_position_target_index(target=q, joint_ids=joint_ids)
    else:
        robot.set_joint_position_target(q, joint_ids=joint_ids)


def _write_q(robot, q, qd, joint_ids, env_ids):
    if hasattr(robot, "write_joint_position_to_sim_index"):
        robot.write_joint_position_to_sim_index(position=q, joint_ids=joint_ids, env_ids=env_ids)
        robot.write_joint_velocity_to_sim_index(velocity=qd, joint_ids=joint_ids, env_ids=env_ids)
    else:
        robot.write_joint_position_to_sim(q, joint_ids=joint_ids, env_ids=env_ids)
        robot.write_joint_velocity_to_sim(qd, joint_ids=joint_ids, env_ids=env_ids)


def _write_pose(asset, pose, env_ids):
    if hasattr(asset, "write_root_pose_to_sim_index"):
        asset.write_root_pose_to_sim_index(root_pose=pose, env_ids=env_ids)
    else:
        asset.write_root_pose_to_sim(pose, env_ids=env_ids)


def _write_velocity(asset, velocity, env_ids):
    if hasattr(asset, "write_root_velocity_to_sim_index"):
        asset.write_root_velocity_to_sim_index(root_velocity=velocity, env_ids=env_ids)
    else:
        asset.write_root_velocity_to_sim(velocity, env_ids=env_ids)


@configclass
class XArm7HandoffCfg(DirectRLEnvCfg):
    decimation = 2
    # Dual-arm handoff needs more wall-clock than single-arm lift.
    episode_length_s = 10.0
    action_space = 16  # 7 arm joints + gripper for each arm
    # giver q(7)+qd(7), recv q(7)+qd(7), cube(3), cube_vel(3), cube-giver(3), cube-recv(3),
    # handoff delta(3), two grippers(2) = 45
    observation_space = 45
    state_space = 0
    # Smaller joint deltas keep both arms from winding into a looping pose.
    action_scale = 0.45
    sim: SimulationCfg = (
        SimulationCfg(dt=1.0 / 120.0, render_interval=decimation, physics=_PHYSICS)
        if _PHYSICS is not None
        else SimulationCfg(dt=1.0 / 120.0, render_interval=decimation)
    )
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=128, env_spacing=3.5, replicate_physics=True, clone_in_fabric=True
    )
    robot_usd: sim_utils.UsdFileCfg = sim_utils.UsdFileCfg(usd_path=XARM7_USD, activate_contact_sensors=False)


@configclass
class XArm7HandoffPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 32
    max_iterations = 3000
    save_interval = 500
    experiment_name = "xarm7_dual_handoff"
    actor = RslRlMLPModelCfg(
        hidden_dims=[256, 128, 64],
        activation="elu",
        obs_normalization=True,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=0.22),
    )
    critic = RslRlMLPModelCfg(hidden_dims=[256, 128, 64], activation="elu", obs_normalization=True)
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.003,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=3.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


def _find_articulation_root(prefix: str) -> str:
    import omni.usd
    from pxr import UsdPhysics

    stage = omni.usd.get_context().get_stage()
    roots = [
        str(prim.GetPath())
        for prim in stage.Traverse()
        if str(prim.GetPath()).startswith(prefix) and prim.HasAPI(UsdPhysics.ArticulationRootAPI)
    ]
    if not roots:
        raise RuntimeError(f"No ArticulationRootAPI under {prefix}. USD={XARM7_USD}")
    return max(roots, key=len)


class XArm7HandoffEnv(DirectRLEnv):
    """Centralized PPO policy: giver picks, receiver takes, then separates."""

    cfg: XArm7HandoffCfg

    def _setup_scene(self):
        self.cfg.robot_usd.func("/World/envs/env_0/Giver", self.cfg.robot_usd, translation=GIVER_BASE)
        # Isaac Lab 3 uses xyzw: (0, 0, 1, 0) is 180 deg about Z.
        self.cfg.robot_usd.func(
            "/World/envs/env_0/Receiver",
            self.cfg.robot_usd,
            translation=RECV_BASE,
            orientation=(0.0, 0.0, 1.0, 0.0),
        )
        for _ in range(8):
            simulation_app.update()
        giver_root = _find_articulation_root("/World/envs/env_0/Giver")
        receiver_root = _find_articulation_root("/World/envs/env_0/Receiver")
        giver_regex = giver_root.replace("/World/envs/env_0/", "/World/envs/env_.*/")
        receiver_regex = receiver_root.replace("/World/envs/env_0/", "/World/envs/env_.*/")

        # Spawn table/cube into env_0 only; cloner.replicate copies them to sibling envs.
        table_cfg = RigidObjectCfg(
            prim_path="/World/envs/env_0/Table",
            spawn=sim_utils.CuboidCfg(
                size=TABLE_SIZE,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.10, 0.12, 0.16)),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=TABLE_CENTER),
        )
        cube_cfg = RigidObjectCfg(
            prim_path="/World/envs/env_0/Cube",
            spawn=sim_utils.CuboidCfg(
                size=(CUBE_SIZE, CUBE_SIZE, CUBE_SIZE),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    solver_position_iteration_count=8, solver_velocity_iteration_count=1
                ),
                mass_props=sim_utils.MassPropertiesCfg(mass=0.075),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.2, dynamic_friction=1.0),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.82, 0.09, 0.06)),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.20, 0.0, TABLE_Z + CUBE_SIZE / 2.0)),
        )
        # Materialize env_0 props before cloning.
        RigidObject(table_cfg)
        RigidObject(cube_cfg)
        sim_utils.GroundPlaneCfg().func("/World/ground", sim_utils.GroundPlaneCfg())

        src, dest = "/World/envs/env_0", "/World/envs/env_{}"
        pos = cloner.grid_transforms(self.scene.num_envs, self.scene.cfg.env_spacing)[0]
        plan = cloner.clone_plan_from_env_0(
            src, dest, self.scene.num_envs, pos, global_paths=("/World/ground",)
        )
        cloner.replicate(plan)
        if "physx" in self.scene.physics_backend:
            self.scene.filter_collisions(global_prim_paths=["/World/ground"])

        arm_joints = {
            "joint1": HOME_Q[0], "joint2": HOME_Q[1], "joint3": HOME_Q[2], "joint4": HOME_Q[3],
            "joint5": HOME_Q[4], "joint6": HOME_Q[5], "joint7": HOME_Q[6],
        }
        receiver_joints = {
            **arm_joints,
            "joint2": RECEIVER_HOME_Q[1],
            "joint6": RECEIVER_HOME_Q[5],
            "joint7": RECEIVER_HOME_Q[6],
        }
        arm_actuators = {
                "arm": ImplicitActuatorCfg(joint_names_expr=["joint[1-7]"], stiffness=450.0, damping=90.0),
                "gripper": ImplicitActuatorCfg(joint_names_expr=["drive_joint", ".*finger.*", ".*knuckle.*"], stiffness=200.0, damping=40.0),
            }
        self.giver = Articulation(
            ArticulationCfg(
                prim_path=giver_regex,
                spawn=None,
                init_state=ArticulationCfg.InitialStateCfg(pos=GIVER_BASE, joint_pos=arm_joints),
                actuators=arm_actuators,
            )
        )
        self.receiver = Articulation(ArticulationCfg(
            prim_path=receiver_regex,
            spawn=None,
            init_state=ArticulationCfg.InitialStateCfg(
                pos=RECV_BASE, rot=(0.0, 0.0, 1.0, 0.0), joint_pos=receiver_joints
            ),
            actuators=arm_actuators,
        ))
        self.table = RigidObject(
            RigidObjectCfg(prim_path="/World/envs/env_.*/Table", spawn=None, init_state=table_cfg.init_state)
        )
        self.cube = RigidObject(
            RigidObjectCfg(prim_path="/World/envs/env_.*/Cube", spawn=None, init_state=cube_cfg.init_state)
        )
        self.scene.articulations["giver"] = self.giver
        self.scene.articulations["receiver"] = self.receiver
        self.scene.rigid_objects["table"] = self.table
        self.scene.rigid_objects["cube"] = self.cube
        light = sim_utils.DomeLightCfg(intensity=2200.0, color=(0.75, 0.78, 0.85))
        light.func("/World/Light", light)

    def __init__(self, cfg, render_mode=None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.arm_ids, arm_names = self.giver.find_joints("joint[1-7]")
        self.r_arm_ids, _ = self.receiver.find_joints("joint[1-7]")
        self.grip_ids = self.giver.find_joints("drive_joint")[0]
        self.r_grip_ids = self.receiver.find_joints("drive_joint")[0]
        if len(self.arm_ids) != 7 or len(self.r_arm_ids) != 7 or not len(self.grip_ids) or not len(self.r_grip_ids):
            raise RuntimeError("Expected two xArm7 models with 7 joints and a drive_joint each.")
        bodies = self.giver.find_bodies("xarm_gripper_base_link")
        rbodies = self.receiver.find_bodies("xarm_gripper_base_link")
        self.ee_id = bodies[0][0] if len(bodies[0]) else self.giver.find_bodies("link7")[0][0]
        self.r_ee_id = rbodies[0][0] if len(rbodies[0]) else self.receiver.find_bodies("link7")[0][0]
        self._tcp_offset = torch.tensor(TCP_OFFSET, device=self.device).repeat(self.num_envs, 1)
        self.actions = torch.zeros(self.num_envs, 16, device=self.device)
        # Force table-safe home even if the USD default is folded into the tabletop.
        self.default_q = torch.tensor(HOME_Q, device=self.device, dtype=torch.float32).unsqueeze(0).repeat(
            self.num_envs, 1
        )
        self.receiver_default_q = torch.tensor(
            RECEIVER_HOME_Q, device=self.device, dtype=torch.float32
        ).unsqueeze(0).repeat(self.num_envs, 1)
        self._q_arm_cmd = self.default_q.clone(); self._r_q_arm_cmd = self.receiver_default_q.clone()
        self._q_grip_cmd = torch.zeros(self.num_envs, 1, device=self.device); self._r_q_grip_cmd = torch.zeros(self.num_envs, 1, device=self.device)
        self._prev_actions = torch.zeros(self.num_envs, 16, device=self.device)
        limits = _t(self.giver.data.soft_joint_pos_limits)[0, self.grip_ids[0]]
        self.grip_low, self.grip_high = float(limits[0]), float(limits[1])
        arm_limits = _t(self.giver.data.soft_joint_pos_limits)[0, self.arm_ids]
        self.arm_low, self.arm_high = arm_limits[:, 0], arm_limits[:, 1]
        self._q_grip_cmd[:] = self.grip_low
        self._r_q_grip_cmd[:] = self.grip_low
        self.successes = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.returns = torch.zeros(self.num_envs, device=self.device)
        self.grasp_seen = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.lift_seen = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.handoff_seen = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.giver_hold_steps = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
        self.receiver_hold_steps = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
        self.min_giver_dist = torch.full((self.num_envs,), 1.0e6, device=self.device)
        self.min_receiver_dist = torch.full((self.num_envs,), 1.0e6, device=self.device)
        self.max_cube_z = torch.zeros(self.num_envs, device=self.device)
        self.min_handoff_dist = torch.full((self.num_envs,), 1.0e6, device=self.device)
        self._prev_dist = torch.ones(self.num_envs, device=self.device)
        self._prev_rdist = torch.ones(self.num_envs, device=self.device)
        self._prev_handoff = torch.ones(self.num_envs, device=self.device)
        self._prev_height_err = torch.ones(self.num_envs, device=self.device)
        self._prev_recv_ready = torch.ones(self.num_envs, device=self.device)
        self._prev_gclose = torch.zeros(self.num_envs, device=self.device)
        self._handoff = torch.tensor(HANDOFF_POS, device=self.device, dtype=torch.float32)
        self._receiver_ready = torch.tensor(RECEIVER_READY_POS, device=self.device, dtype=torch.float32)
        print(f"dual xArm7 handoff: giver={arm_names}, receiver={self.receiver.joint_names}")

    def _cube_pos(self):
        return _t(self.cube.data.root_pos_w)[:, :3]

    def _cube_vel(self):
        return _t(self.cube.data.root_lin_vel_w)[:, :3]

    def _ee_pos(self, robot, body_id):
        base_pos = _t(robot.data.body_pos_w)[:, body_id, :]
        base_quat = _t(robot.data.body_quat_w)[:, body_id, :]
        return base_pos + quat_apply(base_quat, self._tcp_offset)

    def _pre_physics_step(self, actions):
        # Clone so later in-place resets are valid outside torch.inference_mode.
        self.actions = actions.clamp(-1.0, 1.0).detach().clone()

    def _apply_action(self):
        q_arm_raw = torch.clamp(self.default_q + self.cfg.action_scale * self.actions[:, :7], self.arm_low, self.arm_high)
        r_arm_raw = torch.clamp(self.receiver_default_q + self.cfg.action_scale * self.actions[:, 8:15], self.arm_low, self.arm_high)
        q_grip_raw = (self.grip_low + 0.5 * (self.actions[:, 7] + 1.0) * (self.grip_high - self.grip_low)).unsqueeze(-1)
        r_grip_raw = (self.grip_low + 0.5 * (self.actions[:, 15] + 1.0) * (self.grip_high - self.grip_low)).unsqueeze(-1)
        a = ACTION_SMOOTH
        self._q_arm_cmd = (1.0 - a) * self._q_arm_cmd + a * q_arm_raw
        self._r_q_arm_cmd = (1.0 - a) * self._r_q_arm_cmd + a * r_arm_raw
        self._q_grip_cmd = (1.0 - a) * self._q_grip_cmd + a * q_grip_raw
        self._r_q_grip_cmd = (1.0 - a) * self._r_q_grip_cmd + a * r_grip_raw
        _set_q_target(self.giver, self._q_arm_cmd, self.arm_ids); _set_q_target(self.giver, self._q_grip_cmd, self.grip_ids)
        _set_q_target(self.receiver, self._r_q_arm_cmd, self.r_arm_ids); _set_q_target(self.receiver, self._r_q_grip_cmd, self.r_grip_ids)

    def _get_observations(self):
        q, qd = _t(self.giver.data.joint_pos), _t(self.giver.data.joint_vel)
        rq, rqd = _t(self.receiver.data.joint_pos), _t(self.receiver.data.joint_vel)
        cube = self._cube_pos()
        cvel = self._cube_vel()
        ee, ree = self._ee_pos(self.giver, self.ee_id), self._ee_pos(self.receiver, self.r_ee_id)
        handoff = self.scene.env_origins + self._handoff.unsqueeze(0)
        obs = torch.cat(
            (
                q[:, self.arm_ids] - self.default_q,
                qd[:, self.arm_ids],
                rq[:, self.r_arm_ids] - self.receiver_default_q,
                rqd[:, self.r_arm_ids],
                cube - self.scene.env_origins,
                cvel,
                cube - ee,
                cube - ree,
                cube - handoff,
                q[:, self.grip_ids],
                rq[:, self.r_grip_ids],
            ),
            dim=-1,
        )
        return {"policy": obs}

    def _get_rewards(self):
        cube = self._cube_pos()
        ee, ree = self._ee_pos(self.giver, self.ee_id), self._ee_pos(self.receiver, self.r_ee_id)
        dist = torch.linalg.norm(cube - ee, dim=-1)
        rdist = torch.linalg.norm(cube - ree, dim=-1)
        handoff_world = self.scene.env_origins + self._handoff.unsqueeze(0)
        handoff_dist = torch.linalg.norm(cube - handoff_world, dim=-1)
        cube_local = cube - self.scene.env_origins
        cube_z = cube[:, 2] - TABLE_Z
        cube_speed = torch.linalg.norm(self._cube_vel(), dim=-1)
        height_err = (cube_z - 0.16).abs()
        gclose = ((_t(self.giver.data.joint_pos)[:, self.grip_ids[0]] - self.grip_low) / max(self.grip_high - self.grip_low, 1e-6)).clamp(0.0, 1.0)
        rclose = ((_t(self.receiver.data.joint_pos)[:, self.r_grip_ids[0]] - self.grip_low) / max(self.grip_high - self.grip_low, 1e-6)).clamp(0.0, 1.0)
        giver_near = dist < GRASP_DIST
        receiver_near = rdist < GRASP_DIST
        # Proximity plus a closed gripper is not evidence of a load-bearing grasp.
        giver_holding = giver_near & (gclose > 0.55) & (cube_z > 0.055)
        self.giver_hold_steps = torch.where(giver_holding, self.giver_hold_steps + 1, 0)
        grasped = self.giver_hold_steps >= 4
        lifted = grasped & (cube_z > LIFT_HEIGHT)
        # Receiver can move toward a staging pose from the start, then track the cube
        # as soon as it is lifted. It must not wait for a narrow handoff-zone gate.
        in_zone = lifted & (handoff_dist < 0.14) & (cube_z < 0.28)
        acquired = in_zone & receiver_near & (rclose > 0.55)
        acquire_now = acquired & (~self.handoff_seen)
        # Acquisition is latched: after the giver opens, it cannot still satisfy
        # the giver-grasp predicate. The old simultaneous test made success impossible.
        receiver_support = (self.handoff_seen | acquired) & receiver_near & (rclose > 0.55)
        released = receiver_support & (gclose < 0.35) & (dist > 0.11) & (cube_z > 0.08)
        self.receiver_hold_steps = torch.where(released, self.receiver_hold_steps + 1, 0)
        stable = (self.receiver_hold_steps >= 18) & (cube_speed < 0.30)
        success_now = stable & (~self.successes)
        lift_now = lifted & (~self.lift_seen)
        grasp_now = grasped & (~self.grasp_seen)

        reach_prog = (self._prev_dist - dist).clamp(-0.04, 0.04)
        carry_prog = (self._prev_handoff - handoff_dist).clamp(-0.04, 0.04)
        recv_prog = (self._prev_rdist - rdist).clamp(-0.04, 0.04)
        height_prog = (self._prev_height_err - height_err).clamp(-0.04, 0.04)
        recv_ready_dist = torch.linalg.norm(
            ree - (self.scene.env_origins + self._receiver_ready.unsqueeze(0)), dim=-1
        )
        recv_ready_prev = self._prev_recv_ready
        ready_prog = (recv_ready_prev - recv_ready_dist).clamp(-0.04, 0.04)
        grip_close_prog = (gclose - self._prev_gclose).clamp(-0.04, 0.04)
        giver_retreat_prog = (dist - self._prev_dist).clamp(-0.04, 0.04)

        ee_clearance = torch.minimum(ee[:, 2], ree[:, 2]) - TABLE_Z
        table_pen = torch.clamp(0.02 - ee_clearance, min=0.0)
        behind_pen = torch.clamp(0.10 - cube_local[:, 0], min=0.0)
        too_high = torch.clamp(cube_z - 0.28, min=0.0)
        action_rate = (self.actions - self._prev_actions).square().sum(dim=-1)
        g_qd = _t(self.giver.data.joint_vel)[:, self.arm_ids].square().sum(dim=-1)
        r_qd = _t(self.receiver.data.joint_vel)[:, self.r_arm_ids].square().sum(dim=-1)
        pre_grasp = (~self.grasp_seen).float()
        holding = giver_holding.float() * (~self.handoff_seen).float()
        after_lift = self.lift_seen.float() * (~self.handoff_seen).float()
        receiver_stage = self.handoff_seen.float()
        reward = (
            10.0 * reach_prog * pre_grasp
            + 3.0 * grip_close_prog * giver_near.float() * pre_grasp
            + 6.0 * grasp_now.float()
            + 14.0 * carry_prog * holding
            + 7.0 * height_prog * holding
            + 8.0 * lift_now.float()
            + 12.0 * ready_prog * (~self.lift_seen).float()
            + 12.0 * recv_prog * after_lift
            + 3.0 * rclose * receiver_near.float() * after_lift
            + 16.0 * acquire_now.float()
            + 10.0 * giver_retreat_prog * receiver_stage
            + 0.12 * released.float()
            + 40.0 * success_now.float()
            - 0.004 * self.actions.square().sum(dim=-1)
            - 0.03 * action_rate
            - 0.002 * (g_qd + r_qd)
            - 12.0 * table_pen
            - 8.0 * behind_pen
            - 8.0 * too_high
            - 2.0 * torch.clamp(cube_speed - 0.45, min=0.0) * self.lift_seen.float()
            - 2.0 * ((gclose < 0.35) & self.lift_seen & (~self.handoff_seen)).float()
        ).detach()

        self._prev_actions = self.actions.detach().clone()
        self._prev_dist = dist.detach().clone()
        self._prev_rdist = rdist.detach().clone()
        self._prev_handoff = handoff_dist.detach().clone()
        self._prev_height_err = height_err.detach().clone()
        self._prev_recv_ready = recv_ready_dist.detach().clone()
        self._prev_gclose = gclose.detach().clone()
        self.grasp_seen = (self.grasp_seen | grasped).detach().clone()
        self.lift_seen = (self.lift_seen | lifted).detach().clone()
        self.handoff_seen = (self.handoff_seen | acquired).detach().clone()
        self.successes = (self.successes | stable).detach().clone()
        self.returns = (self.returns + reward).detach().clone()
        self.min_giver_dist = torch.minimum(self.min_giver_dist.detach(), dist.detach()).detach().clone()
        self.min_receiver_dist = torch.minimum(self.min_receiver_dist.detach(), rdist.detach()).detach().clone()
        self.max_cube_z = torch.maximum(self.max_cube_z.detach(), cube_z.detach()).detach().clone()
        self.min_handoff_dist = torch.minimum(self.min_handoff_dist.detach(), handoff_dist.detach()).detach().clone()
        return reward

    def _get_dones(self):
        cube = self._cube_pos()
        dropped = cube[:, 2] < (TABLE_Z - 0.12)
        timeout = self.episode_length_buf >= self.max_episode_length - 1
        # End the episode on a completed handoff so success credit is sharp.
        success = self.successes & (self.episode_length_buf > 5)
        return dropped | success, timeout

    def _detach_buffers(self):
        """Ensure episode buffers are normal tensors (not inference-mode)."""
        self.actions = self.actions.detach().clone()
        self._prev_actions = self._prev_actions.detach().clone()
        self._q_arm_cmd = self._q_arm_cmd.detach().clone()
        self._r_q_arm_cmd = self._r_q_arm_cmd.detach().clone()
        self._q_grip_cmd = self._q_grip_cmd.detach().clone()
        self._r_q_grip_cmd = self._r_q_grip_cmd.detach().clone()
        self.default_q = self.default_q.detach().clone()
        self.receiver_default_q = self.receiver_default_q.detach().clone()
        self.successes = self.successes.detach().clone()
        self.returns = self.returns.detach().clone()
        self.grasp_seen = self.grasp_seen.detach().clone()
        self.lift_seen = self.lift_seen.detach().clone()
        self.handoff_seen = self.handoff_seen.detach().clone()
        self.giver_hold_steps = self.giver_hold_steps.detach().clone()
        self.receiver_hold_steps = self.receiver_hold_steps.detach().clone()
        self.min_giver_dist = self.min_giver_dist.detach().clone()
        self.min_receiver_dist = self.min_receiver_dist.detach().clone()
        self.max_cube_z = self.max_cube_z.detach().clone()
        self.min_handoff_dist = self.min_handoff_dist.detach().clone()
        self._prev_dist = self._prev_dist.detach().clone()
        self._prev_rdist = self._prev_rdist.detach().clone()
        self._prev_handoff = self._prev_handoff.detach().clone()
        self._prev_height_err = self._prev_height_err.detach().clone()
        self._prev_recv_ready = self._prev_recv_ready.detach().clone()
        self._prev_gclose = self._prev_gclose.detach().clone()

    def _reset_idx(self, env_ids):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        self._detach_buffers()
        if len(env_ids):
            self.extras["log"] = {
                "Metrics/giver_grasp_rate": self.grasp_seen[env_ids].float().mean().item(),
                "Metrics/cube_lift_rate": self.lift_seen[env_ids].float().mean().item(),
                "Metrics/receiver_acquisition_rate": self.handoff_seen[env_ids].float().mean().item(),
                "Metrics/handoff_success_rate": self.successes[env_ids].float().mean().item(),
                "Metrics/min_giver_cube_m": self.min_giver_dist[env_ids].mean().item(),
                "Metrics/min_receiver_cube_m": self.min_receiver_dist[env_ids].mean().item(),
                "Metrics/max_cube_height_m": self.max_cube_z[env_ids].mean().item(),
                "Metrics/min_handoff_zone_m": self.min_handoff_dist[env_ids].mean().item(),
                "Episode/return": self.returns[env_ids].mean().item(),
                "Episode/length": self.episode_length_buf[env_ids].float().mean().item(),
            }
        self.giver.reset(env_ids); self.receiver.reset(env_ids)
        super()._reset_idx(env_ids)
        q = self.default_q[env_ids] + 0.02 * torch.randn_like(self.default_q[env_ids])
        rq = self.receiver_default_q[env_ids] + 0.02 * torch.randn_like(self.receiver_default_q[env_ids])
        zeros = torch.zeros_like(q)
        _write_q(self.giver, q, zeros, self.arm_ids, env_ids)
        _write_q(self.receiver, rq, zeros, self.r_arm_ids, env_ids)
        open_g = torch.full((len(env_ids), 1), self.grip_low, device=self.device)
        _write_q(self.giver, open_g, torch.zeros_like(open_g), self.grip_ids, env_ids)
        _write_q(self.receiver, open_g, torch.zeros_like(open_g), self.r_grip_ids, env_ids)
        n = len(env_ids)
        cube_pos = torch.zeros(n, 7, device=self.device)
        cube_pos[:, 0] = torch.empty(n, device=self.device).uniform_(0.12, 0.28)
        cube_pos[:, 1] = torch.empty(n, device=self.device).uniform_(-0.12, 0.12)
        cube_pos[:, 2] = TABLE_Z + CUBE_SIZE / 2.0
        cube_pos[:, 6] = 1.0  # Isaac Lab 3 root poses use xyzw.
        cube_pos[:, :3] += self.scene.env_origins[env_ids]
        _write_pose(self.cube, cube_pos, env_ids)
        _write_velocity(self.cube, torch.zeros(n, 6, device=self.device), env_ids)
        self.actions[env_ids] = 0
        self._prev_actions[env_ids] = 0
        self._q_arm_cmd[env_ids] = self.default_q[env_ids]
        self._r_q_arm_cmd[env_ids] = self.receiver_default_q[env_ids]
        self._q_grip_cmd[env_ids] = self.grip_low
        self._r_q_grip_cmd[env_ids] = self.grip_low
        self.successes[env_ids] = False
        self.grasp_seen[env_ids] = False
        self.lift_seen[env_ids] = False
        self.handoff_seen[env_ids] = False
        self.giver_hold_steps[env_ids] = 0
        self.receiver_hold_steps[env_ids] = 0
        self.returns[env_ids] = 0
        self.min_giver_dist[env_ids] = 1.0e6
        self.min_receiver_dist[env_ids] = 1.0e6
        self.max_cube_z[env_ids] = 0.0
        self.min_handoff_dist[env_ids] = 1.0e6
        self._prev_dist[env_ids] = 1.0
        self._prev_rdist[env_ids] = 1.0
        self._prev_handoff[env_ids] = 1.0
        self._prev_height_err[env_ids] = 1.0
        self._prev_recv_ready[env_ids] = 1.0
        self._prev_gclose[env_ids] = 0.0


def _policy_step(policy, obs):
    # Do not leave InferenceMode tensors inside the env.
    with torch.inference_mode():
        act = policy(obs)
    if torch.is_tensor(act):
        return act.detach().clone()
    return act


def evaluate(env, policy, episodes: int) -> dict:
    with torch.inference_mode(False):
        obs, _ = env.reset()
    completed = 0
    keys = [
        "Metrics/giver_grasp_rate",
        "Metrics/cube_lift_rate",
        "Metrics/receiver_acquisition_rate",
        "Metrics/handoff_success_rate",
    ]
    metrics = {k: [] for k in keys}
    while completed < episodes:
        with torch.inference_mode(False):
            obs, _, terminated, truncated, info = env.step(_policy_step(policy, obs))
        done = terminated | truncated
        if done.any() and "log" in info:
            take = min(int(done.sum().item()), episodes - completed)
            completed += take
            for key in metrics:
                metrics[key].append(info["log"].get(key, 0.0))
    summary = {k: (sum(v) / len(v) if v else 0.0) for k, v in metrics.items()}
    print(
        "Evaluation: "
        f"episodes={completed} "
        f"grasp={summary['Metrics/giver_grasp_rate']:.3f} "
        f"lift={summary['Metrics/cube_lift_rate']:.3f} "
        f"acquire={summary['Metrics/receiver_acquisition_rate']:.3f} "
        f"handoff={summary['Metrics/handoff_success_rate']:.3f}"
    )
    return summary


def record_isaac_states(
    env: XArm7HandoffEnv, policy, steps: int | None = None, action_fn=None, fixed_cube: bool = False
) -> Path:
    """Record one real PhysX episode for dual-arm MuJoCo replay."""
    if steps is None:
        steps = int(env.max_episode_length)
    with torch.inference_mode(False):
        obs, _ = env.reset()
    if fixed_cube:
        # Scripted review uses a known cube pose; PPO training keeps the random reset.
        cube_pose = torch.zeros(env.num_envs, 7, device=env.device)
        cube_pose[:, 0] = 0.20
        cube_pose[:, 2] = TABLE_Z + CUBE_SIZE / 2.0
        cube_pose[:, 6] = 1.0
        cube_pose[:, :3] += env.scene.env_origins
        all_env_ids = torch.arange(env.num_envs, device=env.device)
        _write_pose(env.cube, cube_pose, all_env_ids)
        _write_velocity(env.cube, torch.zeros(env.num_envs, 6, device=env.device), all_env_ids)
    g_arm, r_arm, g_grip, r_grip, cube = [], [], [], [], []
    prev_cube = None
    for step in range(steps):
        act = (
            action_fn(env, step)
            if action_fn is not None
            else _policy_step(policy, obs)
            if policy is not None
            else torch.zeros((env.num_envs, 16), device=env.device)
        )
        with torch.inference_mode(False):
            obs, _, terminated, truncated, _ = env.step(act)
        # The first reset ends this episode. Concatenating another episode makes
        # an apparently frozen or teleporting milestone video.
        if bool((terminated | truncated)[0].item()):
            print(f"Replay ended at step {step}: episode reset.", flush=True)
            break
        c = (env._cube_pos()[0] - env.scene.env_origins[0]).detach().cpu().numpy()
        if prev_cube is not None and float(np.linalg.norm(c - prev_cube)) > 0.25:
            prev_cube = c
            continue
        prev_cube = c
        if action_fn is not None and step % 60 == 0:
            print(
                f"motion step {step}: cube={np.round(c, 3)} "
                f"giver-cube={float(torch.linalg.norm(env._ee_pos(env.giver, env.ee_id)[0] - env._cube_pos()[0])):.3f} "
                f"receiver-cube={float(torch.linalg.norm(env._ee_pos(env.receiver, env.r_ee_id)[0] - env._cube_pos()[0])):.3f}",
                flush=True,
            )
        gq = _t(env.giver.data.joint_pos)
        rq = _t(env.receiver.data.joint_pos)
        g_arm.append(gq[0, env.arm_ids].detach().cpu().numpy())
        r_arm.append(rq[0, env.r_arm_ids].detach().cpu().numpy())
        g_grip.append(float(gq[0, env.grip_ids[0]].detach().cpu()))
        r_grip.append(float(rq[0, env.r_grip_ids[0]].detach().cpu()))
        cube.append(c)
    if len(g_arm) < 8:
        raise RuntimeError("Too few frames recorded for dual-arm video")
    out = Path("/tmp/xarm7_dual_isaac_states.npz")
    np.savez(
        out,
        giver_arm_q=np.stack(g_arm),
        recv_arm_q=np.stack(r_arm),
        giver_gripper=np.asarray(g_grip),
        recv_gripper=np.asarray(r_grip),
        cube_pos=np.stack(cube),
    )
    return out


def scripted_handoff_actions(env: XArm7HandoffEnv, step: int) -> torch.Tensor:
    """Kinematics checked open-loop reference for a single review video.

    These joint poses put the two MuJoCo TCP sites on reachable points above
    the table within the PPO action range. PhysX still determines whether the
    cube is actually grasped and transferred.
    """
    home = np.asarray(HOME_Q, dtype=np.float32)
    plan = (
        (0, home, home, -1.0, -1.0),
        (60, np.array((0, -.2592, 0, .5928, 0, 1.4343, 0)),
         np.array((0, -.1091, 0, .9127, 0, 1.0547, 0)), -1.0, -1.0),
        (120, np.array((0, -.0137, 0, .5187, 0, 1.1322, 0)),
         np.array((0, -.1091, 0, .9127, 0, 1.0547, 0)), -1.0, -1.0),
        (160, np.array((0, -.0137, 0, .5187, 0, 1.1322, 0)),
         np.array((0, -.1091, 0, .9127, 0, 1.0547, 0)), 1.0, -1.0),
        (220, np.array((0, -.3996, 0, .6407, 0, 1.5650, 0)),
         np.array((0, -.1091, 0, .9127, 0, 1.0547, 0)), 1.0, -1.0),
        (320, np.array((0, .0497, 0, 1.0267, 0, .7857, 0)),
         np.array((0, -.1091, 0, .9127, 0, 1.0547, 0)), 1.0, -1.0),
        (380, np.array((0, .0497, 0, 1.0267, 0, .7857, 0)),
         np.array((.0765, .0887, .0747, 1.1482, .0343, .7589, 0)), 1.0, -1.0),
        (470, np.array((0, .0497, 0, 1.0267, 0, .7857, 0)),
         np.array((.0866, .2861, .0820, 1.2764, .0385, .5778, 0)), 1.0, 1.0),
        (525, np.array((0, -.4612, 0, .8133, 0, 1.2477, 0)),
         np.array((.0866, .2861, .0820, 1.2764, .0385, .5778, 0)), -1.0, 1.0),
        (590, np.array((0, -.4612, 0, .8133, 0, 1.2477, 0)),
         np.array((.1283, -.3638, .1246, .9116, .0445, 1.1086, 0)), -1.0, 1.0),
    )
    left, right = plan[-2], plan[-1]
    for a, b in zip(plan[:-1], plan[1:]):
        if step <= b[0]:
            left, right = a, b
            break
    alpha = np.clip((step - left[0]) / max(right[0] - left[0], 1), 0.0, 1.0)
    giver_q = left[1] + alpha * (right[1] - left[1])
    recv_q = left[2] + alpha * (right[2] - left[2])
    recv_q[6] = RECEIVER_HOME_Q[6]
    action = np.zeros(16, dtype=np.float32)
    action[:7] = np.clip((giver_q - home) / env.cfg.action_scale, -1.0, 1.0)
    action[8:15] = np.clip((recv_q - np.asarray(RECEIVER_HOME_Q)) / env.cfg.action_scale, -1.0, 1.0)
    action[7] = left[3] + alpha * (right[3] - left[3])
    action[15] = left[4] + alpha * (right[4] - left[4])
    return torch.as_tensor(action, device=env.device).unsqueeze(0).repeat(env.num_envs, 1)


def render_mujoco_video(states: Path | None, output: Path, label: str, seconds: float | None = None) -> None:
    cmd = [
        SYSTEM_PYTHON,
        "-u",
        str(VIDEO_SCRIPT),
        "--output",
        str(output),
        "--label",
        label,
    ]
    playback_s = seconds if seconds is not None else 10.0
    if states is not None:
        cmd.extend(["--states", str(states)])
    cmd.extend(["--seconds", str(playback_s)])
    env = os.environ.copy()
    env["MUJOCO_GL"] = "osmesa"
    env["PYTHONUNBUFFERED"] = "1"
    print("Rendering dual-arm non-RTX MuJoCo video:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, env=env, cwd=str(PROJECT_ROOT))


def smoke_test(env: XArm7HandoffEnv) -> None:
    print("=== DUAL-ARM SMOKE TEST (no training) ===")
    obs, _ = env.reset()
    print(f"num_envs={env.num_envs} obs={tuple(obs['policy'].shape)} action_dim=16")
    if obs["policy"].shape[-1] != env.cfg.observation_space:
        raise RuntimeError(f"obs dim {obs['policy'].shape[-1]} != cfg {env.cfg.observation_space}")
    zeros = torch.zeros((env.num_envs, 16), device=env.device)
    for step in range(16):
        obs, rew, term, trunc, info = env.step(zeros)
        if step == 0:
            print(f"reward[0]={float(rew[0]):.4f} cube={env._cube_pos()[0].detach().cpu().numpy()}")
    print("Isaac PhysX dual-arm env stepped successfully (no RTX camera used).")
    video = PROJECT_ROOT / args.video_dir / "dual_smoke_mujoco_non_rtx.mp4"
    states = record_isaac_states(env, policy=None, steps=90)
    render_mujoco_video(states, video, "dual smoke / Isaac-state replay", seconds=3.0)
    print(f"SMOKE_OK video={video}")


def dump_milestone_video(env, runner, iteration: int, video_dir: Path, log_dir: Path) -> Path:
    """Write a full-episode dual-arm non-RTX MuJoCo MP4 from the current policy."""
    if hasattr(env, "_detach_buffers"):
        env._detach_buffers()
    policy = runner.get_inference_policy(device=env.device)
    states = record_isaac_states(env, policy)
    output = video_dir / f"xarm7_dual_iter_{iteration:04d}_mujoco_non_rtx.mp4"
    render_mujoco_video(
        states, output, f"dual iter {iteration} handoff replay", seconds=float(env.cfg.episode_length_s)
    )
    if not output.is_file() or output.stat().st_size < 1024:
        raise RuntimeError(f"Milestone video was not written: {output}")
    print(f"MILESTONE_VIDEO {output} bytes={output.stat().st_size}")
    try:
        metrics = evaluate(env, policy, min(args.eval_episodes, 8))
        (log_dir / f"metrics_{iteration}.json").write_text(json.dumps({"iteration": iteration, **metrics}, indent=2))
    except Exception as exc:
        print(f"WARNING: eval metrics at {iteration} failed: {exc}")
    return output


def main() -> None:
    torch.manual_seed(args.seed)
    cfg = XArm7HandoffCfg()
    cfg.scene.num_envs = 1 if args.play or args.motion_test or args.motion_probe or args.geometry_probe else args.num_envs
    cfg.sim.device = args.device or "cuda:0"
    cfg.seed = args.seed
    env = XArm7HandoffEnv(cfg, render_mode=None)
    if args.geometry_probe:
        env.reset()
        print("GEOMETRY giver root", _t(env.giver.data.root_pos_w)[0].detach().cpu().numpy(), flush=True)
        print("GEOMETRY recv root", _t(env.receiver.data.root_pos_w)[0].detach().cpu().numpy(), flush=True)
        print("GEOMETRY giver base body", _t(env.giver.data.body_pos_w)[0, env.ee_id].detach().cpu().numpy(), flush=True)
        print("GEOMETRY recv base body", _t(env.receiver.data.body_pos_w)[0, env.r_ee_id].detach().cpu().numpy(), flush=True)
        print("GEOMETRY giver tcp", env._ee_pos(env.giver, env.ee_id)[0].detach().cpu().numpy(), flush=True)
        print("GEOMETRY recv tcp", env._ee_pos(env.receiver, env.r_ee_id)[0].detach().cpu().numpy(), flush=True)
        print("GEOMETRY cube", env._cube_pos()[0].detach().cpu().numpy(), flush=True)
        print("GEOMETRY cube vel", env._cube_vel()[0].detach().cpu().numpy(), flush=True)
        env.close()
        simulation_app.close()
        return
    if args.smoke:
        smoke_test(env)
        env.close()
        simulation_app.close()
        return
    if args.motion_test or args.motion_probe:
        states = record_isaac_states(
            env, policy=None, steps=595, action_fn=scripted_handoff_actions, fixed_cube=True
        )
        output = PROJECT_ROOT / args.video_dir / "xarm7_dual_motion_test_non_rtx.mp4"
        if args.motion_test:
            render_mujoco_video(states, output, "scripted PhysX handoff attempt", seconds=10.0)
        traj = np.load(states)
        cubes = traj["cube_pos"]
        print(
            f"MOTION_TEST frames={len(cubes)} max_cube_height={float(cubes[:, 2].max() - TABLE_Z):.3f} "
            f"max_cube_x={float(cubes[:, 0].max()):.3f} "
            f"giver_grasp={bool(env.grasp_seen[0])} lift={bool(env.lift_seen[0])} "
            f"receiver_acquired={bool(env.handoff_seen[0])} handoff_success={bool(env.successes[0])}",
            flush=True,
        )
        if args.motion_test:
            print(f"MOTION_TEST_VIDEO {output}", flush=True)
        env.close()
        simulation_app.close()
        return

    runner_cfg = XArm7HandoffPPORunnerCfg()
    runner_cfg.max_iterations = args.max_iterations
    runner_cfg = handle_deprecated_rsl_rl_cfg(runner_cfg, metadata.version("rsl-rl-lib"))
    log_dir = PROJECT_ROOT / args.log_dir
    log_dir.mkdir(parents=True, exist_ok=True)
    video_dir = PROJECT_ROOT / args.video_dir
    video_dir.mkdir(parents=True, exist_ok=True)
    wrapped = RslRlVecEnvWrapper(env, clip_actions=runner_cfg.clip_actions)
    runner = OnPolicyRunner(wrapped, runner_cfg.to_dict(), log_dir=str(log_dir), device=cfg.sim.device)
    if args.checkpoint:
        runner.load(args.checkpoint)

    if args.play:
        evaluate(env, runner.get_inference_policy(device=env.device), args.eval_episodes)
        env.close()
        simulation_app.close()
        return

    if args.video_only:
        if not args.checkpoint:
            raise RuntimeError("--video_only requires --checkpoint")
        iteration = args.video_iteration
        if iteration is None:
            digits = "".join(ch for ch in Path(args.checkpoint).stem if ch.isdigit())
            iteration = int(digits) if digits else 0
            if iteration == 599:
                iteration = 600
        print(f"Writing non-RTX MuJoCo video for iteration {iteration} from {args.checkpoint}")
        dump_milestone_video(env, runner, iteration, video_dir, log_dir)
        env.close()
        simulation_app.close()
        return

    milestones = sorted({int(x) for x in args.video_milestones.split(",") if x.strip()})
    print(f"Training PPO for {args.max_iterations} iterations; video milestones={milestones}")
    print("Isaac RTX cameras are disabled. Videos are MuJoCo CPU-offscreen (non-RTX).")
    start = getattr(runner, "current_learning_iteration", 0)
    for milestone in milestones:
        if milestone > args.max_iterations:
            continue
        chunk = milestone - start
        if chunk > 0:
            runner.learn(num_learning_iterations=chunk, init_at_random_ep_len=(start == 0))
        # RSL-RL is 0-based (599 after 600 updates). Always emit this milestone's video.
        start = max(milestone, getattr(runner, "current_learning_iteration", 0))
        try:
            dump_milestone_video(env, runner, milestone, video_dir, log_dir)
        except Exception as exc:
            print(f"ERROR: milestone {milestone} video failed: {exc}")
            print("Continuing training so the run is not lost.")
        print(f"CHECKPOINT_DIR {log_dir}")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
