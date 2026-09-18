#!/usr/bin/env python3
"""GPU-parallel xArm7 cube pickup/lift with Isaac Lab PhysX and RSL-RL PPO.

Task: 7-DoF UFACTORY xArm7 + gripper, table, randomized physical cube.
Success is a lift (cube raised while still near the gripper), not a reach.
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

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
os.environ.setdefault("ACCEPT_EULA", "Y")

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Train or evaluate xArm7 cube pickup/lift with PPO.")
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--max_iterations", type=int, default=3000)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--play", action="store_true")
parser.add_argument("--smoke", action="store_true", help="Short launch test; does not train.")
parser.add_argument("--video_only", action="store_true", help="Write one MuJoCo MP4 from a checkpoint; no training.")
parser.add_argument("--video_iteration", type=int, default=None, help="Iteration label for --video_only filenames.")
parser.add_argument("--checkpoint", type=str, default=None)
parser.add_argument("--eval_episodes", type=int, default=20)
parser.add_argument("--video_milestones", type=str, default="600,1200,1800,2400,3000")
parser.add_argument("--log_dir", type=str, default="logs/xarm7_cube_lift")
parser.add_argument("--video_dir", type=str, default="videos")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
# This Isaac Lab build has no --headless CLI flag; always run headless on the pod.
args.headless = True
if args.smoke or args.video_only:
    args.num_envs = min(args.num_envs, 4)
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
LIFT_HEIGHT = 0.08
GRASP_DIST = 0.07
# The UFACTORY gripper's link_tcp is 0.172 m along gripper-base local +Z.
TCP_OFFSET = (0.0, 0.0, 0.172)
# Menagerie / UFACTORY home — keeps the gripper well above the tabletop.
HOME_Q = (0.0, -0.247, 0.0, 0.909, 0.0, 1.15644, 0.0)
# Exponential smoothing on joint targets (1 = no smoothing).
ACTION_SMOOTH = 0.35
PROJECT_ROOT = Path(__file__).resolve().parents[1]
VIDEO_SCRIPT = PROJECT_ROOT / "scripts" / "xarm7_mujoco_video.py"
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


@configclass
class XArm7LiftCfg(DirectRLEnvCfg):
    decimation = 2
    episode_length_s = 5.0
    action_space = 8  # 7 arm joints + gripper
    observation_space = 21  # q(7), qd(7), cube(3), ee-cube(3), gripper(1)
    state_space = 0
    # Absolute targets around home; 0.12 rad could not bring the TCP to the cube.
    action_scale = 1.0
    sim: SimulationCfg = (
        SimulationCfg(dt=1.0 / 120.0, render_interval=decimation, physics=_PHYSICS)
        if _PHYSICS is not None
        else SimulationCfg(dt=1.0 / 120.0, render_interval=decimation)
    )
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=256, env_spacing=2.4, replicate_physics=True, clone_in_fabric=True
    )
    robot_usd: sim_utils.UsdFileCfg = sim_utils.UsdFileCfg(usd_path=XARM7_USD, activate_contact_sensors=False)


@configclass
class XArm7PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 3000
    save_interval = 500
    experiment_name = "xarm7_cube_lift"
    actor = RslRlMLPModelCfg(
        hidden_dims=[256, 128, 64],
        activation="elu",
        obs_normalization=True,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=0.35),
    )
    critic = RslRlMLPModelCfg(hidden_dims=[256, 128, 64], activation="elu", obs_normalization=True)
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.004,
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


class XArm7LiftEnv(DirectRLEnv):
    """Pickup/lift: close the gripper on a cube and raise it off the table."""

    cfg: XArm7LiftCfg

    def _setup_scene(self):
        self.cfg.robot_usd.func("/World/envs/env_0/Robot", self.cfg.robot_usd, translation=(0.0, 0.0, TABLE_Z))
        for _ in range(8):
            simulation_app.update()
        root = _find_articulation_root("/World/envs/env_0/Robot")
        regex = root.replace("/World/envs/env_0/", "/World/envs/env_.*/")

        # Spawn table/cube into env_0 only; cloner.replicate copies them to sibling envs.
        table_cfg = RigidObjectCfg(
            prim_path="/World/envs/env_0/Table",
            spawn=sim_utils.CuboidCfg(
                size=(0.80, 0.70, 0.08),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.10, 0.12, 0.16)),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.30, 0.0, TABLE_Z - 0.04)),
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
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.45, 0.0, TABLE_Z + CUBE_SIZE / 2.0)),
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

        self.robot = Articulation(
            ArticulationCfg(
                prim_path=regex,
                spawn=None,
                init_state=ArticulationCfg.InitialStateCfg(joint_pos={
                    "joint1": HOME_Q[0],
                    "joint2": HOME_Q[1],
                    "joint3": HOME_Q[2],
                    "joint4": HOME_Q[3],
                    "joint5": HOME_Q[4],
                    "joint6": HOME_Q[5],
                    "joint7": HOME_Q[6],
                }),
                actuators={
                    "arm": ImplicitActuatorCfg(joint_names_expr=["joint[1-7]"], stiffness=450.0, damping=90.0),
                    "gripper": ImplicitActuatorCfg(
                        joint_names_expr=["drive_joint", ".*finger.*", ".*knuckle.*"],
                        stiffness=200.0,
                        damping=40.0,
                    ),
                },
            )
        )
        self.table = RigidObject(
            RigidObjectCfg(prim_path="/World/envs/env_.*/Table", spawn=None, init_state=table_cfg.init_state)
        )
        self.cube = RigidObject(
            RigidObjectCfg(prim_path="/World/envs/env_.*/Cube", spawn=None, init_state=cube_cfg.init_state)
        )
        self.scene.articulations["robot"] = self.robot
        self.scene.rigid_objects["table"] = self.table
        self.scene.rigid_objects["cube"] = self.cube
        light = sim_utils.DomeLightCfg(intensity=2200.0, color=(0.75, 0.78, 0.85))
        light.func("/World/Light", light)

    def __init__(self, cfg, render_mode=None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.arm_ids, arm_names = self.robot.find_joints("joint[1-7]")
        if len(self.arm_ids) != 7:
            raise RuntimeError(f"Expected 7 arm joints, found {arm_names}")
        grip = self.robot.find_joints("drive_joint")
        if len(grip[0]) == 0:
            grip = self.robot.find_joints(".*driver.*")
        if len(grip[0]) == 0:
            raise RuntimeError(f"No gripper drive joint in {self.robot.joint_names}")
        self.grip_ids = grip[0]
        bodies = self.robot.find_bodies("xarm_gripper_base_link")
        if len(bodies[0]) == 0:
            bodies = self.robot.find_bodies("link7")
        self.ee_id = bodies[0][0]
        self._tcp_offset = torch.tensor(TCP_OFFSET, device=self.device).repeat(self.num_envs, 1)
        self.actions = torch.zeros(self.num_envs, 8, device=self.device)
        # Force table-safe home even if the USD default is folded into the tabletop.
        self.default_q = torch.tensor(HOME_Q, device=self.device, dtype=torch.float32).unsqueeze(0).repeat(
            self.num_envs, 1
        )
        self._q_arm_cmd = self.default_q.clone()
        self._q_grip_cmd = torch.zeros(self.num_envs, 1, device=self.device)
        self._prev_actions = torch.zeros(self.num_envs, 8, device=self.device)
        limits = _t(self.robot.data.soft_joint_pos_limits)[0, self.grip_ids[0]]
        self.grip_low, self.grip_high = float(limits[0]), float(limits[1])
        arm_limits = _t(self.robot.data.soft_joint_pos_limits)[0, self.arm_ids]
        self.arm_low, self.arm_high = arm_limits[:, 0], arm_limits[:, 1]
        self._q_grip_cmd[:] = self.grip_low
        self.successes = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.returns = torch.zeros(self.num_envs, device=self.device)
        self.min_ee_dist = torch.full((self.num_envs,), 1.0e6, device=self.device)
        self.max_cube_z = torch.zeros(self.num_envs, device=self.device)
        print(f"xArm7 USD={XARM7_USD}")
        print(f"arm joints={arm_names}")
        print(f"gripper joints={[self.robot.joint_names[i] for i in self.grip_ids]}")
        print(f"ee body={self.robot.body_names[self.ee_id]}")
        print(f"home_q={list(HOME_Q)} action_scale={self.cfg.action_scale} smooth={ACTION_SMOOTH}")

    def _cube_pos(self):
        return _t(self.cube.data.root_pos_w)[:, :3]

    def _ee_pos(self):
        base_pos = _t(self.robot.data.body_pos_w)[:, self.ee_id, :]
        base_quat = _t(self.robot.data.body_quat_w)[:, self.ee_id, :]
        return base_pos + quat_apply(base_quat, self._tcp_offset)

    def _pre_physics_step(self, actions):
        # Clone so later in-place resets are valid outside torch.inference_mode.
        self.actions = actions.clamp(-1.0, 1.0).detach().clone()

    def _apply_action(self):
        q_arm_raw = self.default_q + self.cfg.action_scale * self.actions[:, :7]
        q_arm_raw = torch.clamp(q_arm_raw, self.arm_low, self.arm_high)
        close = 0.5 * (self.actions[:, 7] + 1.0)
        q_grip_raw = (self.grip_low + close * (self.grip_high - self.grip_low)).unsqueeze(-1)
        # Low-pass joint targets so the arm does not thrash frame-to-frame.
        a = ACTION_SMOOTH
        self._q_arm_cmd = (1.0 - a) * self._q_arm_cmd + a * q_arm_raw
        self._q_grip_cmd = (1.0 - a) * self._q_grip_cmd + a * q_grip_raw
        _set_q_target(self.robot, self._q_arm_cmd, self.arm_ids)
        _set_q_target(self.robot, self._q_grip_cmd, self.grip_ids)

    def _get_observations(self):
        q = _t(self.robot.data.joint_pos)
        qd = _t(self.robot.data.joint_vel)
        cube = self._cube_pos()
        ee = self._ee_pos()
        obs = torch.cat(
            (
                q[:, self.arm_ids] - self.default_q,
                qd[:, self.arm_ids],
                cube - self.scene.env_origins,
                cube - ee,
                q[:, self.grip_ids],
            ),
            dim=-1,
        )
        return {"policy": obs}

    def _get_rewards(self):
        cube = self._cube_pos()
        ee = self._ee_pos()
        dist = torch.linalg.norm(cube - ee, dim=-1).detach()
        cube_z = (cube[:, 2] - TABLE_Z).detach()
        near = dist < GRASP_DIST
        lifted = (cube_z > LIFT_HEIGHT) & near
        grip_pos = _t(self.robot.data.joint_pos)[:, self.grip_ids[0]]
        close = ((grip_pos - self.grip_low) / (self.grip_high - self.grip_low)).clamp(0.0, 1.0)
        ee_clearance = (ee[:, 2] - TABLE_Z).detach()
        table_pen = torch.clamp(0.015 - ee_clearance, min=0.0)
        action_rate = (self.actions - self._prev_actions).square().sum(dim=-1)
        reward = (
            3.0 * (1.0 - torch.tanh(5.0 * dist))
            + 1.5 * close * near.float()
            + 8.0 * torch.clamp(cube_z, min=0.0, max=0.18) * near.float()
            + 6.0 * lifted.float()
            - 0.002 * self.actions.square().sum(dim=-1)
            - 0.002 * action_rate
            - 12.0 * table_pen
        ).detach()
        self._prev_actions = self.actions.detach().clone()
        # Always reassign writable clones — inference-mode tensors break later resets.
        self.successes = (self.successes | lifted).detach().clone()
        self.returns = (self.returns + reward).detach().clone()
        self.min_ee_dist = torch.minimum(self.min_ee_dist.detach(), dist).detach().clone()
        self.max_cube_z = torch.maximum(self.max_cube_z.detach(), cube_z).detach().clone()
        return reward

    def _get_dones(self):
        cube = self._cube_pos()
        dropped = cube[:, 2] < (TABLE_Z - 0.12)
        timeout = self.episode_length_buf >= self.max_episode_length - 1
        return dropped, timeout

    def _detach_buffers(self):
        """Ensure episode buffers are normal tensors (not inference-mode)."""
        self.actions = self.actions.detach().clone()
        self._prev_actions = self._prev_actions.detach().clone()
        self._q_arm_cmd = self._q_arm_cmd.detach().clone()
        self._q_grip_cmd = self._q_grip_cmd.detach().clone()
        self.default_q = self.default_q.detach().clone()
        self.successes = self.successes.detach().clone()
        self.returns = self.returns.detach().clone()
        self.min_ee_dist = self.min_ee_dist.detach().clone()
        self.max_cube_z = self.max_cube_z.detach().clone()

    def _reset_idx(self, env_ids):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        self._detach_buffers()
        if len(env_ids):
            self.extras["log"] = {
                "Metrics/lift_success_rate": self.successes[env_ids].float().mean().item(),
                "Metrics/min_ee_cube_m": self.min_ee_dist[env_ids].mean().item(),
                "Metrics/max_cube_height_m": self.max_cube_z[env_ids].mean().item(),
                "Episode/return": self.returns[env_ids].mean().item(),
            }
        self.robot.reset(env_ids)
        super()._reset_idx(env_ids)
        q = self.default_q[env_ids] + 0.02 * torch.randn_like(self.default_q[env_ids])
        zeros = torch.zeros_like(q)
        _write_q(self.robot, q, zeros, self.arm_ids, env_ids)
        open_g = torch.full((len(env_ids), 1), self.grip_low, device=self.device)
        _write_q(self.robot, open_g, torch.zeros_like(open_g), self.grip_ids, env_ids)
        n = len(env_ids)
        cube_pos = torch.zeros(n, 7, device=self.device)
        cube_pos[:, 0] = torch.empty(n, device=self.device).uniform_(0.34, 0.56)
        cube_pos[:, 1] = torch.empty(n, device=self.device).uniform_(-0.18, 0.18)
        cube_pos[:, 2] = TABLE_Z + CUBE_SIZE / 2.0
        cube_pos[:, 3] = 1.0
        cube_pos[:, :3] += self.scene.env_origins[env_ids]
        _write_pose(self.cube, cube_pos, env_ids)
        self.actions[env_ids] = 0
        self._prev_actions[env_ids] = 0
        self._q_arm_cmd[env_ids] = self.default_q[env_ids]
        self._q_grip_cmd[env_ids] = self.grip_low
        self.successes[env_ids] = False
        self.returns[env_ids] = 0
        self.min_ee_dist[env_ids] = 1.0e6
        self.max_cube_z[env_ids] = 0.0


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
    metrics = {"Metrics/lift_success_rate": [], "Metrics/min_ee_cube_m": [], "Metrics/max_cube_height_m": []}
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
        f"episodes={completed} lift_success={summary['Metrics/lift_success_rate']:.3f} "
        f"min_ee_cube_m={summary['Metrics/min_ee_cube_m']:.4f} "
        f"max_cube_height_m={summary['Metrics/max_cube_height_m']:.4f}"
    )
    return summary


def record_isaac_states(env: XArm7LiftEnv, policy, steps: int | None = None) -> Path:
    """Record one env's Isaac joint/cube states for a full-episode non-RTX MuJoCo replay."""
    # Full episode: episode_length_s / (dt * decimation) ≈ 5 / (1/120 * 2) = 300 steps.
    if steps is None:
        steps = int(env.max_episode_length)
    with torch.inference_mode(False):
        obs, _ = env.reset()
    arm, grip, cube = [], [], []
    for _ in range(steps):
        act = (
            _policy_step(policy, obs)
            if policy is not None
            else torch.zeros((env.num_envs, 8), device=env.device)
        )
        with torch.inference_mode(False):
            obs, _, _, _, _ = env.step(act)
        q = _t(env.robot.data.joint_pos)
        arm.append(q[0, env.arm_ids].detach().cpu().numpy())
        grip.append(float(q[0, env.grip_ids[0]].detach().cpu()))
        cube.append((env._cube_pos()[0] - env.scene.env_origins[0]).detach().cpu().numpy())
    out = Path("/tmp/xarm7_isaac_states.npz")
    import numpy as np

    np.savez(out, arm_q=np.stack(arm), gripper=np.asarray(grip), cube_pos=np.stack(cube))
    return out


def render_mujoco_video(states: Path | None, output: Path, label: str, seconds: float | None = None) -> None:
    cmd = [
        SYSTEM_PYTHON,
        str(VIDEO_SCRIPT),
        "--output",
        str(output),
        "--label",
        label,
    ]
    if states is not None:
        cmd.extend(["--states", str(states)])
        # Frame count comes from the NPZ; seconds only matters for cinematic fallback.
        cmd.extend(["--seconds", str(seconds if seconds is not None else 5.0)])
    else:
        cmd.extend(["--seconds", str(seconds if seconds is not None else 5.0)])
    env = os.environ.copy()
    env["MUJOCO_GL"] = "osmesa"
    print("Rendering non-RTX MuJoCo video:", " ".join(cmd))
    subprocess.run(cmd, check=True, env=env, cwd=str(PROJECT_ROOT))


def smoke_test(env: XArm7LiftEnv) -> None:
    print("=== SMOKE TEST (no training) ===")
    obs, _ = env.reset()
    print(f"num_envs={env.num_envs} obs={tuple(obs['policy'].shape)} action={env.action_space}")
    zeros = torch.zeros((env.num_envs, 8), device=env.device)
    for step in range(16):
        obs, rew, term, trunc, info = env.step(zeros)
        if step == 0:
            print(f"reward[0]={float(rew[0]):.4f} cube={env._cube_pos()[0].detach().cpu().numpy()}")
    print("Isaac PhysX env stepped successfully (no RTX camera used).")
    video = PROJECT_ROOT / args.video_dir / "smoke_mujoco_non_rtx.mp4"
    states = record_isaac_states(env, policy=None, steps=90)
    render_mujoco_video(states, video, "smoke / Isaac-state replay", seconds=3.0)
    print(f"SMOKE_OK video={video}")


def dump_milestone_video(env, runner, iteration: int, video_dir: Path, log_dir: Path) -> Path:
    """Always write a full-episode non-RTX MuJoCo MP4 from the current policy."""
    if hasattr(env, "_detach_buffers"):
        env._detach_buffers()
    policy = runner.get_inference_policy(device=env.device)
    states = record_isaac_states(env, policy)  # full episode (~300 steps)
    output = video_dir / f"xarm7_iter_{iteration:04d}_mujoco_non_rtx.mp4"
    render_mujoco_video(states, output, f"iter {iteration} full-episode Isaac-state replay")
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
    cfg = XArm7LiftCfg()
    cfg.scene.num_envs = 1 if args.play else args.num_envs
    cfg.sim.device = args.device or "cuda:0"
    cfg.seed = args.seed
    env = XArm7LiftEnv(cfg, render_mode=None)
    if args.smoke:
        smoke_test(env)
        env.close()
        simulation_app.close()
        return

    runner_cfg = XArm7PPORunnerCfg()
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
