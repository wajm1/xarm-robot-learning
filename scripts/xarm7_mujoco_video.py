#!/usr/bin/env python3
"""CPU-offscreen MuJoCo showcase video of the official UFACTORY xArm7 mesh.

Isaac's Vulkan/RTX camera path is broken on this pod (black frames). This
renderer is the honest fallback: Menagerie xArm7 + gripper, table, cube,
OSMesa/CPU offscreen, labeled non-RTX. It never claims to be an Isaac render.

Workcell layout matches Isaac Lab training: robot base mounted on the table
top (TABLE_Z), cube on the table, arm starts from the Menagerie home pose.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

# Must be set before importing mujoco.
os.environ.setdefault("MUJOCO_GL", "osmesa")

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

MENAGERIE = Path("/root/mujoco_menagerie/ufactory_xarm7")
ARM_JOINTS = [f"joint{i}" for i in range(1, 8)]
GRIPPER_JOINT = "left_driver_joint"
# Same table height as scripts/xarm7_cube_ppo.py (Isaac mounts the USD at this Z).
TABLE_Z = 0.44
CUBE_HALF = 0.0275
# Official Menagerie home (arm joints only).
HOME_Q = np.array([0.0, -0.247, 0.0, 0.909, 0.0, 1.15644, 0.0], dtype=np.float64)
CAPTION = "MuJoCo CPU-offscreen (non-RTX)  |  xArm7 mounted on table  |  home pose"
BLACK_MEAN = 8.0


def _mounted_robot_xml() -> str:
    """Return xarm7.xml with the base mounted on the table top."""
    raw = (MENAGERIE / "xarm7.xml").read_text()
    # Drop the outer <mujoco> wrapper so we can compose a workcell around it.
    raw = re.sub(r"^\s*<mujoco[^>]*>", "", raw, count=1)
    raw = re.sub(r"</mujoco>\s*$", "", raw, count=1)
    # Isaac places the USD root on the table top. Menagerie's stock z=0.12 is only a
    # floor pedestal offset — do NOT add it again or the base floats above the table.
    raw, n = re.subn(
        r'(<body name="link_base"\s+pos=")[^"]+(")',
        rf"\g<1>0 0 {TABLE_Z:.6f}\2",
        raw,
        count=1,
    )
    if n != 1:
        raise RuntimeError("Failed to remount Menagerie link_base onto the table.")
    return raw


def _scene_xml() -> str:
    robot = _mounted_robot_xml()
    # Match Isaac workcell: table size (0.80, 0.70, 0.08), center under/around the base.
    table_half = (0.40, 0.35, 0.04)
    table_pos = (0.30, 0.0, TABLE_Z - table_half[2])
    cube_z = TABLE_Z + CUBE_HALF
    return f"""
<mujoco model="xarm7_cube_workcell">
  <compiler angle="radian" autolimits="true" meshdir="assets"/>
  <option integrator="implicitfast" timestep="0.002"/>
  <statistic center="0.30 0.0 {TABLE_Z + 0.22:.3f}" extent="1.0"/>
  <visual>
    <headlight diffuse="0.55 0.55 0.55" ambient="0.25 0.28 0.32" specular="0.1 0.1 0.1"/>
    <global azimuth="145" elevation="-18" offwidth="1280" offheight="720"/>
  </visual>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.16 0.22 0.30" rgb2="0.02 0.03 0.04" width="512" height="512"/>
    <texture name="grid" type="2d" builtin="checker" rgb1="0.18 0.20 0.22" rgb2="0.12 0.13 0.15" width="256" height="256"/>
    <material name="grid" texture="grid" texuniform="true" texrepeat="8 8" reflectance="0.05"/>
    <material name="table" rgba="0.10 0.12 0.15 1"/>
    <material name="cube" rgba="0.82 0.09 0.06 1"/>
  </asset>
{robot}
  <worldbody>
    <light pos="0.4 -0.6 1.8" dir="-0.1 0.3 -1" diffuse="0.85 0.80 0.70"/>
    <geom name="floor" type="plane" size="2 2 0.05" material="grid"/>
    <geom name="table" type="box" size="{table_half[0]} {table_half[1]} {table_half[2]}"
          pos="{table_pos[0]} {table_pos[1]} {table_pos[2]:.4f}" material="table"/>
    <body name="cube" pos="0.45 0.0 {cube_z:.4f}">
      <freejoint name="cube_free"/>
      <geom name="cube" type="box" size="{CUBE_HALF} {CUBE_HALF} {CUBE_HALF}" material="cube" mass="0.075"
            friction="1.1 0.05 0.001"/>
    </body>
    <camera name="showcase" pos="1.18 -1.10 1.08" xyaxes="0.80 0.60 0 -0.14 0.19 0.97"/>
  </worldbody>
</mujoco>
"""


def _overlay(frame: np.ndarray, extra: str) -> np.ndarray:
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    w, h = img.size
    draw.rectangle([0, h - 40, w, h], fill=(18, 18, 22))
    font = ImageFont.load_default()
    draw.text((12, h - 28), f"{CAPTION}  |  {extra}", fill=(240, 214, 70), font=font)
    return np.asarray(img)


def _validate(frame: np.ndarray) -> None:
    mean = float(frame.mean())
    if mean < BLACK_MEAN:
        raise RuntimeError(
            f"Refusing to write a black/blank video (mean RGB={mean:.2f}). "
            "This is not an Isaac RTX capture."
        )


def _apply_state(model: mujoco.MjModel, data: mujoco.MjData, arm_q, gripper, cube_pos) -> None:
    for i, name in enumerate(ARM_JOINTS):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        data.qpos[model.jnt_qposadr[jid]] = float(arm_q[i])
    gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, GRIPPER_JOINT)
    # Isaac drive_joint is typically negative=open; Menagerie uses 0 open → 0.85 closed.
    g = float(gripper)
    if g < 0.0:
        # Map common Isaac open range (~-0.85..0) into Menagerie.
        g = float(np.clip((-g) / 0.85, 0.0, 1.0) * 0.85)
    else:
        g = float(np.clip(g, 0.0, 0.85))
    data.qpos[model.jnt_qposadr[gid]] = g
    cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "cube_free")
    adr = model.jnt_qposadr[cid]
    cube = np.asarray(cube_pos, dtype=np.float64).copy()
    # Keep the cube on the table surface if a rollout only provides XY.
    if cube[2] < TABLE_Z:
        cube[2] = TABLE_Z + CUBE_HALF
    data.qpos[adr : adr + 3] = cube
    data.qpos[adr + 3 : adr + 7] = np.array([1.0, 0.0, 0.0, 0.0])
    mujoco.mj_forward(model, data)


def _smooth_traj(arm_q: np.ndarray, gripper: np.ndarray, cube_pos: np.ndarray, win: int = 7):
    """Causal-ish moving average to remove jitter in showcase replays."""
    n = len(arm_q)
    w = max(1, int(win))
    arm_s = np.empty_like(arm_q)
    grip_s = np.empty_like(gripper)
    cube_s = np.empty_like(cube_pos)
    for i in range(n):
        lo = max(0, i - w + 1)
        arm_s[i] = arm_q[lo : i + 1].mean(axis=0)
        grip_s[i] = gripper[lo : i + 1].mean()
        cube_s[i] = cube_pos[lo : i + 1].mean(axis=0)
    return arm_s, grip_s, cube_s


def _cinematic(model, data, step: int, total: int):
    """Gentle sweep around the Menagerie home pose; base stays table-mounted."""
    t = step / max(total - 1, 1)
    sweep = np.array(
        [
            0.18 * np.sin(1.6 * t),
            -0.08 * t,
            0.08 * np.sin(1.4 * t),
            0.06 * t,
            0.05 * np.sin(1.8 * t),
            -0.06 * t,
            0.20 * np.sin(1.2 * t),
        ]
    )
    grip = 0.05 + 0.55 * t
    cube = (0.45 + 0.03 * np.sin(1.5 * t), 0.06 * np.cos(1.2 * t), TABLE_Z + CUBE_HALF)
    _apply_state(model, data, HOME_Q + sweep, grip, cube)


def main() -> None:
    parser = argparse.ArgumentParser(description="Record a labeled non-RTX xArm7 MuJoCo MP4.")
    parser.add_argument("--output", default="videos/xarm7_mujoco_preview.mp4")
    parser.add_argument("--states", default=None, help="Optional JSON/NPZ rollout from Isaac Lab.")
    parser.add_argument("--seconds", type=float, default=4.0)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--label", default="")
    parser.add_argument(
        "--from_home",
        action="store_true",
        help="Ignore trajectory motion and hold the Menagerie home pose (still table-mounted).",
    )
    args = parser.parse_args()

    if not (MENAGERIE / "xarm7.xml").is_file():
        raise FileNotFoundError(f"Missing Menagerie xArm7 model at {MENAGERIE}")

    out = Path(args.output).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    states_path = Path(args.states).expanduser().resolve() if args.states else None

    os.chdir(MENAGERIE)
    model = mujoco.MjModel.from_xml_string(_scene_xml())
    data = mujoco.MjData(model)
    # Start at home before any frames.
    _apply_state(model, data, HOME_Q, 0.0, (0.45, 0.0, TABLE_Z + CUBE_HALF))
    renderer = mujoco.Renderer(model, height=720, width=1280)

    traj = None
    if states_path is not None and not args.from_home:
        if states_path.suffix == ".npz":
            packed = np.load(states_path)
            traj = {k: packed[k] for k in packed.files}
        else:
            traj = json.loads(states_path.read_text())
        if traj is not None and "arm_q" in traj:
            # Isaac resets the cube at episode end; do not show that teleport as motion.
            cube_raw = np.asarray(traj["cube_pos"])
            jumps = np.linalg.norm(np.diff(cube_raw, axis=0), axis=1)
            resets = np.flatnonzero(
                (jumps > 0.25)
                & (cube_raw[1:, 2] < TABLE_Z + CUBE_HALF + 0.02)
            )
            if len(resets) and resets[0] >= len(cube_raw) - 4:
                keep = int(resets[0]) + 1
                print(f"Trimming final episode-reset frame(s): {len(cube_raw)} -> {keep}")
                traj = {key: np.asarray(traj[key])[:keep] for key in ("arm_q", "gripper", "cube_pos")}
            arm_s, grip_s, cube_s = _smooth_traj(
                np.asarray(traj["arm_q"]),
                np.asarray(traj["gripper"]),
                np.asarray(traj["cube_pos"]),
                win=7,
            )
            traj = {"arm_q": arm_s, "gripper": grip_s, "cube_pos": cube_s}

    nframes = int(args.seconds * args.fps)
    if traj is not None and "arm_q" in traj:
        nframes = len(traj["arm_q"])
    extra = args.label or ("Isaac-state replay" if traj else "home + light sweep")
    validated = False
    with imageio.get_writer(out, fps=args.fps, macro_block_size=16, codec="libx264", pixelformat="yuv420p") as writer:
        for i in range(nframes):
            if args.from_home:
                _apply_state(model, data, HOME_Q, 0.0, (0.45, 0.0, TABLE_Z + CUBE_HALF))
            elif traj is not None and "arm_q" in traj:
                cube = traj["cube_pos"][i]
                _apply_state(model, data, traj["arm_q"][i], traj["gripper"][i], cube)
            else:
                _cinematic(model, data, i, nframes)
            renderer.update_scene(data, camera="showcase")
            frame = _overlay(renderer.render(), extra)
            if not validated:
                _validate(frame)
                validated = True
                print(f"First-frame mean RGB={float(frame.mean()):.1f} (not black).")
                print(f"Table top Z={TABLE_Z}; robot base flush on table at Z={TABLE_Z}")
            writer.append_data(frame)

    print(f"MUJOCO_NON_RTX_VIDEO {out}")
    print(f"frames={nframes} renderer=OSMesa/CPU mounted_on_table=1 home={HOME_Q.tolist()}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
