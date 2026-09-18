#!/usr/bin/env python3
"""CPU-offscreen MuJoCo video for the dual xArm7 handoff scene.

Renders BOTH arms, table, and cube from an Isaac state NPZ.
Labeled non-RTX / OSMesa. Does not use Isaac cameras.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "osmesa")

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

MENAGERIE = Path("/root/mujoco_menagerie/ufactory_xarm7")
ARM_JOINTS = [f"joint{i}" for i in range(1, 8)]
GRIPPER_JOINT = "left_driver_joint"
TABLE_Z = 0.44
CUBE_HALF = 0.0275
GIVER_XY = (0.0, 0.0)
RECV_XY = (1.10, 0.0)
TABLE_HALF = (0.75, 0.40, 0.04)
TABLE_CENTER = (0.55, 0.0, TABLE_Z - TABLE_HALF[2])
HOME_Q = np.array([0.0, -0.247, 0.0, 0.909, 0.0, 1.15644, 0.0], dtype=np.float64)
CAPTION = "MuJoCo CPU-offscreen (non-RTX)  |  dual xArm7 handoff"
BLACK_MEAN = 8.0


def _extract_balanced(raw: str, open_tag: str, close_tag: str) -> str:
    start = raw.find(open_tag)
    if start < 0:
        raise RuntimeError(f"Missing {open_tag}")
    depth = 0
    i = start
    open_prefix = open_tag[:-1]  # "<default" / "<asset" / ...
    while i < len(raw):
        if raw.startswith(open_prefix, i) and (raw[i + len(open_prefix)] in " >"):
            depth += 1
            i = raw.find(">", i) + 1
            continue
        if raw.startswith(close_tag, i):
            depth -= 1
            i += len(close_tag)
            if depth == 0:
                return raw[start:i]
            continue
        i += 1
    raise RuntimeError(f"Unbalanced {open_tag}")


def _inner(block: str, tag: str) -> str:
    return re.sub(rf"^<{tag}>|</{tag}>$", "", block, flags=re.S).strip()


def _parse_menagerie() -> tuple[str, str, str, str, str, str, str]:
    """Return (defaults, materials, meshes, worldbody, tendons, actuators, equality)."""
    raw = (MENAGERIE / "xarm7.xml").read_text()
    raw = re.sub(r"^\s*<mujoco[^>]*>", "", raw, count=1)
    raw = re.sub(r"</mujoco>\s*$", "", raw, count=1)
    defaults = _extract_balanced(raw, "<default>", "</default>")
    asset = _extract_balanced(raw, "<asset>", "</asset>")
    world = _extract_balanced(raw, "<worldbody>", "</worldbody>")
    try:
        tendon_inner = _inner(_extract_balanced(raw, "<tendon>", "</tendon>"), "tendon")
    except RuntimeError:
        tendon_inner = ""
    try:
        actuator_inner = _inner(_extract_balanced(raw, "<actuator>", "</actuator>"), "actuator")
    except RuntimeError:
        actuator_inner = ""
    try:
        equality_inner = _inner(_extract_balanced(raw, "<equality>", "</equality>"), "equality")
    except RuntimeError:
        equality_inner = ""
    asset_inner = _inner(asset, "asset")
    world_inner = _inner(world, "worldbody")
    materials = "\n".join(re.findall(r"<material\b[^/]*/>", asset_inner))
    meshes = "\n".join(re.findall(r"<mesh\b[^/]*/>", asset_inner))
    return defaults, materials, meshes, world_inner, tendon_inner, actuator_inner, equality_inner


def _prefix_robot(
    name: str,
    meshes: str,
    world: str,
    tendon: str,
    actuator: str,
    equality: str,
    pos: str,
    quat: str | None,
) -> tuple[str, str, str, str, str]:
    """Prefix per-arm mesh/body/joint/tendon names. Keep shared class/material names untouched."""

    def pref_meshes(s: str) -> str:
        def repl(m: re.Match[str]) -> str:
            attrs = m.group(1)
            if re.search(r'\bname="', attrs):
                attrs = re.sub(r'\bname="([^"]+)"', rf'name="{name}_\1"', attrs, count=1)
            else:
                fm = re.search(r'\bfile="([^"]+)"', attrs)
                stem = Path(fm.group(1)).stem if fm else "mesh"
                attrs = f' name="{name}_{stem}"' + attrs
            return f"<mesh{attrs}/>"

        return re.sub(r"<mesh([^>]*?)/>", repl, s)

    def pref(s: str) -> str:
        for tag in ("body", "joint", "geom", "site", "tendon", "fixed", "motor", "position", "general"):
            s = re.sub(rf'(<{tag}\b[^>]*\bname=")([^"]+)(")', rf"\1{name}_\2\3", s)
        s = re.sub(r'(mesh=")([^"]+)(")', rf"\1{name}_\2\3", s)
        s = re.sub(r'\bjoint="([^"]+)"', rf'joint="{name}_\1"', s)
        s = re.sub(r'\bjoint1="([^"]+)"', rf'joint1="{name}_\1"', s)
        s = re.sub(r'\bjoint2="([^"]+)"', rf'joint2="{name}_\1"', s)
        s = re.sub(r'\bbody1="([^"]+)"', rf'body1="{name}_\1"', s)
        s = re.sub(r'\bbody2="([^"]+)"', rf'body2="{name}_\1"', s)
        s = re.sub(r'\bsite1="([^"]+)"', rf'site1="{name}_\1"', s)
        s = re.sub(r'\bsite2="([^"]+)"', rf'site2="{name}_\1"', s)
        s = re.sub(r'\btendon="([^"]+)"', rf'tendon="{name}_\1"', s)
        s = re.sub(r'\bsite="([^"]+)"', rf'site="{name}_\1"', s)
        return s

    meshes = pref_meshes(meshes)
    world = pref(world)
    tendon = pref(tendon)
    actuator = pref(actuator)
    equality = pref(equality)
    quat_attr = f' quat="{quat}"' if quat else ""
    world, n = re.subn(
        rf'(<body name="{name}_link_base"\s+pos=")[^"]+(")',
        rf"\g<1>{pos}\2{quat_attr}",
        world,
        count=1,
    )
    if n != 1:
        world, n = re.subn(
            rf'(<body name="{name}_link_base")(\s*)',
            rf'\1 pos="{pos}"{quat_attr}\2',
            world,
            count=1,
        )
    if n != 1:
        raise RuntimeError(f"Failed to place {name} link_base")
    return meshes, world, tendon, actuator, equality


def _scene_xml() -> str:
    defaults, materials, meshes, world, tendon, actuator, equality = _parse_menagerie()
    g_mesh, g_world, g_ten, g_act, g_eq = _prefix_robot(
        "giver", meshes, world, tendon, actuator, equality, f"{GIVER_XY[0]} {GIVER_XY[1]} {TABLE_Z:.6f}", None
    )
    # MuJoCo quat is w x y z. 180 deg about Z: (0, 0, 0, 1).
    r_mesh, r_world, r_ten, r_act, r_eq = _prefix_robot(
        "recv", meshes, world, tendon, actuator, equality, f"{RECV_XY[0]} {RECV_XY[1]} {TABLE_Z:.6f}", "0 0 0 1"
    )
    cube_z = TABLE_Z + CUBE_HALF
    return f"""
<mujoco model="xarm7_dual_handoff">
  <compiler angle="radian" autolimits="true" meshdir="assets"/>
  <option integrator="implicitfast" timestep="0.002"/>
  <statistic center="0.55 0.0 {TABLE_Z + 0.28:.3f}" extent="1.55"/>
  <visual>
    <headlight diffuse="0.55 0.55 0.55" ambient="0.25 0.28 0.32" specular="0.1 0.1 0.1"/>
    <global azimuth="90" elevation="-32" offwidth="1280" offheight="720"/>
  </visual>
  {defaults}
  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.16 0.22 0.30" rgb2="0.02 0.03 0.04" width="512" height="512"/>
    <texture name="grid" type="2d" builtin="checker" rgb1="0.18 0.20 0.22" rgb2="0.12 0.13 0.15" width="256" height="256"/>
    <material name="grid" texture="grid" texuniform="true" texrepeat="8 8" reflectance="0.05"/>
    <material name="table" rgba="0.10 0.12 0.15 1"/>
    <material name="cube" rgba="0.82 0.09 0.06 1"/>
    {materials}
    {g_mesh}
    {r_mesh}
  </asset>
  <worldbody>
    <light pos="0.55 -1.0 2.3" dir="-0.05 0.35 -1" diffuse="0.85 0.80 0.70"/>
    <geom name="floor" type="plane" size="3.0 3.0 0.05" material="grid"/>
    <geom name="table" type="box" size="{TABLE_HALF[0]} {TABLE_HALF[1]} {TABLE_HALF[2]}"
          pos="{TABLE_CENTER[0]} {TABLE_CENTER[1]} {TABLE_CENTER[2]:.4f}" material="table"/>
    <body name="cube" pos="0.22 0.0 {cube_z:.4f}">
      <freejoint name="cube_free"/>
      <geom name="cube" type="box" size="{CUBE_HALF} {CUBE_HALF} {CUBE_HALF}" material="cube" mass="0.075"
            friction="1.1 0.05 0.001"/>
    </body>
    {g_world}
    {r_world}
    <!-- Front view aimed at table center: lower cam, mild downward look. -->
    <camera name="showcase" pos="0.55 -2.0 1.55" xyaxes="1 0 0 0 0.42 0.907"/>
  </worldbody>
  <tendon>
    {g_ten}
    {r_ten}
  </tendon>
  <actuator>
    {g_act}
    {r_act}
  </actuator>
  <equality>
    {g_eq}
    {r_eq}
  </equality>
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
        raise RuntimeError(f"Black/blank frame (mean RGB={mean:.2f}). Refusing to write.")


def _map_grip(g: float) -> float:
    g = float(g)
    if g < 0.0:
        return float(np.clip((-g) / 0.85, 0.0, 1.0) * 0.85)
    return float(np.clip(g, 0.0, 0.85))


def _apply_state(model, data, giver_q, giver_g, recv_q, recv_g, cube_pos) -> None:
    for i, jn in enumerate(ARM_JOINTS):
        for prefix, q in (("giver", giver_q), ("recv", recv_q)):
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{prefix}_{jn}")
            data.qpos[model.jnt_qposadr[jid]] = float(q[i])
    for prefix, g in (("giver", giver_g), ("recv", recv_g)):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{prefix}_{GRIPPER_JOINT}")
        data.qpos[model.jnt_qposadr[jid]] = _map_grip(g)
    cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "cube_free")
    adr = model.jnt_qposadr[cid]
    cube = np.asarray(cube_pos, dtype=np.float64).copy()
    if cube[2] < TABLE_Z:
        cube[2] = TABLE_Z + CUBE_HALF
    data.qpos[adr : adr + 3] = cube
    data.qpos[adr + 3 : adr + 7] = np.array([1.0, 0.0, 0.0, 0.0])
    mujoco.mj_forward(model, data)


def _smooth(arr: np.ndarray, win: int = 7) -> np.ndarray:
    n = len(arr)
    out = np.empty_like(arr)
    for i in range(n):
        lo = max(0, i - win + 1)
        out[i] = arr[lo : i + 1].mean(axis=0)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Dual-arm non-RTX MuJoCo MP4.")
    parser.add_argument("--output", default="videos/xarm7_dual_handoff_preview.mp4")
    parser.add_argument("--states", default=None)
    parser.add_argument("--seconds", type=float, default=4.0)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--label", default="")
    args = parser.parse_args()

    if not (MENAGERIE / "xarm7.xml").is_file():
        raise FileNotFoundError(f"Missing Menagerie at {MENAGERIE}")

    out = Path(args.output).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    os.chdir(MENAGERIE)
    xml = _scene_xml()
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    _apply_state(model, data, HOME_Q, 0.0, HOME_Q, 0.0, (0.20, 0.0, TABLE_Z + CUBE_HALF))
    renderer = mujoco.Renderer(model, height=720, width=1280)

    traj = None
    if args.states:
        path = Path(args.states)
        if path.suffix == ".npz":
            packed = np.load(path)
            traj = {k: packed[k] for k in packed.files}
        else:
            traj = json.loads(path.read_text())
        # require both arms
        for key in ("giver_arm_q", "recv_arm_q", "cube_pos"):
            if key not in traj:
                raise RuntimeError(f"Dual-arm NPZ missing '{key}'. Refusing single-arm fallback.")
        traj["giver_arm_q"] = _smooth(np.asarray(traj["giver_arm_q"]))
        traj["recv_arm_q"] = _smooth(np.asarray(traj["recv_arm_q"]))
        traj["giver_gripper"] = _smooth(np.asarray(traj["giver_gripper"]))
        traj["recv_gripper"] = _smooth(np.asarray(traj["recv_gripper"]))
        traj["cube_pos"] = _smooth(np.asarray(traj["cube_pos"]))

    nframes = int(args.seconds * args.fps)
    if traj is not None:
        keys = ("giver_arm_q", "recv_arm_q", "giver_gripper", "recv_gripper", "cube_pos")
        raw_frames = len(traj["giver_arm_q"])
        target_frames = max(1, int(args.seconds * args.fps))
        stride = max(1, int(round(raw_frames / target_frames))) if raw_frames > target_frames else 1
        if stride > 1:
            for key in keys:
                traj[key] = np.asarray(traj[key])[::stride]
        nframes = len(traj["giver_arm_q"])
        print(f"Encoding {nframes}/{raw_frames} PhysX frames at {args.fps} fps with OSMesa.", flush=True)
    extra = args.label or ("Isaac dual-arm replay" if traj else "dual-arm home")
    validated = False
    with imageio.get_writer(
        out, fps=args.fps, macro_block_size=16, codec="libx264", pixelformat="yuv420p",
        ffmpeg_params=["-preset", "ultrafast", "-crf", "23"],
    ) as writer:
        for i in range(nframes):
            if traj is not None:
                _apply_state(
                    model, data,
                    traj["giver_arm_q"][i], traj["giver_gripper"][i],
                    traj["recv_arm_q"][i], traj["recv_gripper"][i],
                    traj["cube_pos"][i],
                )
            else:
                _apply_state(model, data, HOME_Q, 0.0, HOME_Q, 0.0, (0.20, 0.0, TABLE_Z + CUBE_HALF))
            renderer.update_scene(data, camera="showcase")
            frame = _overlay(renderer.render(), extra)
            if not validated:
                _validate(frame)
                validated = True
                print(f"First-frame mean RGB={float(frame.mean()):.1f} (not black).", flush=True)
                # crude check both bases exist
                for b in ("giver_link_base", "recv_link_base"):
                    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, b)
                    if bid < 0:
                        raise RuntimeError(f"Missing body {b} in dual scene")
                    print(f"  {b} z={data.xpos[bid, 2]:.3f}", flush=True)
            writer.append_data(frame)
            if (i + 1) % 30 == 0 or i + 1 == nframes:
                print(f"  encoded {i + 1}/{nframes} frames", flush=True)

    print(f"DUAL_MUJOCO_NON_RTX_VIDEO {out} frames={nframes}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
