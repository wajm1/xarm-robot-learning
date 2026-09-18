# xArm robot learning

Two small robot-learning experiments with xArm7 arms and a red cube.

- **Single-arm pickup:** one arm reaches, grips, and lifts the cube. This task has trained successfully.
- **Dual-arm handoff:** one arm picks up the cube and passes it to a second arm. This is still being tested; a complete transfer has not been verified yet.

## How it works

```text
cube + robot(s) in Isaac Lab
          ↓
   physics simulation
          ↓
 PPO learns arm and gripper movements
          ↓
 checkpoints, metrics, and review videos
```

Isaac Lab and PhysX simulate the robots. RSL-RL runs PPO training. We render saved videos with MuJoCo and OSMesa because the pod's Isaac camera output was blank. Training runs on RunPod.

## Files

| File | Purpose |
| --- | --- |
| `scripts/xarm7_cube_ppo.py` | Single-arm training |
| `scripts/xarm7_dual_arm_handoff_ppo.py` | Dual-arm training and motion test |
| `scripts/*mujoco_video.py` | Video rendering |
| `docs/dual_arm_handoff.md` | Dual-arm status and commands |
| `docs/runpod.md` | Pod paths and notes |

`logs/` and `videos/` hold training output and are not committed to Git.

## Try the dual-arm scene

On a pod with Isaac Lab already installed:

```bash
cd /workspace/xarm-robot-learning
/root/IsaacLab/env_isaaclab/bin/python -u scripts/xarm7_dual_arm_handoff_ppo.py --motion_test --num_envs 1
```

This saves `videos/xarm7_dual_motion_test_non_rtx.mp4`. It is a scripted physics check, not a trained policy. Check that the cube really transfers before starting a long training run. See [the dual-arm notes](docs/dual_arm_handoff.md) for the current result and training command.
