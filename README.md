# xArm7 cube learning

I'm using xArm7 arms to pick up a cube and, eventually, pass it from one arm to another. The single-arm pickup works. The handoff is still in progress.

There are two trainers in `scripts/`:

- `xarm7_cube_ppo.py` — pick up the cube with one arm.
- `xarm7_dual_arm_handoff_ppo.py` — pick it up with the left arm and pass it to the right.

The stack is Isaac Lab/PhysX for the simulation, RSL-RL (PPO) for training, and MuJoCo for saved videos. I run it on a RunPod GPU.

```text
cube + xArm7 → Isaac Lab → PPO → checkpoints
                         └──→ MuJoCo videos
```

To see the current two-arm motion on a pod that already has Isaac Lab:

```bash
cd /workspace/xarm-robot-learning
/root/IsaacLab/env_isaaclab/bin/python -u scripts/xarm7_dual_arm_handoff_ppo.py --motion_test --num_envs 1
```

The video lands in `videos/`. Right now the giver lifts and carries the cube, but the receiver does not reliably hold it after release. I would check that before spending hours on training. The latest status and training command are in [the handoff notes](docs/dual_arm_handoff.md); pod paths are in [the RunPod notes](docs/runpod.md).
