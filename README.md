# xArm7 robot learning

Isaac Lab + PPO on a RunPod GPU. One xArm7 picks up a cube. Two arms try to pass it.

Isaac is at `/root/IsaacLab`. This repo lives at `/workspace/xarm-robot-learning`. Videos are MuJoCo (CPU), not Isaac cameras.

```
scripts/xarm7_cube_ppo.py              one arm
scripts/xarm7_dual_arm_handoff_ppo.py  two arms
```

Checkpoints go in `logs/`. Videos go in `videos/`.

## Dual-arm

Needs the current `scripts/xarm7_dual_arm_handoff_ppo.py` on the pod. Do not resume old dual-arm checkpoints.

```bash
cd /workspace/xarm-robot-learning
export OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y

# short run — left arm should go for the cube
/root/IsaacLab/env_isaaclab/bin/python -u scripts/xarm7_dual_arm_handoff_ppo.py \
  --num_envs 16 --max_iterations 20 --video_milestones 20 \
  --log_dir logs/xarm7_dual_handoff_mini --video_dir videos

# longer run
/root/IsaacLab/env_isaaclab/bin/python -u scripts/xarm7_dual_arm_handoff_ppo.py \
  --num_envs 128 --max_iterations 1000 --video_milestones 500,1000 \
  --log_dir logs/xarm7_dual_handoff_1000 --video_dir videos
```

Keep it in the foreground. After Isaac starts you should see iteration lines. A milestone video can sit on the first-frame RGB line for a minute while it encodes — that is not a crash.

Pod SSH and install notes: [docs/runpod.md](docs/runpod.md).
