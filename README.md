# xArm7 robot learning

Isaac Lab + PPO on a RunPod GPU. One xArm7's goal is to pick up the cube and the other xArm tries to grab it.

## Demo

[Watch the xArm7 dual-arm motion test]

https://github.com/user-attachments/assets/7f7c5847-fb4c-46bb-a40d-63382f9e82bc



Videos are MuJoCo


```
scripts/xarm7_cube_ppo.py              arm 1
scripts/xarm7_dual_arm_handoff_ppo.py  arm 2
```

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

Pod SSH and install notes: [docs/runpod.md](docs/runpod.md).
