# Dual-arm handoff

The left arm picks up a cube, carries it toward the middle, and the right arm tries to take it. A single PPO policy controls both arms: seven joints and one gripper per arm. The single-arm trainer is separate and unchanged.

## Current status

The scene runs and the giver can lift and carry the cube. The receiver reaches it, but a reliable physical transfer has **not** been shown yet. Earlier training rewarded lifting too much, so the giver held the cube instead of passing it. That old checkpoint should not be reused. The current reward requires the receiver to keep the cube up after the giver releases, but this has not been validated by training.

To check the motion on the current pod without training:

```bash
cd /workspace/xarm-robot-learning
/root/IsaacLab/env_isaaclab/bin/python -u scripts/xarm7_dual_arm_handoff_ppo.py \
  --motion_test --num_envs 1
```

The test saves `videos/xarm7_dual_motion_test_non_rtx.mp4`. This is a scripted physics test, not a learned policy. The video shows both arms; no Isaac RTX camera or debug marker is used.

## Training

Only start the longer run once the test shows a real transfer. Run it in the foreground, starting fresh:

```bash
cd /workspace/xarm-robot-learning
/root/IsaacLab/env_isaaclab/bin/python -u scripts/xarm7_dual_arm_handoff_ppo.py \
  --num_envs 128 --max_iterations 1000 --video_milestones 500,1000 \
  --log_dir logs/xarm7_dual_handoff_fresh_1000 --video_dir videos
```

The two milestone videos are saved in `videos/`. Rendering each one can take a minute or two while training waits. Keep the pod running until the command finishes.
