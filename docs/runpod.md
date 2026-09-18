# RunPod notes

The project and its `logs/` and `videos/` folders live at `/workspace/xarm-robot-learning` on the pod volume. Isaac Lab lives at `/root/IsaacLab/env_isaaclab`; that path may disappear if the pod is replaced. The MuJoCo robot model is at `/root/mujoco_menagerie/ufactory_xarm7`.

Use the SSH command shown in the pod's **Connect** tab. On the current pod, direct SSH works. No AWS key is needed for this project; do not put cloud credentials in this repository.

Run tests and training in the foreground so progress and errors stay visible. Videos are saved to `videos/` with MuJoCo's CPU renderer, not Isaac RTX cameras.

For the dual-arm check and training command, see [dual_arm_handoff.md](dual_arm_handoff.md).  
