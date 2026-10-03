# Migration notes from `robot-Arm`

## Reused directly
- `src/sim/robots/trs_so101/`: SO-ARM101 config, URDF and mesh assets.
- Isaac Lab packaging/dependency setup from the original `pyproject.toml`.
- The original project's joint-target control convention as the initial robot-control assumption.

## Replaced
- Joystick/Pong task -> `SOArm-PickPlace-v0`.
- Direction-command rewards -> reach/lift/place/success signals.
- Teleoperation/SmolVLA data source -> personally recorded human hand demonstrations.

## Deliberately not copied
- trained checkpoints under `final/`;
- `src/ale/` Pong/game integration;
- arcade-stick USD/textures;
- joystick-specific training, reward and buffer code.

## First engineering milestones
1. Launch `SOArm-PickPlace-v0` locally and tune table/object/goal transforms.
2. Implement a stable action term (joint target first; differential IK later if useful).
3. Record one hand pick-and-place video.
4. Emit `HumanFrame` trajectory data.
5. Calibrate human workspace -> robot workspace.
6. Retarget the trajectory and replay it in Isaac.
7. Only then add BC/RL and multi-object experiments.
