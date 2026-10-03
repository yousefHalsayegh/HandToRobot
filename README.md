# Human Hand → SO-ARM Manipulation

Fork-ready challenge repository derived from the existing `robot-Arm` project.

## Goal
Translate personally collected human-hand demonstrations into SO-ARM101
manipulation trajectories in Isaac Lab, beginning with pick-and-place and
expanding to varied objects/tasks.

## Core pipeline
```text
human video
    -> hand + object tracking
    -> canonical HumanFrame trajectory
    -> workspace calibration / embodiment retargeting
    -> SO-ARM end-effector + gripper targets
    -> Isaac Lab replay
    -> dataset / imitation learning / optional RL refinement
```

## What came from the original project
The SO-ARM101 configuration, URDF/assets, Isaac Lab dependency stack and the
robot-control conventions are retained. Pong/ALE, joystick assets, joystick
rewards and historical checkpoints are intentionally excluded from the active
challenge repo.

## Repository map
```text
src/
  sim/
    robots/trs_so101/       # reused SO-ARM model
    tasks/pick_place/       # new manipulation task
  human_data/               # new human data + retargeting layer
data/
  raw_human/
  processed_human/
  robot_demos/
experiments/
results/
```

## First task: `SOArm-PickPlace-v0`
The first scene contains the SO-ARM, a table, one cube and one goal region.
It is deliberately minimal so the first milestone is trajectory transfer,
not policy complexity.

## Status
This scaffold is a migration starting point, not a finished training stack.
The SO-ARM asset is carried over from the original repository. The pick/place
scene and hand-data interfaces are new and should be validated against the
local Isaac Lab 2.3 / Isaac Sim 5.1 installation before training.

See `MIGRATION_NOTES.md` for the build order.
