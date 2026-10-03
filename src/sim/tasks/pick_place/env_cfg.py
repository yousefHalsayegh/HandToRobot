"""First challenge environment: SO-ARM101 pick-and-place.

Reuses the SO-ARM definition from the MSc repository and replaces the joystick
scene with a cube + goal manipulation scene. Object/table transforms should be
validated in the user's local Isaac Lab installation before training.
"""
from dataclasses import MISSING

import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import ActionTerm, ActionTermCfg as ActTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg, OffsetCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg
from isaaclab.utils import configclass

from sim.robots import SO_ARM101_CFG
import sim.tasks.pick_place.mdp as mdp


@configclass
class PickPlaceSceneCfg(InteractiveSceneCfg):
    robot: ArticulationCfg = MISSING
    ee_frame: FrameTransformerCfg = MISSING

    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.45, 0.0, -0.02)),
        spawn=sim_utils.CuboidCfg(
            size=(0.70, 0.60, 0.04),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        ),
    )
    object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.30, 0.0, 0.04)),
        spawn=sim_utils.CuboidCfg(
            size=(0.04, 0.04, 0.04),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
    )
    goal = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Goal",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.45, 0.15, 0.025)),
        spawn=sim_utils.CylinderCfg(
            radius=0.07,
            height=0.01,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
        ),
    )
    ground = AssetBaseCfg(prim_path="/World/Ground", spawn=GroundPlaneCfg())
    light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=2500.0)
    )


class JointTargetActionTerm(ActionTerm):
    """Six joint-position targets, matching the original repository convention."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._actions = torch.zeros((env.num_envs, 6), device=env.device)

    @property
    def action_dim(self) -> int:
        return 6

    @property
    def raw_actions(self):
        return self._actions

    @property
    def processed_actions(self):
        return self._actions

    def process_actions(self, actions: torch.Tensor):
        self._actions = actions

    def apply_actions(self):
        robot = self._env.scene[self.cfg.asset_name]
        robot.set_joint_position_target(self._actions.float())
        robot.write_data_to_sim()


@configclass
class JointTargetActionCfg(ActTerm):
    class_type: type = JointTargetActionTerm
    asset_name: str = "robot"


@configclass
class ActionsCfg:
    joint_targets = JointTargetActionCfg(asset_name="robot")


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(func=mdp.joint_positions)
        object_pos = ObsTerm(func=mdp.object_position_w)
        ee_pos = ObsTerm(func=mdp.ee_position_w)
        object_to_goal = ObsTerm(func=mdp.object_to_goal)

        def __post_init__(self):
            self.concatenate_terms = True
            self.enable_corruption = False

    policy: PolicyCfg = PolicyCfg()


@configclass
class RewardsCfg:
    reach = RewTerm(func=mdp.reaching_reward, weight=1.0)
    lift = RewTerm(func=mdp.lift_reward, weight=2.0)
    place = RewTerm(func=mdp.place_reward, weight=4.0)


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(func=mdp.object_in_goal)


@configclass
class SOArmPickPlaceEnvCfg(ManagerBasedRLEnvCfg):
    scene: PickPlaceSceneCfg = PickPlaceSceneCfg(num_envs=1, env_spacing=1.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = SO_ARM101_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base_link",
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/gripper_link",
                    name="end_effector",
                    offset=OffsetCfg(pos=(0.01, 0.0, -0.09)),
                )
            ],
        )
        self.decimation = 4
        self.episode_length_s = 8.0
        self.sim.dt = 1.0 / 120.0
        self.sim.render_interval = self.decimation
