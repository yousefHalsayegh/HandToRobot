"""SO-ARM101 pick-and-place environment.

Key changes vs the original fork:
  - cube at (0.20, -0.08) and goal at (0.20, 0.12)  →  both inside reach
  - EE frame targets gripper_frame_link (proper tool frame from URDF)
  - front camera added (256x256, matching SmolVLA/ACT training resolution)
  - cube and goal reset with uniform randomisation every episode
  - camera observation added to the policy group
"""
from dataclasses import MISSING

import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import ActionTerm, ActionTermCfg as ActTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import TiledCameraCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg, OffsetCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg
from isaaclab.utils import configclass
import isaaclab.envs.mdp as mdp

from sim.robots import SO_ARM101_CFG
import sim.tasks.pick_place.mdp as pick_mdp


@configclass
class PickPlaceSceneCfg(InteractiveSceneCfg):
    robot: ArticulationCfg = MISSING
    ee_frame: FrameTransformerCfg = MISSING

    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.35, 0.0, -0.02)),
        spawn=sim_utils.CuboidCfg(
            size=(0.50, 0.40, 0.04),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        ),
    )
    object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.20, -0.08, 0.04),
            lin_vel=(0.0, 0.0, 0.0),
            ang_vel=(0.0, 0.0, 0.0),
        ),
        spawn=sim_utils.CuboidCfg(
            size=(0.09, 0.07, 0.03),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.15, 0.08, 0.02)),  # dark brown
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.08),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
    )
    goal_marker = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Goal",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.20, 0.12, 0.005),
            lin_vel=(0.0, 0.0, 0.0),
            ang_vel=(0.0, 0.0, 0.0),
        ),
        spawn=sim_utils.CuboidCfg(
             size=(0.09, 0.07, 0.03),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.15, 0.08, 0.02)),  
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=False,
                disable_gravity=True,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.001),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
        ),
    )
    ground = AssetBaseCfg(prim_path="/World/Ground", spawn=GroundPlaneCfg())
    light  = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=2500.0),
    )


class JointTargetActionTerm(ActionTerm):
    """Absolute joint-position targets in radians (same convention as the thesis)."""
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._actions = torch.zeros((env.num_envs, 6), device=env.device)

    @property
    def action_dim(self):
        return 6

    @property
    def raw_actions(self):
        return self._actions

    @property
    def processed_actions(self):
        return self._actions

    def process_actions(self, actions: torch.Tensor):
        self._actions = actions.clone()

    def apply_actions(self):
        robot = self._env.scene["robot"]
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
        joint_pos    = ObsTerm(func=pick_mdp.joint_positions)
        object_pos   = ObsTerm(func=pick_mdp.object_position_w)
        ee_pos       = ObsTerm(func=pick_mdp.ee_position_w)
        object_to_goal = ObsTerm(func=pick_mdp.object_to_goal)
        # camera image — shape (H, W, 3) uint8 per env
        # front_image  = ObsTerm(func=pick_mdp.front_camera_rgb)

        def __post_init__(self):
            self.concatenate_terms = False   # keep image separate from vectors
            self.enable_corruption = False

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Randomise cube start and goal positions every episode."""
    reset_scene = EventTerm(func=mdp.reset_scene_to_default, mode="reset")
    reset_cube = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (0.15, 0.24),
                "y": (-0.12, 0.00),
                "z": (0.0, 0.0),
            },
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("object"),
        },
    )
    reset_goal = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (0.15, 0.24),
                "y": (0.04, 0.14),
                "z": (0.0, 0.0),
            },
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("goal_marker"),
        },
    )


@configclass
class RewardsCfg:
    reach  = RewTerm(func=pick_mdp.reaching_reward,  weight=1.0)
    lift   = RewTerm(func=pick_mdp.lift_reward,       weight=2.0)
    place  = RewTerm(func=pick_mdp.place_reward,      weight=4.0)


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=pick_mdp.time_out,      time_out=True)
    success  = DoneTerm(func=pick_mdp.object_in_goal)
    
@configclass
class CommandsCfg:
    """Command terms for the MDP."""

    object_pose = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name=MISSING,  # will be set by agent env cfg
        resampling_time_range=(5.0, 5.0),
        debug_vis=False,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(-0.1, 0.1),
            pos_y=(-0.3, -0.1),
            pos_z=(0.2, 0.35),
            roll=(0.0, 0.0),
            pitch=(0.0, 0.0),
            yaw=(0.0, 0.0),
        ),
    )


@configclass
class SOArmPickPlaceEnvCfg(ManagerBasedRLEnvCfg):
    scene:        PickPlaceSceneCfg = PickPlaceSceneCfg(num_envs=1, env_spacing=1.5)
    observations: ObservationsCfg  = ObservationsCfg()
    actions:      ActionsCfg       = ActionsCfg()
    events:       EventCfg         = EventCfg()
    rewards:      RewardsCfg       = RewardsCfg()
    terminations: TerminationsCfg  = TerminationsCfg()
    commands: CommandsCfg = CommandsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = SO_ARM101_CFG.replace(prim_path="{ENV_REGEX_NS}/robot")
        # target the URDF's dedicated tool frame instead of a hand-tuned offset
         # Set the body name for the end effector
        self.commands.object_pose.body_name = ["gripper_link"]

        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/robot/base_link",
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/robot/gripper_link",
                    name="end_effector",
                    offset=OffsetCfg( pos=(-0.0079, -0.000218, -0.0981),
                        rot=(0.0, 0.0, 1.0, 0.0)),
                )
            ],
        )
        self.decimation       = 4
        self.episode_length_s = 10.0
        self.sim.dt           = 1.0 / 120.0
        self.sim.render_interval = self.decimation
