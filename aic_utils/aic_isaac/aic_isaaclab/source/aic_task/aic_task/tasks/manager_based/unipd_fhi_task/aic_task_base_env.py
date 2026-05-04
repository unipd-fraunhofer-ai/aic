# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import os
from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import ActionTermCfg as ActionTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.sensors import FrameTransformerCfg, OffsetCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise


from . import mdp


# Resolve asset directory relative to this file (portable across machines)
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
AIC_ASSET_DIR = os.path.join(_THIS_DIR, "Intrinsic_assets")
AIC_SCENE_DIR = AIC_ASSET_DIR
AIC_PARTS_DIR = os.path.join(AIC_ASSET_DIR, "assets")

EXTENSION_PATH = os.path.dirname(os.path.abspath(__file__))


##
# Scene definition
##


@configclass
class AICTaskSceneCfg(InteractiveSceneCfg):
    """Scene for aic task: UR5e robot, aic_scene, task_board."""

    # UR5e + gripper (fully defined here using local asset)
    robot: ArticulationCfg = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=os.path.join(AIC_ASSET_DIR, "aic_unified_robot_cable_sdf.usd"),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=5.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True,
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=8,
            ),
            activate_contact_sensors=False,
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(-0.18, -0.122, 0),
            rot=(0.0, 0.0, 0.0, 1.0),
            joint_pos={
                "shoulder_pan_joint": -0.1597,
                "shoulder_lift_joint": -1.3542,
                "elbow_joint": -1.6648,
                "wrist_1_joint": -1.6933,
                "wrist_2_joint": 1.5710,
                "wrist_3_joint": 1.4110,
            },
        ),
        actuators={
            "arm": ImplicitActuatorCfg(
                joint_names_expr=[
                    "shoulder_pan_joint",
                    "shoulder_lift_joint",
                    "elbow_joint",
                    "wrist_1_joint",
                    "wrist_2_joint",
                    "wrist_3_joint",
                ],
                effort_limit_sim=87.0,
                stiffness=2000.0,
                damping=100.0,
            ),
        },
    )

    # world
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -1.05)),
    )

    aic_scene = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/aic_scene",
        spawn=sim_utils.UsdFileCfg(
            usd_path=os.path.join(AIC_SCENE_DIR, "scene", "aic.usd"),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.0, 0.0, -1.15),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )

    task_board = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/task_board",
        spawn=sim_utils.UsdFileCfg(
            usd_path=os.path.join(
                AIC_PARTS_DIR, "Task Board Base", "task_board_rigid.usd"
            ),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.35, -0.30, 0.0),
        ),
    )

    nic_card = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/nic_card",
        spawn=sim_utils.UsdFileCfg(
            usd_path=os.path.join(AIC_PARTS_DIR, "NIC Card", "nic_card.usd"),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.31765, -0.27671, 0.0743),
            rot=(0.0, 0.0, -0.7068252, 0.7073883),
        ),
    )

    sfp_tip_sensor = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/cable/sfp_module/sfp_tip_link",
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/cable/sfp_module/sfp_tip_link",
                name="sfp_tip",
            )
        ],
        visualizer_cfg = VisualizationMarkersCfg(
            prim_path="/Visuals/SfpTip",
            markers={
                "frame": sim_utils.UsdFileCfg(
                    usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/frame_prim.usd",
                    scale=(0.01, 0.01, 0.01),
                ),
            },
        ),
        debug_vis=True,
    )

    sfp_port_sensor = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/nic_card",
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/nic_card",
                name="port_0",
                offset=OffsetCfg(
                    pos=(0.01295, -0.031571999742310328, 0.0050100016239075385),
                    rot=(-4.3005889216184035e-17, 4.3587761463281758e-17, -0.70233947038650502, 0.71184216532683964),
                ),
            ),
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/nic_card",
                name="port_1",
                offset=OffsetCfg(
                    pos=(-0.01025, -0.031571999742310328, 0.0050100016239075385),
                    rot=(-4.3005889216184035e-17, 4.3587761463281758e-17, -0.70233947038650502, 0.71184216532683964),
                ),
            ),
        ],
        debug_vis=False,
    )


##
# MDP settings
##


@configclass
class EventCfg:
    """Configuration for events."""

    reset_scene = EventTerm(
        func=mdp.reset_to_near_completion,
        mode="reset",
        params={
            "reset_data_filename": "near_completion_states.pt",
            "only_success": False,
        },
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    arm_action: ActionTerm = MISSING


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    sfp_port_pose_command = mdp.SfpPoseTargetCommandCfg()


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    failed_insertion = DoneTerm(func=mdp.failed_insertion, params={
        "command_name": "sfp_port_pose_command",
        "tip_sensor_cfg": SceneEntityCfg("sfp_tip_sensor"),
        "port_sensor_cfg": SceneEntityCfg("sfp_port_sensor"),
    })


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP: robot state, ee pose, pose command."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy: joint state, ee pose, pose command."""

        # Minimal target port position and orientation (x, y, yaw = 3 dims)
        port_target = ObsTerm(
            func=mdp.target_port_base,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )

        # End effector position, orientation, linear velocity, angular velocity (12 dims)
        ee_pos = ObsTerm(
            func=mdp.ee_pos_base,
            params={"asset_cfg": SceneEntityCfg("robot", body_names="gripper_tcp")},
        )
        ee_rpy = ObsTerm(
            func=mdp.ee_rpy_base,
            params={"asset_cfg": SceneEntityCfg("robot", body_names="gripper_tcp")},
        )
        ee_lin_vel = ObsTerm(
            func=mdp.ee_lin_vel_base,
            params={"asset_cfg": SceneEntityCfg("robot", body_names="gripper_tcp")},
        )
        ee_ang_vel = ObsTerm(
            func=mdp.ee_ang_vel_base,
            params={"asset_cfg": SceneEntityCfg("robot", body_names="gripper_tcp")},
        )

        # Body forces (wrench) at the end-effector (force xyz + torque xyz = 6 dims)
        body_forces = ObsTerm(
            func=mdp.body_incoming_wrench,
            scale=0.1,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=["wrist_3_link"])
            },
        )

        # Last action (6 dims)
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True # Total obs dim = 3 + 12 + 6 + 6 = 27

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    insertion_completed = RewTerm(
        func=mdp.insertion_completed,
        weight=10.0,
        params={
            "threshold": 0.005,
            "command_name": "sfp_port_pose_command",
            "tip_sensor_cfg": SceneEntityCfg("sfp_tip_sensor"),
            "port_sensor_cfg": SceneEntityCfg("sfp_port_sensor"),
        },
    )

    insertion_position_error = RewTerm(
        func=mdp.insertion_position_error,
        weight=1,
        params={
            "command_name": "sfp_port_pose_command",
            "tip_sensor_cfg": SceneEntityCfg("sfp_tip_sensor"),
            "port_sensor_cfg": SceneEntityCfg("sfp_port_sensor"),
        },
    )
    insertion_position_error_tanh = RewTerm(
        func=mdp.insertion_position_error_tanh,
        weight=1,
        params={
            "std": 0.05,
            "command_name": "sfp_port_pose_command",
            "tip_sensor_cfg": SceneEntityCfg("sfp_tip_sensor"),
            "port_sensor_cfg": SceneEntityCfg("sfp_port_sensor"),
        },
    )

    insertion_pose_error = RewTerm(
        func=mdp.pose_error,
        weight=-1.5,
        params={
            "command_name": "sfp_port_pose_command",
            "tip_sensor_cfg": SceneEntityCfg("sfp_tip_sensor"),
            "port_sensor_cfg": SceneEntityCfg("sfp_port_sensor"),
        },
    )

    # -- Smoothness penalties --
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.001)
    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-0.001,
        params={"asset_cfg": SceneEntityCfg(
            "robot", 
                joint_names=[
                "shoulder_pan_joint", 
                "shoulder_lift_joint", 
                "elbow_joint", 
                "wrist_1_joint", 
                "wrist_2_joint", 
                "wrist_3_joint"
            ]
        )},
    )
    joint_acc = RewTerm(
        func=mdp.joint_acc_l2,
        weight=-1.0e-5,
        params={"asset_cfg": SceneEntityCfg(
            "robot", 
            joint_names=[
                "shoulder_pan_joint", 
                "shoulder_lift_joint", 
                "elbow_joint", 
                "wrist_1_joint", 
                "wrist_2_joint", 
                "wrist_3_joint"
            ]
        )},
    )
    joint_torques = RewTerm(
        func=mdp.joint_torques_l2,
        weight=-1.0e-6,
        params={"asset_cfg": SceneEntityCfg(
            "robot", 
            joint_names=[
                "shoulder_pan_joint", 
                "shoulder_lift_joint", 
                "elbow_joint", 
                "wrist_1_joint", 
                "wrist_2_joint", 
                "wrist_3_joint"
            ]
        )}
    )


##
# Environment configuration
##


@configclass
class AICTaskBaseEnv(ManagerBasedRLEnvCfg):
    """Base environment configuration for the AIC task"""

    # Scene settings
    scene: AICTaskSceneCfg = AICTaskSceneCfg(num_envs=2048, env_spacing=4.0)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    # MDP settings
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self) -> None:
        super().__post_init__()

        # General settings
        self.decimation = 4
        self.sim.render_interval = self.decimation
        self.episode_length_s = 10.0
        self.sim.dt = 1.0 / 120.0
        self.viewer.eye = (8.0, 0.0, 5.0)