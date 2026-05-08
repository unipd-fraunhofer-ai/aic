from __future__ import annotations

import torch

from isaaclab.controllers import OperationalSpaceController, OperationalSpaceControllerCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import ActionTermCfg as ActionTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import SceneEntityCfg
from isaaclab.actuators import IdealPDActuatorCfg
from isaaclab.envs.mdp import JointEffortActionCfg
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.utils import configclass
from isaaclab.utils.math import (
    axis_angle_from_quat,
    quat_apply,
    quat_conjugate,
    quat_from_euler_xyz,
    quat_mul,
)

from ..aic_task_base_env_cfg import AICTaskBaseEnvCfg
from ..aic_task_base_env import AICTaskBaseEnv
from .. import mdp

@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    arm_action: ActionTerm = JointEffortActionCfg(
        asset_name="robot",
        joint_names=[
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "elbow_joint",
            "wrist_1_joint",
            "wrist_2_joint",
            "wrist_3_joint",
        ],
        scale=0.0, # This makes the action manager output 0 effort
    )

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
class EventCfg:
    """Configuration for events."""

    reset_scene = EventTerm(
        func=mdp.reset_to_near_completion,
        mode="reset",
        params={
            "reset_data_filename": "near_completion_states_100k.pt",
            "partially_inserted_prob": 0.8,  # Start with high probability of partially inserted states
        },
    )

@configclass
class CurriculumCfg:
    """Configuration for curriculum terms."""

    modify_reset_prob = CurrTerm(
        func=mdp.modify_reset_prob,
        params={
            "event_term_name": "reset_scene",
            "reward_term_name": "insertion_completed",
            "update_threshold": 0.6,
            "step": 0.10,
            "min_prob": 0.2,
        },
    )


##
# Env definition
##


@configclass
class RelCartesianOSPNoRefEnvCfg(AICTaskBaseEnvCfg):
    """Configuration for Relative Cartesian Operational Space Controller environment."""
    
    # Action scaling
    action_delta_pos_scale: float = 0.0005   # 0.5 mm/step
    action_delta_ori_scale: float = 0.009    # ~0.5°/step
    
    # MDP settings
    actions: ActionsCfg = ActionsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()


class RelCartesianOSPNoRefEnv(AICTaskBaseEnv):

    cfg: RelCartesianOSPNoRefEnvCfg
    
    def __init__(self, cfg: RelCartesianOSPNoRefEnvCfg, **kwargs) -> None:
        super().__init__(cfg, **kwargs)
    
    def _update_target(self, action: torch.Tensor) -> None:
        """Accumulate delta actions into the world-frame target pose."""
        pos_scale = self.cfg.action_delta_pos_scale
        ori_scale = self.cfg.action_delta_ori_scale

        # delta_pos is expressed in EE local frame; rotate to world.
        delta_pos_ee = action[:, :3] * pos_scale
        ee_quat = self._robot.data.body_quat_w[:, self._ee_body_idx, :]
        delta_pos = quat_apply(ee_quat, delta_pos_ee)
        delta_rpy = action[:, 3:6] * ori_scale

        self._target_pos_w = self._target_pos_w + delta_pos

        dq = quat_from_euler_xyz(delta_rpy[:, 0], delta_rpy[:, 1], delta_rpy[:, 2])
        self._target_quat_w = quat_mul(dq, self._target_quat_w)
        self._target_quat_w = self._target_quat_w / self._target_quat_w.norm(dim=-1, keepdim=True)