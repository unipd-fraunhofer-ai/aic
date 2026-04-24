from isaaclab.utils import configclass

from isaaclab.actuators import IdealPDActuatorCfg
from isaaclab.envs.mdp import OperationalSpaceControllerActionCfg
from isaaclab.controllers.operational_space_cfg import OperationalSpaceControllerCfg

from ..aic_task_base_env import AICTaskBaseEnv


@configclass
class RelCartesianOSPNoRefEnvCfg(AICTaskBaseEnv):
    """Relative Cartesian Operational Space Controller w.r.t. the current EE pose.

    Action space: 6D = [dx, dy, dz, dax, day, daz]
    """

    def __post_init__(self) -> None:
        super().__post_init__()

        self.actions.arm_action = OperationalSpaceControllerActionCfg(
            asset_name="robot",
            joint_names=[
                "shoulder_pan_joint",
                "shoulder_lift_joint",
                "elbow_joint",
                "wrist_1_joint",
                "wrist_2_joint",
                "wrist_3_joint",
            ],
            body_name="gripper_tcp",
            body_offset=None,
            controller_cfg=OperationalSpaceControllerCfg(
                target_types=["pose_rel"],
                impedance_mode="fixed",
                motion_control_axes_task=(1, 1, 1, 1, 1, 1),
                contact_wrench_control_axes_task=(0, 0, 0, 0, 0, 0),
                inertial_dynamics_decoupling=True,
                partial_inertial_dynamics_decoupling=False,
                gravity_compensation=True,
              
                # Kp
                motion_stiffness_task=(1500.0, 1500.0, 1500.0, 300.0, 300.0, 300.0),

                # choose zeta so that d = 2*sqrt(Kp)*zeta
                motion_damping_ratio_task=(0.5, 0.5, 0.5, 0.25, 0.25, 0.25),
            ),
            position_scale=0.01,
            orientation_scale=0.1,
        )

        # replace implicit actuators with explicit torque actuators
        self.scene.robot.actuators["arm"] = IdealPDActuatorCfg(
            joint_names_expr=[
                "shoulder_pan_joint",
                "shoulder_lift_joint",
                "elbow_joint",
                "wrist_1_joint",
                "wrist_2_joint",
                "wrist_3_joint",
            ],
            stiffness=0.0,
            damping=0.0,
            effort_limit=187.0,
            effort_limit_sim=187.0,
            velocity_limit_sim=100.0,
        )