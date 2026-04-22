from isaaclab.utils import configclass
from ..aic_task_base_env import AICTaskBaseEnv
from ..mdp.actions.cartesian_impedance_actions import CartesianImpedanceTargetActionCfg

@configclass
class RelCartesianNoRefEnvCfg(AICTaskBaseEnv):
    """Environment for relative cartesian impedance control w.r.t. tcp frame"""


    def __post_init__(self) -> None:
        super().__post_init__()

        self.actions.arm_action = CartesianImpedanceTargetActionCfg(
            asset_name="robot",
            joint_names=["shoulder.*", "elbow.*", "wrist.*"],
            body_name="wrist_3_link",
            command_frame="gripper/tcp",
            use_relative_mode=True,
            position_scale=0.003,
            orientation_scale=0.03,
            translational_stiffness=(85.0, 85.0, 85.0),
            rotational_stiffness=(85.0, 85.0, 85.0),
            translational_damping=(75.0, 75.0, 75.0),
            rotational_damping=(75.0, 75.0, 75.0),
            torque_limit=150.0,
        )