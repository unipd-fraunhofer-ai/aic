from isaaclab.utils import configclass
from ..aic_task_base_env import AICTaskBaseEnv
from isaaclab.envs.mdp import DifferentialInverseKinematicsActionCfg
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg


@configclass
class RelCartesianDiffIKNoRefEnvCfg(AICTaskBaseEnv):
    """Relative Cartesian control using Differential Inverse Kinematics w.r.t. the current EE pose."""

    def __post_init__(self) -> None:
        super().__post_init__()

        self.actions.arm_action = DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=["shoulder.*", "elbow.*", "wrist.*"],
            body_name="gripper_tcp",
            controller=DifferentialIKControllerCfg(
                command_type="pose",
                use_relative_mode=True,
                ik_method="dls",
            ),            
            scale=0.003,
        )