import gymnasium as gym

from .. import agents

##
# Register Gym environments.
##


gym.register(
    id="Rel-Cart-No-Ref",
    entry_point=f"{__name__}.rel_cart_no_ref:RelCartesianOSPNoRefEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rel_cart_no_ref:RelCartesianOSPNoRefEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
    },
)

gym.register(
    id="Rel-Cart-Residual",
    entry_point=f"{__name__}.rel_cart_residual:RelCartesianOSPResidualEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rel_cart_residual:RelCartesianOSPResidualEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
    },
)
