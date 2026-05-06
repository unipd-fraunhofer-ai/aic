from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import EventTermCfg, ManagerTermBase, SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class modify_reset_prob(ManagerTermBase):
    """
    Curriculum term to decrease the probability of resetting in a "semi-inserted" state.    
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.success_count = 0
        self.terminated_count = 0
        self.cooldown_counter = 0

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor,
        event_term_name: str,
        reward_term_name: str,
        update_threshold: float,
        param_name: str = "partially_inserted_prob",
        step: float = 0.05,
        min_prob: float = 0.0,
        cooldown_eps: int = 25,
    ) -> float:
        # Get the reward manager and the index of the monitored term
        reward_manager = env.reward_manager
        if reward_term_name not in reward_manager.active_terms:
            return 0.0
        
        term_idx = reward_manager.active_terms.index(reward_term_name)
        weight = reward_manager.get_term_cfg(reward_term_name).weight
        
        self.terminated_count += len(env_ids)
        term_rewards = reward_manager._step_reward[env_ids, term_idx]
        successes = (term_rewards / weight) > 0.5 # Assume sparse reward 0.0/1.0 -> success when > 0
        self.success_count += torch.sum(successes).item()

        # Get the event term configuration to access current param value
        event_term_cfg = env.event_manager.get_term_cfg(event_term_name)
        current_prob = event_term_cfg.params.get(param_name, 0.0)

        # Check if we have reached the total number of environments for evaluation
        if self.terminated_count >= env.num_envs:
            success_rate = self.success_count / self.terminated_count
            print(f"----------------------------------------")
            print(f"Success Rate: {success_rate:.2f}")
            print(f"Current Prob: {current_prob:.2f}")
            print(f"Cooldown Counter: {self.cooldown_counter}")
            print(f"----------------------------------------")

            # Update the probability if success rate is high enough and cooldown is finished
            if self.cooldown_counter > 0:
                self.cooldown_counter -= 1
            elif success_rate > update_threshold:
                new_prob = max(min_prob, current_prob - step)
                event_term_cfg.params[param_name] = new_prob
                # Reset cooldown counter 
                self.cooldown_counter = cooldown_eps
                
            # Reset counters for the next batch of N environments
            self.success_count = 0
            self.terminated_count = 0
        
        return event_term_cfg.params.get(param_name, 0.0)
