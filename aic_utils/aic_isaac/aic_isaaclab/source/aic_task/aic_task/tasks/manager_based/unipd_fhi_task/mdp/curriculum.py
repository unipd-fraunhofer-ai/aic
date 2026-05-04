from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def modify_reset_prob(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    event_term_name: str,
    reward_term_name: str,
    update_threshold: float,
    param_name: str = "partially_inserted_prob",
    step: float = 0.05,
    min_prob: float = 0.0,
) -> float:
    """
    Curriculum term to decrease the probability of resetting in a "semi-inserted" state.
    """
    # Get the reward manager and the index of the monitored term
    reward_manager = env.reward_manager
    if reward_term_name not in reward_manager.active_terms:
        return 0.0
    
    term_idx = reward_manager.active_terms.index(reward_term_name)
    weight = reward_manager.get_term_cfg(reward_term_name).weight
    
    # Get current success rate from the reward term's step buffer
    # _step_reward stores (value / dt), which is (reward_func_output * weight)
    if weight != 0:
        term_rewards = reward_manager._step_reward[:, term_idx]
        success_rate = torch.mean(term_rewards / weight).item()
    else:
        success_rate = 0.0

    # Get the event term configuration
    event_term_cfg = env.event_manager.get_term_cfg(event_term_name)
    current_prob = event_term_cfg.params.get(param_name, 0.0)

    # Update the probability if success rate is high enough
    if success_rate > update_threshold:
        new_prob = max(min_prob, current_prob - step)
        event_term_cfg.params[param_name] = new_prob
        print(f"[Curriculum] Success rate: {success_rate:.4f} > {update_threshold}. Updating {event_term_name} '{param_name}' to {new_prob:.4f}")
    
    return event_term_cfg.params.get(param_name, 0.0)

