import gymnasium as gym
import numpy as np

class DiscountRewardWrapper(gym.RewardWrapper):
    """
    Wrapper for discounting rewards. The reward of the last step is discounted
    by the number of timesteps the agent took to reach the terminal state.
    This wrapper is useful for episodic tasks where the agent receives a reward
    only at the end of the episode, which is sparse and delayed.

    This wrapper is only compatible with episodic tasks, therefore, the environment
    should be wrapped with the TimeLimit wrapper.

    Args:
        env (gym.Env): The environment to be wrapped.
    """
    def __init__(self, env):
        super().__init__(env)
        if not isinstance(env, gym.wrappers.TimeLimit):
            raise ValueError("DiscountReward is only compatible with episodic tasks.",
                             "Please wrap the environment with the TimeLimit wrapper.")

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        if terminated:
            reward -= self.env._elapsed_steps / self.env._max_episode_steps
        return observation, reward, terminated, truncated, info