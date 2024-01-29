import gymnasium as gym
import random
import numpy as np
from abc import abstractmethod


class MDPEnv(gym.Env):
    """
    Description:
        A maze environment with a goal state and a wall.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def reset(self, seed=None, options=None):
        NotImplementedError

    def step(self, action):
        NotImplementedError

    def render(self, mode='human'):
        NotImplementedError

    def close(self):
        NotImplementedError

    """ Return all states of this MDP """
    @abstractmethod
    def get_states(self):
        pass

    """ Return all actions with non-zero probability from this state """
    @abstractmethod
    def get_actions(self, state):
        pass

    """ Return all non-zero probability transitions for this action
        from this state, as a list of (state, probability) pairs
    """
    @abstractmethod
    def get_transitions(self, state, action):
        pass

    """ Return the reward for transitioning from state to
        nextState via action
    """
    @abstractmethod
    def get_reward(self, state, action, next_state):
        pass

    """ Return true if and only if state is a terminal state of this MDP """
    @abstractmethod
    def is_terminal(self, state):
        pass

    """ Return the discount factor for this MDP """
    @abstractmethod
    def get_discount_factor(self):
        pass

    """ Return the initial state of this MDP """
    @abstractmethod
    def get_initial_state(self):
        pass

    """ Return all goal states of this MDP """
    @abstractmethod
    def get_goal_states(self):
        pass
