""" Base Reinforcement Learning module """
import numpy as np


class BaseLearner():
    """Base Reinforcement Learning Learner.

    Base class for further implementations of learners.

    Attributes:
        value_function (str):type of the value function, can be
            either state or state_action
    """
    def __init__(self, params):
        """constructor """
        self.params = params
        self.value_function = params.get("value_function", "state")
        self.observation_space = params.get("observation_space")
        self.tanh_bounds = None
        self.tanh_mask = None

        self.num_features = params.get("num_features", None)
        self.num_actions = params.get("num_actions", None)

        if self.observation_space.get("low", None) is not None:

            self.low_limits = self.observation_space.get("low")
            self.high_limits = self.observation_space.get("high")

            # if low limits or high limits are infinite, apply tanh on mask
            tanh_low = np.where(self.low_limits <= -1e38,
                                -1,
                                self.low_limits)
            tanh_high = np.where(self.high_limits >= -1e38,
                                 1,
                                 self.high_limits)
            self.tanh_bounds = np.hstack([tanh_low.reshape(-1, 1),
                                          tanh_high.reshape(-1, 1)]).T

            low = np.where(self.low_limits <= -1e38)
            high = np.where(self.high_limits >= 1e38)
            self.tanh_mask = np.unique(np.stack([low, high]))
            self.low_limits[self.tanh_mask] = -1
            self.high_limits[self.tanh_mask] = 1
        else:
            self.num_observation_space = \
                self.observation_space.get("num_observations")
