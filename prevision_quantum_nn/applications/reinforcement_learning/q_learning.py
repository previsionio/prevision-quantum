""" Q-Learning module """

import numpy as np
import tensorflow as tf

from .base_learner import BaseLearner


class QLearner(BaseLearner):
    """Q-table Learner

    This class implements a Q-table.

    Attributes:
        params (dictionary):containing the parameters of the model
    """
    def __init__(self, params):
        """Constructs the Q-table.

        Args:
            params (dictionary):contains the parameters of the learner
        """
        super().__init__(params)
        self.num_bins = 10

        # compute the shape of the Q-table
        self.state_shape = tuple(self.num_bins *
                                 np.ones(self.num_features, dtype=int))
        # initialize array with number of visits
        self.num_visits = np.zeros(self.state_shape)

        # initialize array with the cumulative reward
        self.reward = np.zeros(self.state_shape)

        # initialize the cell sizes for each feature
        self.cell_sizes = (self.high_limits - self.low_limits) / self.num_bins

    def get_cell(self, state):
        """Returns the cell in which the state is.

        Args:
            state (numpy array):state at which we wish to obtain the
                corresponding cell

        Returns:
            cell: tuple
                cell indices at which the Q-table will be updated
        """
        if self.tanh_mask is not None:
            state[:, self.tanh_mask] = \
                    tf.tanh(state[:, self.tanh_mask]).numpy()
        cell = (state - self.low_limits) / self.cell_sizes
        return tuple(cell.astype(int))

    def fit(self, state, reward):
        """Fits the Q-table at a given state.

        Args:
            state (numpy array):state at which the Q-table needs to fit
            reward (float):reward that was yielded when passing
                through this state
        """
        state_cell = self.get_cell(state)
        self.reward[state_cell] += reward
        self.num_visits[state_cell] += 1

    def forward(self, state, action=None):
        """Forwards the Q-able through the state (action) provided.

        Args:
            state (numpy array):state at which the Q-table needs
                to be evaluated
            action (int):if the Q-table is formed with the state action method,
                action needs to be provided as an option.

        Returns:
            value: float
                the actual Q-value at the state provided
        """
        state_cell = self.get_cell(state)
        if self.num_visits[state_cell] > 0:
            value = self.reward[state_cell]/self.num_visits[state_cell]
        else:
            value = 0
        return value
