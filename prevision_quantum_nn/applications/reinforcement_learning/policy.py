""" Policy module

contains the classes to handle policies in the
Reinforcement Learning framework.
"""
import logging

import numpy as np
import pandas as pd
import tensorflow as tf

LEARNING_TYPES = ["monte_carlo", "q_learning", "td_learning"]


class Policy:
    """Policy, base class for definiing policies.

    Attributes:
        params (dictionary):parameters of the policy
        iteration (int):the current iteration at which the policy is
        epsilon (float):the parameter of the epsilon-greedy method
        epsilon_decay (float):the decay parameter that will decay
            epsilon after each iteration
        gamma (float):the parameter of the temporal difference method
        learner (RLLearner):the learner used to evaluate the Q-value
            function
        use_memory_replay (bool):if True, memory replay will be activated
        memory_replay_period (int):the period of the memory replay method
        learning_type (str):one instance of learning_types. It can be
            "monte_carlo", "q_learning" or "td_learning"
    """
    def __init__(self, params):
        """Constructor.

        Args:
            params: (dictionary):contains the parameters of the policy
        """
        self.params = params
        self.iteration = 0
        self.epsilon = 1.0
        self.epsilon_decay = 0.999
        self.gamma = 0.99
        self.memory_replay_period = 64
        self.num_actions = None
        self.logger = logging.getLogger("policy")
        self.learner = None

        self.use_memory_replay = self.params.get("use_memory_replay", True)
        self.learning_type = self.params.get("learning_type", "td_learning")
        self.num_actions = self.params.get("num_actions", None)

        if self.learning_type not in LEARNING_TYPES:
            raise ValueError(f"Unknown learning type: {self.learning_type}")
        if self.use_memory_replay:
            self.memory = pd.DataFrame(columns=["features", "target"])

    def associate_learner(self, learner):
        """ associates a learner to the policy """
        self.learner = learner

    def step(self, state):
        """Step one iteration further.

        Args:
            state (numpy array):state at which the agent is before
                taking an action

        Returns:
            action: int
                action taken by the policy
        """
        self.iteration += 1
        action = self.get_action(state)
        return action

    def update_epsilon_greedy_parameter(self):
        """ updates epsilon if higher than minimum """
        if self.epsilon > 0.01:
            self.epsilon *= self.epsilon_decay
            self.logger.info(f"epsilon decay: {self.epsilon:.3e}")

    def fit_learner(self, state, action, reward, new_state):
        """Fits the learner.

        Args:
            state (numpy array):state at which the learner is
            action (int):action that has just been taken in that state
            reward (float):reward obtained by taking this action in the
                current state
        """
        # get target
        if self.learner.value_function == "state":
            target = self.learner.forward(state)
            new_target = self.learner.forward(new_state)
        elif self.learner.value_function == "state_action":
            target = self.learner.forward(state, action=action)
            new_target = self.learner.forward(new_state, action=action)

        if self.learning_type == "monte_carlo":
            target[0][action] = reward
        elif self.learning_type == "q_learning":
            target[0][action] = reward + np.max(new_target)
        elif self.learning_type == "td_learning":
            target[0][action] = reward + np.max(new_target) * self.gamma

        if self.learner.type == "quantum":
            target = tf.sigmoid(target).numpy()

        # fit model at the end of memory_replay_period
        if self.use_memory_replay:
            if len(self.memory) > self.memory_replay_period:
                # fit
                self.learner.fit(np.vstack(self.memory["features"]),
                                 np.vstack(self.memory["target"]))
                self.update_epsilon_greedy_parameter()

                # empty memory
                self.memory = self.memory[0:0]
            else:
                self.memory = \
                    self.memory.append({"features": state,
                                        "target": target},
                                       ignore_index=True)
        # or at each step if no memory replay is used
        else:
            self.learner.fit(state, target)
            self.update_epsilon_greedy_parameter()

    def get_action(self, state, action=None):
        """ get action for current state
        to be overridden
        """
        raise NotImplementedError("Implement in doughter class")


class BehaviorPolicy(Policy):
    """Behavior Policy.

    Implements an exploratory policy with the epsilon greedy method.
    """
    def get_action(self, state, action=None):
        """Returns an action to be taken by the agent.

        According to epsilon greedy policy.

        Args:
            state (numpy array):state at which the agent is
            action (int):action that has just been taken
        """
        if np.random.rand(1) < self.epsilon:
            action = np.random.randint(0, self.num_actions)
        else:
            if self.learner.value_function == "state":
                action = np.argmax(self.learner.forward(state))
            elif self.learner.value_function == "state_action":
                if action is not None:
                    action = np.argmax(self.learner.forward(state,
                                                            acition=action))
                else:
                    raise ValueError("You chosed and action_value function, "
                                     "without providing get_action with "
                                     "an action")
        return action


class TargetPolicy(Policy):
    """Target Policy.

    Implements an purely exploitation policy without random actions
    """
    def get_action(self, state, action=None):
        """Get action.

        Args:
            state (numpy array):state at which the action needs to be taken

        Returns:
            action: int
                the action taken at that state
        """

        action = np.argmax(self.learner.forward(state))
        return action
