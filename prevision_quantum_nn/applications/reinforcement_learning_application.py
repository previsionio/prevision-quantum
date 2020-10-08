""" Reinforcement Learning module
    provides with the class for reinforcement learning applications
"""
from tqdm.auto import tqdm
import numpy as np
import tensorflow as tf
import gym

from prevision_quantum_nn.applications.application import Application
from prevision_quantum_nn.applications.reinforcement_learning.qnn_q_learning \
        import QNNQLearner
from prevision_quantum_nn.applications.reinforcement_learning.q_learning \
        import QLearner
from prevision_quantum_nn.applications.reinforcement_learning.deep_q_learning \
        import DeepFullyConnectedLearner
from prevision_quantum_nn.applications.reinforcement_learning.policy \
        import BehaviorPolicy


class ReinforcementLearningApplication(Application):
    """Reinforcement Learning application.

    Attributes:
        value_function: (str):type of the value function, can be either
            state or state_action
        environment (gym.env or custom environment):environment to be solved
        num_acitons (int):number of possible actions for the agent
    """
    def __init__(self,
                 prefix="qnn",
                 preprocessing_params=None,
                 model_params=None,
                 postprocessing_params=None,
                 rl_learner_type="quantum"):
        """constructor """
        super().__init__(prefix)
        self.value_function = "state"
        self.num_actions = None
        self.environment = None
        self.preprocessing_params = preprocessing_params
        self.model_params = model_params
        self.postprocessing_params = postprocessing_params
        self.learner_type = rl_learner_type
        self.learner = None
        self.policy = None

    def build(self):
        """Build the Application (learner, policy)."""
        use_memory_replay = False

        # build the learner of the value function
        if isinstance(self.environment.observation_space, gym.spaces.Box):
            learner_params = {
                "value_function": self.value_function,
                "num_features": len(self.environment.observation_space.low),
                "observation_space": {
                    "low": self.environment.observation_space.low,
                    "high": self.environment.observation_space.high
                    },
                "num_actions": self.environment.action_space.n
            }
        else:
            learner_params = {
                "value_function": self.value_function,
                "num_features": self.environment.observation_space.n,
                "observation_space": {
                    "num_observations": self.environment.observation_space.n
                    },
                "num_actions": self.environment.action_space.n
            }

        if self.learner_type == "quantum":
            self.learner = QNNQLearner(learner_params,
                                       self.preprocessing_params,
                                       self.model_params,
                                       self.preprocessing_params)
        elif self.learner_type == "qtable":
            self.learner = QLearner(learner_params)
        elif self.learner_type == "deep":
            self.learner = DeepFullyConnectedLearner(learner_params)

        if not isinstance(self.environment.action_space, gym.spaces.Discrete):
            raise ValueError("Only discrete actions are supported for now")

        # build the policy
        policy_params = {
            "learning_type": "td_learning",
            "use_memory_replay": use_memory_replay,
            "epsilon": 0.9,
            "num_actions": self.environment.action_space.n
        }
        self.policy = BehaviorPolicy(policy_params)
        self.policy.associate_learner(self.learner)

    def solve(self,
              environment,
              tqdm_verbose=True,
              render=False):
        """Solves the problem given a certain environment.

        Args:
            environment (gym like env):environment of RL app
            tqdm_verbose (bool):activates tqdm (optionnal)
            render (bool):activates rendering (optionnal)
        """
        self.environment = environment

        self.build()

        num_episodes = 500
        max_num_steps_per_episode = 1000

        self.logger.info("starting episodes")

        # init episode tqdm
        episodes_range = tqdm(range(num_episodes), 
                              position=1,
                              desc="episodes",
                              disable=not tqdm_verbose)

        # for each episode
        for episode in episodes_range:

            # reset the environment
            state = self.preprocess_state(self.environment.reset())
            total_reward = 0

            # init steps tqdm
            steps_range = tqdm(range(max_num_steps_per_episode),
                               position=0,
                               desc="steps",
                               disable=not tqdm_verbose,
                               leave=True)

            # for each step
            for _ in steps_range:

                if render:
                    self.environment.render()

                action = self.policy.step(state)
                new_state, reward, done, _ = self.environment.step(action)
                new_state = self.preprocess_state(new_state)

                total_reward += reward

                if done:
                    break

                self.policy.fit_learner(state, action, reward, new_state)

                state = new_state

            self.logger.info(f"episode: {episode} "
                             f"reward: {total_reward:.3e} ")

    def preprocess_state(self, state):
        """Preprocess the state.

        Args:
            state (int or state):state, discrete or continuous

        Returns:
            state: np.array
                preprocessed state
        """
        preprocessed_state = None
        # discrete state, one hot encode first
        if isinstance(state, (np.int64, int)):
            preprocessed_state = np.pi * np.array(
                [tf.one_hot(state,
                            self.learner.num_observation_space).numpy()])
        # continuous state
        else:
            preprocessed_state = np.array([state])
        return preprocessed_state
