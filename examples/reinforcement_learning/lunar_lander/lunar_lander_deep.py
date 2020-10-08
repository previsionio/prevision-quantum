import gym
import prevision_quantum_nn as qnn

environment = gym.make("LunarLander-v2")

application = qnn.get_application("reinforcement_learning",
                                  prefix="lunar_lander_deep",
                                  rl_learner_type="deep")

application.solve(environment, render=True, tqdm_verbose=False)
