import gym
import prevision_quantum_nn as qnn

environment = gym.make("MountainCar-v0")

application = qnn.get_application("reinforcement_learning")
application.solve(environment, tqdm_verbose=True)
