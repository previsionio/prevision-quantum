import gym
import prevision_quantum_nn as qnn

environment = gym.make("LunarLander-v2")

model_params = {
    "architecture": "cv",
    "encoding": "displacement",
    "interface": "autograd",
    "num_layers": 3,
    "learning_rate": 0.001,
    "cutoff_dim": 4
}
application = qnn.get_application("reinforcement_learning",
                                  prefix="lunar_lander",
                                  model_params=model_params)

application.solve(environment, render=True, tqdm_verbose=False)
