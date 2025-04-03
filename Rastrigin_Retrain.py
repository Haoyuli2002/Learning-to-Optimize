import data
from stable_baselines3 import PPO
from stable_baselines3 import SAC
from stable_baselines3 import DDPG
import random
import numpy as np
from collections import deque
import environments
from environments import RastriginEnv
from IPython.display import clear_output
import torch
import matplotlib.pyplot as plt
import testing

device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

training_functions = data.generate_dataset(n = 100, function_type = 'rastrigin_function', n_dims = 3)
testing_functions = data.generate_dataset(n = 10, function_type = 'rastrigin_function', n_dims = 3)

env = RastriginEnv(functions = training_functions, n_dims = 2, max_steps = 200, reward_type = 1)
model_ppo = PPO("MlpPolicy", env, verbose=0, learning_rate=1e-4)
callback = environments.JupyterNotebookPlotCallback()
model_ppo.learn(total_timesteps=1e6, callback=callback)
model_ppo.save("Rastrigin_ppo")

env = RastriginEnv(functions = training_functions, n_dims = 2, max_steps = 200)
model_ddpg = DDPG("MlpPolicy", env, verbose=0, learning_rate=1e-4)
callback = environments.JupyterNotebookPlotCallback()
model_ddpg.learn(total_timesteps=1e6, callback=callback)
model_ddpg.save("Rastrigin_ddpg")

env = RastriginEnv(functions = training_functions, n_dims = 2, max_steps = 200)
model_sac = SAC("MlpPolicy", env, verbose=0, learning_rate=1e-4)
callback = environments.JupyterNotebookPlotCallback()
model_sac.learn(total_timesteps=1e6, callback=callback)
model_sac.save("Rastrigin_sac")

models = {
        'PPO': model_ppo,
        'SAC': model_sac,
        'DDPG': model_ddpg,
        "Adam": None,
        "RMSProp": None,
        "SGD": None,
}

env = RastriginEnv(functions = testing_functions, n_dims = 2, max_steps = 200)
max_iterations = 200
results = testing.run_optimizer_tests(env, testing_functions, models, max_iterations = 200, function_type = 'rastrigin_function')
avg_values = testing.calculate_average_values(results)
testing.plot_results(avg_values, max_iterations, "rastrigin_retrained")