import gym
import numpy as np
from gym import spaces
from collections import deque
import random
from stable_baselines3.common.callbacks import BaseCallback
from IPython.display import clear_output
import matplotlib.pyplot as plt
import torch
import torch.optim as optim

class OptimizationEnv(gym.Env):
    """
    A generic optimization environment with customizable reward functions.
    """
    def __init__(self, functions, n_dims=1, history_length=5, termination_prob=0.01, max_steps=200, reward_type=1):
        super(OptimizationEnv, self).__init__()
        
        self.functions = functions
        self.current_function = None
        self.n_dims = n_dims
        self.history_length = history_length
        self.termination_prob = termination_prob
        self.max_steps = max_steps
        self.reward_type = reward_type

        # Action space: Continuous step size for each dimension
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(n_dims,), dtype=np.float32)

        # Observation space: history_length recent positions + gradients + function values for each dimension
        obs_space_dim = 2 * history_length * n_dims + history_length  
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_space_dim,),
            dtype=np.float32
        )

        # Store recent positions, gradients, and function values
        self.positions = deque(maxlen=history_length)
        self.gradients = deque(maxlen=history_length)
        self.function_values = deque(maxlen=history_length)

        self.current_step = 0

    def calculate_function(self, x):
        raise NotImplementedError

    def calculate_gradient(self, x):
        raise NotImplementedError

    def predict_adam_action(self, x):
        """ Simulates Adam optimizer for 10 steps and returns the action taken. """
        x_tensor = torch.tensor(x, dtype=torch.float32, requires_grad=True)
        optimizer = optim.Adam([x_tensor], lr=0.1)

        for _ in range(10):
            optimizer.zero_grad()
            loss = self.calculate_function(x_tensor).sum()
            loss.backward()
            optimizer.step()

        return x_tensor.detach().cpu().numpy() - x  # Ensure detachment before conversion

    def step(self, action):
        current_x = self.positions[-1] if self.positions else np.zeros(self.n_dims)
        new_x = np.clip(current_x + action, -10, 10)
        function_value = self.calculate_function(new_x)
        gradient = self.calculate_gradient(new_x)

        if self.reward_type == 1:
            # Negative function value
            reward = -function_value
        elif self.reward_type == 2:
            # Function value difference
            prev_function_value = self.function_values[-1] if self.function_values else float('inf')
            reward = prev_function_value - function_value
        elif self.reward_type == 3:
            # Distance to Adam action
            adam_action = self.predict_adam_action(current_x)
            action_diff = np.linalg.norm(adam_action - action)
            reward = -function_value - action_diff
        elif self.reward_type == 4:
            # Stagewise reward with weight decay
            adam_action = self.predict_adam_action(current_x)
            action_diff = np.linalg.norm(adam_action - action)
            weight = 1
            if self.current_step > 50:
                # Define a weight decay function (linear decay)
                T = 100  # Decay period
                weight = max(0, 1 - (self.current_step-50) / T)

            # Weighted reward
            reward = weight * (-action_diff) + (1 - weight) * (-function_value)
        else:
            raise ValueError("Invalid reward type")
        
        # Update histories
        self.positions.append(new_x)
        self.gradients.append(gradient)
        self.function_values.append(function_value)

        state = np.concatenate([
            np.ravel(self.positions),
            np.ravel(self.gradients),
            np.ravel(self.function_values)
        ]).astype(np.float32)

        self.current_step += 1
        done = (self.current_step >= self.max_steps or random.random() < self.termination_prob)

        return state, reward, done, {}

    def reset(self):
        self.current_function = random.choice(list(self.functions))
        self.positions.clear()
        self.gradients.clear()
        self.function_values.clear()

        initial_x = np.random.uniform(-10, 10, size=self.n_dims)
        for _ in range(self.history_length - 1):
            self.positions.append(np.zeros(self.n_dims))
        self.positions.append(initial_x)

        initial_gradient = self.calculate_gradient(initial_x)
        initial_function_value = self.calculate_function(initial_x)
        for _ in range(self.history_length - 1):
            self.gradients.append(np.zeros(self.n_dims))
            self.function_values.append(0.0)
        self.gradients.append(initial_gradient)
        self.function_values.append(initial_function_value)

        state = np.concatenate([
            np.ravel(self.positions),
            np.ravel(self.gradients),
            np.ravel(self.function_values)
        ]).astype(np.float32)

        self.current_step = 0
        return state

    def render(self, mode='human'):
        pass

class QuadraticEnv1D(OptimizationEnv):
    def calculate_function(self, x):
        a, b, c = self.current_function
        return a * x[0]**2 + b * x[0] + c

    def calculate_gradient(self, x):
        a, b, _ = self.current_function
        return np.array([2 * a * x[0] + b])

class QuadraticEnvND(OptimizationEnv):
    def calculate_function(self, x):
        A, b, c = self.current_function
        if isinstance(x, torch.Tensor):
            A = torch.tensor(A, dtype=torch.float32, device=x.device)  # Convert A to tensor
            b = torch.tensor(b, dtype=torch.float32, device=x.device)  # Convert b to tensor
            c = torch.tensor(c, dtype=torch.float32, device=x.device)  # Convert c to tensor
            result = x.T @ A @ x + b.T @ x + c
            return result
        else:
            return float(x.T @ A @ x + b.T @ x + c)
    def calculate_gradient(self, x):
        A, b, _ = self.current_function
        return 2 * A @ x + b

class RosenbrockEnv(OptimizationEnv):
    def __init__(self, functions, history_length=5, termination_prob=0.01, max_steps=200, reward_type=1):
        """
        Initializes the Rosenbrock optimization environment for only 2D input (x, y).
        """
        super(RosenbrockEnv, self).__init__(functions, n_dims=2, history_length=history_length,
                                            termination_prob=termination_prob, max_steps=max_steps,
                                            reward_type=reward_type)

    def reset(self):
        """ Resets the environment and ensures initial x values are in [-2, 2]. """
        self.current_function = random.choice(list(self.functions))
        self.positions.clear()
        self.gradients.clear()
        self.function_values.clear()

        # Set initial x values in the range [-2, 2]
        initial_x = np.random.uniform(-2, 2, size=self.n_dims)
        for _ in range(self.history_length - 1):
            self.positions.append(np.zeros(self.n_dims))
        self.positions.append(initial_x)

        initial_gradient = self.calculate_gradient(initial_x)
        initial_function_value = self.calculate_function(initial_x)
        for _ in range(self.history_length - 1):
            self.gradients.append(np.zeros(self.n_dims))
            self.function_values.append(0.0)
        self.gradients.append(initial_gradient)
        self.function_values.append(initial_function_value)

        state = np.concatenate([
            np.ravel(self.positions),
            np.ravel(self.gradients),
            np.ravel(self.function_values)
        ]).astype(np.float32)

        self.current_step = 0
        return state

    def step(self, action):
        """ Steps the environment and ensures x stays within [-2, 2]. """
        current_x = self.positions[-1] if self.positions else np.zeros(self.n_dims)
        new_x = np.clip(current_x + action, -2, 2)  # Clip x within [-2, 2]
        function_value = self.calculate_function(new_x)
        gradient = self.calculate_gradient(new_x)

        # Compute reward
        if self.reward_type == 1:
            reward = -function_value
        elif self.reward_type == 2:
            prev_function_value = self.function_values[-1] if self.function_values else float('inf')
            reward = prev_function_value - function_value
        elif self.reward_type == 3:
            adam_action = self.predict_adam_action(current_x)
            action_diff = np.linalg.norm(adam_action - action)
            reward = -function_value - action_diff
        else:
            raise ValueError("Invalid reward type")
        
        # Update histories
        self.positions.append(new_x)
        self.gradients.append(gradient)
        self.function_values.append(function_value)

        # Construct state
        state = np.concatenate([
            np.ravel(self.positions),
            np.ravel(self.gradients),
            np.ravel(self.function_values)
        ]).astype(np.float32)

        self.current_step += 1
        done = (self.current_step >= self.max_steps or random.random() < self.termination_prob)

        return state, reward, done, {}

    def calculate_function(self, x):
        """ Computes the 2D Rosenbrock function with an offset. """
        a, b, offset = self.current_function

        if isinstance(x, torch.Tensor):
            x0, x1 = x[0], x[1]
            return (a - x0) ** 2 + b * (x1 - x0 ** 2) ** 2 + offset
        else:
            x = np.asarray(x)
            x0, x1 = x[0], x[1]
            return float((a - x0) ** 2 + b * (x1 - x0 ** 2) ** 2 + offset)

    def calculate_gradient(self, x):
        """ Computes the gradient of the 2D Rosenbrock function. """
        a, b, offset = self.current_function
        x = np.asarray(x)
        x0, x1 = x[0], x[1]

        grad_x0 = -2 * (a - x0) - 4 * b * x0 * (x1 - x0 ** 2)
        grad_x1 = 2 * b * (x1 - x0 ** 2)

        return np.array([grad_x0, grad_x1], dtype=np.float32)

class RastriginEnv(OptimizationEnv):
    def calculate_function(self, x):
        """ Computes the Rastrigin function with an offset. """
        A, offset = self.current_function
        n = len(x)

        if isinstance(x, torch.Tensor):
            return A * n + torch.sum(x**2 - A * torch.cos(2 * np.pi * x)) + offset
        
        else:
            return A * n + np.sum(x**2 - A * np.cos(2 * np.pi * x)) + offset

    def calculate_gradient(self, x):
        A, _ = self.current_function  # Unpack the values correctly
        x = np.asarray(x, dtype=np.float32)  # Convert x to a NumPy array
        return (2 * x + 2 * A * np.pi * np.sin(2 * np.pi * x))


class JupyterNotebookPlotCallback(BaseCallback):
    """
    A callback to plot training progress in Jupyter notebooks.
    """
    def __init__(self, verbose=0):
        super(JupyterNotebookPlotCallback, self).__init__(verbose)
        self.rewards = []
        self.steps = []

    def _on_step(self) -> bool:
        if "episode" in self.locals['infos'][0]:
            episode_reward = self.locals['infos'][0]['episode']['r']
            self.rewards.append(episode_reward)
            self.steps.append(self.num_timesteps)

            clear_output(wait=True)
            plt.figure(figsize=(10, 6))
            plt.plot(self.steps, self.rewards, label="Mean Reward")
            plt.xlabel("Steps")
            plt.ylabel("Mean Reward")
            plt.title("Training Progress")
            plt.legend()
            plt.grid()
            plt.show()

        return True