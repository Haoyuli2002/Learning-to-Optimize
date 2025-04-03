import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from gym import spaces
from collections import deque
import matplotlib.pyplot as plt
from stable_baselines3 import SAC, PPO, DDPG
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from stable_baselines3.common.callbacks import BaseCallback
from IPython.display import clear_output
import matplotlib.pyplot as plt
import copy

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

class SmallNN(nn.Module):
    def __init__(self):
        super(SmallNN, self).__init__()
        self.hidden = nn.Linear(2, 2)
        self.output = nn.Linear(2, 1)

        self.init_weights()

    def init_weights(self):
        nn.init.kaiming_normal_(self.hidden.weight, nonlinearity="leaky_relu")
        nn.init.constant_(self.hidden.bias, 0)
        nn.init.xavier_normal_(self.output.weight)
        nn.init.constant_(self.output.bias, 0)

    def forward(self, x):
        x = torch.nn.functional.leaky_relu(self.hidden(x))
        x = torch.sigmoid(self.output(x))
        return x

class LogisticRegressionEnv(gym.Env):
    def __init__(self, datasets):
        super(LogisticRegressionEnv, self).__init__()

        self.datasets = datasets
        self.current_dataset_idx = 0

        self.model = SmallNN()
        self.criterion = nn.BCELoss()
        self.X_train, self.y_train = self.datasets[self.current_dataset_idx]

        self.param_history = deque(maxlen=5)
        self.grad_history = deque(maxlen=5)
        self.loss_history = deque(maxlen=5)
        self.accuracy_history = deque(maxlen=5)

        # SAC learns to optimize 9 parameters (weights & biases)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(9,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(5 * 9 + 5 * 9 + 5 + 5,), dtype=np.float32)

        self.max_steps = 200
        self.current_step = 0

    def get_params(self):
        with torch.no_grad():
            params = torch.cat([
                self.model.hidden.weight.flatten(),
                self.model.hidden.bias.flatten(),
                self.model.output.weight.flatten(),
                self.model.output.bias.view(-1)
            ]).numpy()
        return params

    def compute_gradient(self):
        self.model.zero_grad()
        outputs = self.model(self.X_train)
        loss = self.criterion(outputs, self.y_train)
        loss.backward()

        def safe_grad(tensor):
            return tensor.grad.flatten().numpy() if tensor.grad is not None else np.zeros(tensor.shape).flatten()

        gradients = np.concatenate([
            safe_grad(self.model.hidden.weight),
            safe_grad(self.model.hidden.bias),
            safe_grad(self.model.output.weight),
            np.array([self.model.output.bias.grad.item()]) if self.model.output.bias.grad is not None else np.zeros(1)
        ])
        return gradients

    def compute_accuracy(self):
        with torch.no_grad():
            outputs = self.model(self.X_train)
            predictions = (outputs >= 0.5).float()
            accuracy = (predictions == self.y_train).float().mean().item()
        return accuracy

    def reset(self):
        self.current_dataset_idx = np.random.randint(len(self.datasets))
        self.X_train, self.y_train = self.datasets[self.current_dataset_idx]
        
        self.model = SmallNN()
        params = self.get_params()
        gradients = self.compute_gradient()
        loss = self.criterion(self.model(self.X_train), self.y_train).item()
        accuracy = self.compute_accuracy()

        self.param_history.clear()
        self.grad_history.clear()
        self.loss_history.clear()
        self.accuracy_history.clear()

        zero_params = np.zeros_like(params)
        zero_gradients = np.zeros_like(gradients)
        zero_loss = 0.0
        zero_acc = 0.0

        for _ in range(4):
            self.param_history.append(zero_params)
            self.grad_history.append(zero_gradients)
            self.loss_history.append(zero_loss)
            self.accuracy_history.append(zero_acc)

        self.param_history.append(params)
        self.grad_history.append(gradients)
        self.loss_history.append(loss)
        self.accuracy_history.append(accuracy)

        self.current_step = 0  
        return self._get_state()

    def _get_state(self):
        return np.concatenate([
            np.ravel(self.param_history),
            np.ravel(self.grad_history),
            np.array(self.loss_history),
            np.array(self.accuracy_history)
        ])

    def step(self, action):
        self.current_step += 1  
        step_size = max(0.1 * (0.95 ** self.current_step), 0.001)
        with torch.no_grad():
            current_params = self.get_params()
            new_params = current_params + action * step_size
            self.model.hidden.weight.data = torch.tensor(new_params[:4]).view(2, 2)
            self.model.hidden.bias.data = torch.tensor(new_params[4:6])
            self.model.output.weight.data = torch.tensor(new_params[6:8]).view(1, 2)
            self.model.output.bias.data = torch.tensor(new_params[8])

        # Compute new loss and accuracy
        outputs = self.model(self.X_train)
        loss = self.criterion(outputs, self.y_train).item()
        accuracy = self.compute_accuracy()
        gradients = self.compute_gradient()

        self.param_history.append(self.get_params())
        self.grad_history.append(gradients)
        self.loss_history.append(loss)
        self.accuracy_history.append(accuracy)

        reward = -loss

        done = ((self.current_step % 200) == 0)
        
        return self._get_state(), reward, done, {}

    def render(self, mode='human'):
        """Prints loss and accuracy during training."""
        print(f"Step {self.current_step}: Loss={self.loss_history[-1]:.4f}, Accuracy={self.accuracy_history[-1]:.4f}")
        print(f"Gradients {self.grad_history[-1]}.")

def test_l2o_agent(env, X_test, y_test, base_model, agent, num_steps=200):
    env.reset()
    env.model = base_model
    env.X_train = X_test
    env.y_train = y_test
    env.param_history.clear()
    env.grad_history.clear()
    env.loss_history.clear()
    env.accuracy_history.clear()
    
    params = env.get_params()
    gradients = env.compute_gradient()
    loss = env.criterion(env.model(env.X_train), env.y_train).item()
    accuracy = env.compute_accuracy()

    zero_params = np.zeros_like(params)
    zero_gradients = np.zeros_like(gradients)
    zero_loss = 0.0
    zero_acc = 0.0
    
    for _ in range(4):
        env.param_history.append(zero_params)
        env.grad_history.append(zero_gradients)
        env.loss_history.append(zero_loss)
        env.accuracy_history.append(zero_acc)
    
    env.param_history.append(params)     
    env.grad_history.append(gradients)
    env.loss_history.append(loss)
    env.accuracy_history.append(accuracy)
    
    obs = env._get_state().astype(np.float32)

    loss_values = [env.loss_history[-1]]  
    acc_values = [env.accuracy_history[-1]]

    for i in range(num_steps):  
        action, _ = agent.predict(obs, deterministic=True)
        obs, reward, done, _ = env.step(action)
        obs = obs.astype(np.float32)
        loss_values.append(env.loss_history[-1])  
        acc_values.append(env.accuracy_history[-1])

        if done:
            break

    return loss_values, acc_values

import numpy as np
import torch
import torch.optim as optim

def test_l2o_agent_with_rmsprop(env, X_test, y_test, base_model, agent, num_steps=200, switch_point=10, lr=0.01):
    """
    Test L2O agent for the first `switch_point` iterations, then switch to RMSProp.

    Parameters:
    - env: The LogisticRegressionEnv environment.
    - X_test: Test feature data.
    - y_test: Test labels.
    - base_model: SmallNN model structure.
    - agent: L2O-trained RL agent (PPO, SAC, DDPG).
    - num_steps: Total optimization steps (default: 200).
    - switch_point: Step at which to switch from L2O to RMSProp (default: 10).
    - lr: Learning rate for RMSProp (default: 0.01).

    Returns:
    - loss_values: List of loss values during training.
    - acc_values: List of accuracy values during training.
    """
    # Reset the environment and set the model
    env.reset()
    env.model = base_model
    env.X_train = X_test
    env.y_train = y_test
    env.param_history.clear()
    env.grad_history.clear()
    env.loss_history.clear()
    env.accuracy_history.clear()

    # Initialize RMSProp optimizer
    optimizer = optim.RMSprop(env.model.parameters(), lr=lr)

    # Get initial parameters, gradients, loss, and accuracy
    params = env.get_params()
    gradients = env.compute_gradient()
    loss = env.criterion(env.model(env.X_train), env.y_train).item()
    accuracy = env.compute_accuracy()

    # Initialize zero history for previous steps
    zero_params = np.zeros_like(params)
    zero_gradients = np.zeros_like(gradients)
    zero_loss = 0.0
    zero_acc = 0.0

    for _ in range(4):
        env.param_history.append(zero_params)
        env.grad_history.append(zero_gradients)
        env.loss_history.append(zero_loss)
        env.accuracy_history.append(zero_acc)

    env.param_history.append(params)
    env.grad_history.append(gradients)
    env.loss_history.append(loss)
    env.accuracy_history.append(accuracy)

    # Initialize observation state
    obs = env._get_state().astype(np.float32)

    # Store results
    loss_values = [env.loss_history[-1]]
    acc_values = [env.accuracy_history[-1]]

    for i in range(num_steps):
        if i < switch_point:
            action, _ = agent.predict(obs, deterministic=True)
            obs, reward, done, _ = env.step(action)
            obs = obs.astype(np.float32)
        else:
            optimizer.zero_grad()
            outputs = env.model(env.X_train)
            loss = env.criterion(outputs, env.y_train)
            loss.backward()
            optimizer.step()

            params = env.get_params()
            gradients = env.compute_gradient()
            accuracy = env.compute_accuracy()
            loss_value = loss.item()

            env.param_history.append(params)
            env.grad_history.append(gradients)
            env.loss_history.append(loss_value)
            env.accuracy_history.append(accuracy)

            obs = env._get_state().astype(np.float32)

        loss_values.append(env.loss_history[-1])
        acc_values.append(env.accuracy_history[-1])

        if done:
            break

    return loss_values, acc_values


def test_adam(X_test, y_test, base_model, lr=0.01, num_steps=200):
    adam_model = base_model
    adam_loss_values = []
    adam_acc_values = []
    adam_optimizer = optim.Adam(adam_model.parameters(), lr=lr)
    
    with torch.no_grad():
        initial_outputs = adam_model(X_test)
        initial_loss = nn.BCELoss()(initial_outputs, y_test).item()
        initial_predictions = (initial_outputs >= 0.5).float()
        initial_accuracy = (initial_predictions == y_test).float().mean().item()
    
    adam_loss_values.append(initial_loss)
    adam_acc_values.append(initial_accuracy)
    
    for epoch in range(num_steps):  
        adam_optimizer.zero_grad()
        outputs = adam_model(X_test)
        loss = nn.BCELoss()(outputs, y_test)
        loss.backward()
        adam_optimizer.step()
        adam_loss_values.append(loss.item())
        
        predictions = (outputs >= 0.5).float()
        accuracy = (predictions == y_test).float().mean().item()
        adam_acc_values.append(accuracy)

    return adam_loss_values, adam_acc_values


def test_rmsprop(X_test, y_test, base_model, lr=0.01, alpha=0.99, num_steps=200):
    """
    Test the base model using RMSProp optimizer on the given test set.

    Parameters:
    - X_test: Input features (tensor)
    - y_test: True labels (tensor)
    - base_model: PyTorch model to be optimized
    - lr: Learning rate for RMSProp (default 0.01)
    - alpha: Smoothing constant for RMSProp (default 0.99)
    - num_steps: Number of optimization steps (default 200)

    Returns:
    - rmsprop_loss_values: List of loss values per step
    - rmsprop_acc_values: List of accuracy values per step
    """
    rmsprop_model = base_model
    rmsprop_loss_values = []
    rmsprop_acc_values = []
    rmsprop_optimizer = optim.RMSprop(rmsprop_model.parameters(), lr=lr, alpha=alpha)
    
    with torch.no_grad():
        initial_outputs = rmsprop_model(X_test)
        initial_loss = nn.BCELoss()(initial_outputs, y_test).item()
        initial_predictions = (initial_outputs >= 0.5).float()
        initial_accuracy = (initial_predictions == y_test).float().mean().item()
    
    rmsprop_loss_values.append(initial_loss)
    rmsprop_acc_values.append(initial_accuracy)
    
    for epoch in range(num_steps):  
        rmsprop_optimizer.zero_grad()
        outputs = rmsprop_model(X_test)
        loss = nn.BCELoss()(outputs, y_test)
        loss.backward()
        rmsprop_optimizer.step()
        rmsprop_loss_values.append(loss.item())
        
        predictions = (outputs >= 0.5).float()
        accuracy = (predictions == y_test).float().mean().item()
        rmsprop_acc_values.append(accuracy)

    return rmsprop_loss_values, rmsprop_acc_values


def testing(testing_set, agent, base_model):
    env = LogisticRegressionEnv(testing_set)
    num_datasets = len(testing_set)
    losses_sum = np.zeros(200)
    acc_sum = np.zeros(200)

    for data in testing_set:
        X_test, y_test = data
        loss, acc = test_l2o_agent(env, X_test, y_test, copy.deepcopy(base_model), agent)

        loss = np.array(loss[:200])  
        acc = np.array(acc[:200])

        if len(loss) < 200:
            loss = np.pad(loss, (0, 200 - len(loss)), 'edge')
        if len(acc) < 200:
            acc = np.pad(acc, (0, 200 - len(acc)), 'edge')

        losses_sum += loss
        acc_sum += acc

    avg_losses = losses_sum / num_datasets
    avg_acc = acc_sum / num_datasets

    return avg_losses, avg_acc

def testing_adam(testing_set, base_model, lr=0.01, num_steps=200):
    num_datasets = len(testing_set)

    losses_sum = np.zeros(200)
    acc_sum = np.zeros(200)

    for X_test, y_test in testing_set:
        adam_loss, adam_acc = test_adam(X_test, y_test, copy.deepcopy(base_model), lr, num_steps)
        losses_sum += np.array(adam_loss)[:200]
        acc_sum += np.array(adam_acc)[:200]

    return losses_sum / num_datasets, acc_sum / num_datasets


def generate_dataset(n = 5):
    datasets = []
    for i in range(n):
        X, y = make_classification(n_samples=200, n_features=2, n_informative=2, n_redundant=0,
                                n_clusters_per_class=1, random_state=i)
        X = StandardScaler().fit_transform(X)
        y = torch.tensor(y, dtype=torch.float32).view(-1, 1)
        X = torch.tensor(X, dtype=torch.float32)
        datasets.append((X, y))
    return datasets