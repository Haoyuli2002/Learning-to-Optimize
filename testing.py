import numpy as np
import matplotlib.pyplot as plt
import torch
from collections import deque
from scipy.interpolate import griddata

def run_optimizer_tests(env, testing_functions, models, max_iterations=200, function_type='quadratic_function_1d', lr = None, dim_param = 2):
    """
    Test the performance of multiple optimizers (PPO, SAC, DDPG, Adam, RMSProp, and SGD) on a set of functions.

    Returns:
        Dictionary containing iteration sums, counts, distances to the true solution, and positions over time.
    """
    param_dim = dim_param
    results = {model_name: {
        'iteration_sums': np.zeros(max_iterations),
        'iteration_counts': np.zeros(max_iterations),
        'total_distance_to_solution': 0,
        'positions': np.zeros((max_iterations, param_dim))
    } for model_name in models.keys()}

    lr_sgd = 0.001
    lr_adam = 0.01
    lr_rmsprop = 0.1

    if function_type == "rastrigin_function":
        lr_sgd = 0.01
        print(lr_sgd)
    elif function_type == 'rosenbrock_function':
        lr_sgd = 0.00001

    for function in testing_functions:
        if function_type == "quadratic_function_nd":
            A, b, c = function
            function = (np.array(A), np.array(b), c)

        env.current_function = function
        env.termination_prob = 0  
        env.max_steps = max_iterations
        initial_state = env.reset()
        initial_positions = list(env.positions)
        initial_gradients = list(env.gradients)
        initial_x = initial_positions[-1]

        initial_function_value = function_value(function, initial_x, function_type)
        for model_name in models.keys():
            results[model_name]['iteration_sums'][0] += initial_function_value
            results[model_name]['iteration_counts'][0] += 1
            results[model_name]['positions'][0] = initial_x

        for model_name, model in models.items():
            if model_name == 'Adam':
                if lr != None:
                    run_adam(env, function, initial_x, max_iterations, results[model_name], function_type, lr = lr_adam)
                else:
                    run_adam(env, function, initial_x, max_iterations, results[model_name], function_type)
            elif model_name == 'RMSProp':
                if lr != None:
                    run_rmsprop(env, function, initial_x, max_iterations, results[model_name], function_type, lr = lr_rmsprop)
                else:
                    run_rmsprop(env, function, initial_x, max_iterations, results[model_name], function_type)
            elif model_name == 'SGD':
                if lr != None:
                    run_sgd(env, function, initial_x, max_iterations, results[model_name], function_type, lr = lr_sgd)
                else:
                    run_sgd(env, function, initial_x, max_iterations, results[model_name], function_type, lr=lr_sgd)
            elif model_name in {'PPO_Adam', 'SAC_Adam', 'DDPG_Adam'}:
                run_rl_then_adam(env, model, initial_state, initial_positions, initial_gradients, max_iterations, results[model_name])
            else:
                run_rl_agent(env, model, initial_state, initial_positions, initial_gradients, max_iterations, results[model_name])

    return results

def function_value(function, x, function_type="quadratic_function_1d"):
    """
    Calculate the function value at the current position.
    """
    is_torch = isinstance(x, torch.Tensor)

    if function_type == "quadratic_function_1d":
        a, b, c = function
        return a * (x ** 2) + b * x + c

    elif function_type == "quadratic_function_nd":
        A, b, c = function
        if is_torch:
            A = torch.tensor(A, dtype=x.dtype, device=x.device)
            b = torch.tensor(b, dtype=x.dtype, device=x.device)
        return x.T @ A @ x + b.T @ x + c

    elif function_type == "rastrigin_function":
        A, offset = function
        if is_torch:
            offset_tensor = torch.tensor(offset, dtype=x.dtype, device=x.device)
            shifted = x - offset_tensor
            return A * x.shape[0] + torch.sum(shifted ** 2 - A * torch.cos(2 * torch.pi * shifted))
        else:
            shifted = x - offset
            return A * len(x) + np.sum(shifted ** 2 - A * np.cos(2 * np.pi * shifted))

    elif function_type == "rosenbrock_function":
        a, b, o1, o2 = function
        if is_torch:
            x0 = x[0] - o1
            x1 = x[1] - o2
            return (a - x0) ** 2 + b * (x1 - x0 ** 2) ** 2
        else:
            x0 = x[0] - o1
            x1 = x[1] - o2
            return (a - x0) ** 2 + b * (x1 - x0 ** 2) ** 2

    else:
        raise ValueError("Unsupported function type.")

def analytical_solution(function, function_type="quadratic_function_1d"):
    """
    Calculate the analytical solution for different functions.
    """
    if function_type == "quadratic_function_1d":
        a, b, c = function
        x = -b / (2 * a)
        return a * (x**2) + b * x + c
    elif function_type == "quadratic_function_nd":
        A, b, c = function
        x = -np.linalg.solve(A, b/2)
        return x.T @ A @ x + b.T @ x + c
    elif function_type == "rastrigin_function":
        _, offset = function
        return offset
    elif function_type == "rosenbrock_function":
        _, offset = function
        return offset
    else:
        raise ValueError("Unsupported function type.")

def run_rl_agent(env, model, initial_state, initial_positions, initial_gradients, max_iterations, result):
    """
    Run reinforcement learning agent optimization and store positions over time.
    """
    env.reset()
    env.positions, env.gradients = deque(initial_positions, maxlen=5), deque(initial_gradients, maxlen=5)
    obs = initial_state.copy()
    env.max_steps = max_iterations
    for i in range(1, max_iterations):
        decay = min(1, 2-i/100)
        action, _ = model.predict(obs)
        obs, reward, done, _ = env.step(action*decay)
        result['iteration_sums'][i] += env.function_values[-1]
        result['iteration_counts'][i] += 1
        result['positions'][i] = env.positions[-1]
        if done:
            break
    return env.function_values[-1]

def run_rl_then_adam(env, model, initial_state, initial_positions, initial_gradients, max_iterations, result):
    """
    Runs RL agent for the first 20 steps, then switches to Adam optimizer for the remaining steps.
    """
    env.reset()
    env.positions, env.gradients = deque(initial_positions, maxlen=5), deque(initial_gradients, maxlen=5)
    obs = initial_state.copy()
    
    switch_point = 25

    # Phase 1: RL Agent (First 25 steps)
    for i in range(1, switch_point + 1):
        action, _ = model.predict(obs)
        obs, reward, done, _ = env.step(action)
        result['iteration_sums'][i] += env.function_values[-1]
        result['iteration_counts'][i] += 1
        result['positions'][i] = env.positions[-1]
        if done:
            break

    # Phase 2: Adam Optimizer (Remaining steps)
    x = torch.tensor(env.positions[-1], dtype=torch.float32, requires_grad=True)
    optimizer = torch.optim.RMSprop([x], lr=0.1)

    for i in range(switch_point + 1, max_iterations):
        optimizer.zero_grad()
        loss = function_value(env.current_function, x, 'quadratic_function_nd')
        loss.backward()
        optimizer.step()
        result['iteration_sums'][i] += loss.item()
        result['iteration_counts'][i] += 1
        result['positions'][i] = x.detach().numpy()

    return x.detach().numpy()


def run_adam(env, function, initial_x, max_iterations, result, function_type, lr = 0.1):
    x = torch.tensor(initial_x, dtype=torch.float32, requires_grad=True)
    optimizer = torch.optim.Adam([x], lr=lr)
    for i in range(1, max_iterations):
        optimizer.zero_grad()
        loss = function_value(function, x, function_type)
        loss.backward()
        optimizer.step()
        result['iteration_sums'][i] += loss.item()
        result['iteration_counts'][i] += 1
        result['positions'][i] = x.detach().numpy()
    return x.detach().numpy()

def run_rmsprop(env, function, initial_x, max_iterations, result, function_type, lr = 0.1):
    x = torch.tensor(initial_x, dtype=torch.float32, requires_grad=True)
    optimizer = torch.optim.RMSprop([x], lr=lr, alpha=0.99, eps=1e-08)
    for i in range(1, max_iterations):
        optimizer.zero_grad()
        loss = function_value(function, x, function_type)
        loss.backward()
        optimizer.step()
        result['iteration_sums'][i] += loss.item()
        result['iteration_counts'][i] += 1
        result['positions'][i] = x.detach().numpy()
    return x.detach().numpy()

def run_sgd(env, function, initial_x, max_iterations, result, function_type, lr = 0.01):
    x = torch.tensor(initial_x, dtype=torch.float32, requires_grad=True)
    optimizer = torch.optim.SGD([x], lr=lr)
    for i in range(1, max_iterations):
        optimizer.zero_grad()
        loss = function_value(function, x, function_type)
        loss.backward()
        optimizer.step()
        result['iteration_sums'][i] += loss.item()
        result['iteration_counts'][i] += 1
        result['positions'][i] = x.detach().numpy()
    return x.detach().numpy()

def calculate_average_values(results):
    """
    Compute the average function values at each iteration.
    """
    avg_values = {}
    for model_name, data in results.items():
        avg_values[model_name] = np.divide(
            data['iteration_sums'], data['iteration_counts'],
            out=np.zeros_like(data['iteration_sums']), where=data['iteration_counts'] > 0
        )
    return avg_values

def plot_results(avg_values, max_iterations, save_path = None):
    """
    Plot optimization progress.
    """
    plt.figure(figsize=(12, 8))
    for model_name, values in avg_values.items():
        plt.plot(range(max_iterations), values, label=f"{model_name} Average Function Value")
    plt.title("Comparison of Optimizers", fontsize=16)
    plt.xlabel("Iteration", fontsize=14)
    plt.ylabel("Average Function Value", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def plot_final_function_values(comparison, save_path = None):
    """
    Plot a bar chart comparing the final function values across optimizers.

    Parameters:
    - comparison: List of tuples (model_name, final_function_value)
    """
    optimizers = [item[0] for item in comparison]
    final_values = [item[1] for item in comparison]

    plt.figure(figsize=(12, 5))

    categories = {
        "RL-based": ["PPO", "SAC", "DDPG"],
        "Gradient-based": ["Adam", "RMSProp", "SGD"]
    }

    color_map = {
        "PPO": "blue",
        "SAC": "orange",
        "DDPG": "green",
        "Adam": "brown",
        "RMSProp": "purple",
        "SGD": "red"
    }

    bar_colors = [color_map.get(opt, "gray") for opt in optimizers]

    bars = plt.bar(optimizers, final_values, color = bar_colors, label="Final Function Value")
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height, f'{height:.2f}', 
                 ha='center', va='bottom', fontsize=12)

    plt.xlabel("Optimizers")
    plt.ylabel("Final Average Function Values")
    plt.title("Final Accuracy Comparison Across Optimizers")
    plt.grid(False)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()


def plot_contour_with_trajectory(results, model_name, levels=20):
    """
    Creates a contour plot using function values and overlays the optimization trajectory.
    
    - Contour levels are created using real function values.
    - The trajectory is plotted with markers from start to end.
    - A colorbar represents the function values.
    """
    positions = np.array(results[model_name]['positions'])
    function_values = np.array(results[model_name]['iteration_sums'])

    grid_x, grid_y = np.meshgrid(
        np.linspace(positions[:, 0].min(), positions[:, 0].max(), 100),
        np.linspace(positions[:, 1].min(), positions[:, 1].max(), 100)
    )
    grid_z = griddata((positions[:, 0], positions[:, 1]), function_values, (grid_x, grid_y), method='cubic')

    # Create contour plot
    plt.figure(figsize=(8, 6))
    contour = plt.contourf(grid_x, grid_y, grid_z, levels=levels, cmap="coolwarm", alpha=0.75)
    plt.colorbar(contour, label="Function Value")

    # Overlay optimization trajectory
    plt.plot(positions[:, 0], positions[:, 1], 'r-', linewidth=1, alpha=0.8, label="Optimization Path")

    # Mark start and end points
    plt.scatter(positions[0, 0], positions[0, 1], c="green", marker="*", s=100, label="Start Point", edgecolors='black')
    plt.scatter(positions[-1, 0], positions[-1, 1], c="red", marker="*", s=100, label="End Point", edgecolors='black')

    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.title(f"Contour Plot with Optimization Path ({model_name})")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.show()