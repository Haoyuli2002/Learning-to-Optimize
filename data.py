import random
import numpy as np
from typing import Union, Tuple

def generate_a_random_function(
    min_coeff: int = -10,
    max_coeff: int = 10,
    function_type: str = 'quadratic_function_1d',
    n_dims: int = 1
) -> Union[Tuple[int, int, int], Tuple[list, list, float], Tuple[float, float]]:
    """
    Generates a random function according to the specified function type.

    Parameters:
        min_coeff (int): Minimum value for the coefficients (inclusive).
        max_coeff (int): Maximum value for the coefficients (inclusive).
        function_type (str): The type of function ('quadratic_function_1d', 'quadratic_function_nd', 'rastrigin_function', or 'rosenbrock').
        n_dims (int): The dimensions of the input for the function (relevant for n-dimensional functions).

    Returns:
        Coefficients and parameters for the specified function type.
    """
    if function_type == 'quadratic_function_1d':
        # ax^2 + bx + c
        a = random.randint(min_coeff, max_coeff)
        while a <= 0:  # Ensure a > 0 for a valid quadratic
            a = random.randint(min_coeff, max_coeff)

        b = random.randint(min_coeff, max_coeff)
        c = random.randint(min_coeff, max_coeff)
        return (a, b, c)

    elif function_type == 'quadratic_function_nd':
        # x^T A x + b^T x + c
        Q = np.random.uniform(min_coeff, max_coeff, (n_dims, n_dims))
        A = np.dot(Q, Q.T)  # Symmetric positive semi-definite matrix
        b = np.random.uniform(min_coeff, max_coeff, n_dims)
        c = float(np.random.uniform(min_coeff, max_coeff))
        return (A, b, c)

    elif function_type == 'rastrigin_function':
        # A * n + sum((xi-offset)^2 - A * cos(2π * (xi-offset)))
        A = random.uniform(min_coeff, max_coeff)
        offset = np.random.randn(n_dims)
        return (A, offset)

    elif function_type == 'rosenbrock':
        # Typically, a is 1
        a = random.uniform(0, 3)
        # Typically, b is 100
        b = random.uniform(50, 150)  
        # Ensuring the minimum is not always 0
        o1 = np.random.randn()
        o2 = np.random.randn()
        return (a, b, o1, o2)

    else:
        raise ValueError(f"Unsupported function type: {function_type}")

def generate_dataset(
    n: int = 100,
    min_coeff: int = -10,
    max_coeff: int = 10,
    function_type: str = 'quadratic_function_1d',
    n_dims: int = 1
):
    """
    Generate a dataset of random functions for analysis.

    Parameters:
        n (int): Number of functions to generate.
        min_coeff (int): Minimum value for the coefficients (inclusive).
        max_coeff (int): Maximum value for the coefficients (inclusive).
        function_type (str): The type of function ('quadratic_function_1d', 'quadratic_function_nd', or 'rastrigin_function').
        n_dims (int): The dimensions of the input for the function.

    Returns:
        set: A set of coefficients for the specified function type.
    """
    functions = []
    while len(functions) < n:
        coefficients = generate_a_random_function(min_coeff, max_coeff, function_type, n_dims)
        functions.append(coefficients)
    return functions