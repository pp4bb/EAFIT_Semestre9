"""This module contains the functional API of the library. It contains the
loss functions, activation functions and its derivatives, which are used
in the layers and the neural networks.
"""

# libraries
import numpy as np

""" Loss functions

List of loss functions and their derivatives:
    - insta_energy
    - dinsta_energy
    - mse
    - dmse
"""


def insta_energy(y: np.ndarray, y_hat: np.ndarray) -> float:
    """Instantaneous energy loss function.

    Parameters
    ----------
    y: np.ndarray
        Target array.
    y_hat: np.ndarray
        Predicted array.

    Returns
    -------
    float
        Instantaneous energy.
    """
    return np.sum((y - y_hat) ** 2) / 2


def dinsta_energy(y: np.ndarray, y_hat: np.ndarray) -> np.ndarray:
    """Derivative of the instantaneous energy loss function.

    Parameters
    ----------
    y: np.ndarray
        Target array.
    y_hat: np.ndarray
        Predicted array.

    Returns
    -------
    np.ndarray
        Derivative of the instantaneous energy.
    """
    return y_hat - y


def mse(y: np.ndarray, y_hat: np.ndarray) -> float:
    """Mean squared error loss function.

    Parameters
    ----------
    y: np.ndarray
        Target array.
    y_hat: np.ndarray
        Predicted array.

    Returns
    -------
    float
        Mean squared error.
    """
    return np.mean((y - y_hat) ** 2)


def dmse(y: np.ndarray, y_hat: np.ndarray) -> np.ndarray:
    """Derivative of the mean squared error loss function.

    Parameters
    ----------
    y: np.ndarray
        Target array.
    y_hat: np.ndarray
        Predicted array.

    Returns
    -------
    np.ndarray
        Derivative of the mean squared error.
    """
    return 2 * (y_hat - y) / y.size


"""Activation functions

List of activation functions and their derivatives:
    - linear
    - dlinear
    - sigmoid
    - dsigmoid
"""


def linear(x: np.ndarray, m: float, b: float) -> np.ndarray:
    """Linear activation function.

    Parameters
    ----------
    x: np.ndarray
        Input data.
    m: np.ndarray
        Slope.
    b: np.ndarray
        Intercept.
    """
    return m * x + b


def dlinear(x: np.ndarray, m: float, b: float) -> np.ndarray:
    """Derivative of the linear activation function.

    Parameters
    ----------
    x: np.ndarray
        Input data.
    m: np.ndarray
        Slope array.
    b: np.ndarray
        Intercept array.
    """
    return m


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Sigmoid activation function.

    Parameters
    ----------
    x: np.ndarray
        Input data.
    """
    return 1 / (1 + np.exp(-x))


def dsigmoid(x: np.ndarray) -> np.ndarray:
    """Derivative of the sigmoid activation function.

    Parameters
    ----------
    x: np.ndarray
        Input data.
    """
    return sigmoid(x) * (1 - sigmoid(x))
