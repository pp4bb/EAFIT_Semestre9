"""This module contains the functional API of the library. It contains the
loss functions, activation functions and its derivatives, which are used
in the layers and the neural networks.
"""

# libraries
import numpy as np

########################################################################################
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


########################################################################################
"""Activation functions

List of activation functions and their derivatives:
    - linear
    - dlinear
    - sigmoid
    - dsigmoid
    - relu
    - drelu
"""


def linear(x: np.ndarray, m: float, b: float, *args) -> np.ndarray:
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


def dlinear(x: np.ndarray, m: float, *args) -> np.ndarray:
    """Derivative of the linear activation function.

    Parameters
    ----------
    x: np.ndarray
        Input data.
    m: np.ndarray
        Slope array.
    """
    return m


def sigmoid(x: np.ndarray, *args) -> np.ndarray:
    """Sigmoid activation function.

    Parameters
    ----------
    x: np.ndarray
        Input data.
    """
    return 1 / (1 + np.exp(-x))


def dsigmoid(x: np.ndarray, *args) -> np.ndarray:
    """Derivative of the sigmoid activation function.

    Parameters
    ----------
    x: np.ndarray
        Input data.
    """
    return sigmoid(x) * (1 - sigmoid(x))


def relu(x: np.ndarray, *args) -> np.ndarray:
    """Rectified Linear Unit activation function.

    Parameters
    ----------
    x: np.ndarray
        Input data.
    """
    return np.maximum(0, x)


def drelu(x: np.ndarray, *args) -> np.ndarray:
    """Derivative of the Rectified Linear Unit activation function.

    Parameters
    ----------
    x: np.ndarray
        Input data.
    """
    return np.where(x > 0, 1, 0)
