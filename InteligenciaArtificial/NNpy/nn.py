"""

"""
from layer import Layer
import functional as F
import numpy as np


class Sequential:
    def __init__(self) -> None:
        pass


class Linear(Layer):
    """
    Attributes
    ----------
    in_features: int
        Number of input features.
    out_features: int
        Number of output features.
    weights: np.ndarray
        Numpy array with the weights.
    bias: np.ndarray
        Numpy array with the bias.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = False) -> None:
        """
        Parameters
        ----------
        in_features: int
            Number of input features.
        out_features: int
            Number of output features.
        bias: bool
            If True, the layer will have a bias.
        """
        super().__init__(in_features, out_features, bias)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass of the linear layer.
        Parameters
        ----------
        x: np.ndarray
            Input data.

        Returns
        -------
        np.ndarray
            Output data.
        """
        return F.linear(x, self.weights, self.bias)
