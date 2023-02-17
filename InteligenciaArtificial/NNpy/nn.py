"""

"""
from NNpy.layer import Layer
import NNpy.functional as F
import numpy as np


class Sequential:
    """Sequential neural network.

    Attributes
    ----------
    layers: list
        List of layers.
    """

    def __init__(self, *args) -> None:
        """
        Parameters
        ----------
        *args: Layer
            Layers of the neural network.
        """
        self.layers = list(args)
        self.__check__dims__()

    def __check__dims__(self):
        """Check if the dimensions of the layers are correct."""
        for i in range(len(self.layers) - 1):
            assert self.layers[i].out_features == self.layers[i + 1].in_features

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass of the neural network.

        Parameters
        ----------
        x: np.ndarray
            Input data.

        Returns
        -------
        np.ndarray
            Output of the neural network.
        """

        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, loss: float) -> None:
        """Backward pass of the neural network.

        Parameters
        ----------
        loss: float
            Loss of the neural network.
        """
        x = 0
        gradients = []
        for layer in self.layers[::-1]:
            x = layer.backward(x, loss)
            gradients.append(x)
        return gradients[::-1]

    def step(self, lr: float) -> None:
        """Update the weights of the neural network.

        Parameters
        ----------
        lr: float
            Learning rate.
        """
        for layer in self.layers:
            layer.step(lr)


class Linear(Layer):
    """Linear layer.
    See Also :class:`Layer`
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        last: bool = False,
        args: tuple = (),
    ) -> None:
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
        super().__init__(in_features, out_features, bias, last, args)
        self.phi = F.linear
        self.dphi = F.dlinear


class Sigmoid(Layer):
    """Sigmoid layer.
    See Also :class:`Layer`
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        last: bool = False,
        args: tuple = (),
    ) -> None:
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
        super().__init__(in_features, out_features, bias, last)
        self.phi = F.sigmoid
        self.dphi = F.dsigmoid
