import numpy as np


class Layer:
    """
    Attributes
    ----------
    in_features: int
        Number of input features.
    out_features: int
        Number of output features.
    weights: np.ndarray
        Numpy array with the weights.
    gradient: np.ndarray
        Gradient of the weights.
    bias: np.ndarray
        Numpy array with the bias.
    last: bool
        If True, the layer will be the last layer of the neural network.
    phi: function
        Activation function.
    dphi: function
        Derivative of the activation function.
    args: tuple
        Arguments of the activation function.
    data: np.ndarray
        Data that entered the layer.
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
        last: bool
            If True, the layer will be the last layer of the neural network.
        """
        self.in_features = in_features
        self.out_features = out_features
        self.weights = np.random.rand(in_features, out_features)
        if bias:
            self.bias = np.random.rand(out_features)
        else:
            self.bias = np.zeros(out_features)
        self.last = last
        # Initialice the gradient.
        self.gradient = None
        # Initialice the functions.
        self.phi = None
        self.dphi = None
        self.args = args
        # Initialice the data for the backward pass.
        self.data = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass of the layer.
        Parameters
        ----------
        x: np.ndarray
            Input data.

        Returns
        -------
        np.ndarray
            Output data.
        """
        self.data = x
        localfield = self.weights.T @ self.data + self.bias
        return self.phi(localfield, *self.args)

    def backward(self, grad: float, dloss=None) -> np.ndarray:
        """Backward pass of the layer.
        Parameters
        ----------
        grad: np.ndarray
            Gradient of the next layer.
        dloss: np.ndarray
            Loss of the layer. It must be provided only if the layer is the
            output layer.

        Returns
        -------
        np.ndarray
            Gradient of the weights.
        """
        localfield = self.data
        if self.last:
            self.gradient = self.dphi(localfield, *self.args) * dloss
        else:
            self.gradient = self.dphi(localfield, *self.args) * np.sum(
                self.weights * grad, axis=1
            )
        return self.gradient

    def step(self, lr: float) -> None:
        """Updates the weights of the layer.
        Parameters
        ----------
        lr: float
            Learning rate.
        """
        print(self.weights.shape)
        print(self.gradient.shape)
        print(self.data)
        delta = self.gradient * self.data
        print(self.weights, lr, delta)
        self.weights += lr * delta
        # sprint(self.weights)
