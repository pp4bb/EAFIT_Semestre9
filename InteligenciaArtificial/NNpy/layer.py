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
        self.in_features = in_features
        self.out_features = out_features
        self.weights = np.random.rand(in_features, out_features)
        if bias:
            self.bias = np.random.rand(out_features)
        else:
            self.bias = np.zeros(out_features)

        # Initialice the functions.
        self.phi = None
        self.dphi = None

    @classmethod
    def forward(cls, x: np.ndarray) -> np.ndarray:
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
        return self.phi(np.dot(x, self.weights) + self.bias)

    @classmethod
    def backward(self, x: np.ndarray) -> np.ndarray:
        """Backward pass of the layer.
        Parameters
        ----------
        x: np.ndarray
            Input data.

        Returns
        -------
        np.ndarray
            Gradient of the weights.
        """
        localfield = np.dot(x, self.weights) + self.bias
        return self.dphi(localfield) * np.dot(x.T, self.weights)

    @classmethod
    def step(self, x: np.ndarray, lr: float, gradient: np.array) -> None:
        """Update the weights of the layer.
        Parameters
        ----------
        x: np.ndarray
            Input data.
        lr: float
            Learning rate.
        gradient: np.ndarray
            Gradient of the weights.
        """
        self.weights += lr * gradient
