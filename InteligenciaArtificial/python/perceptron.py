import numpy as np


class Perceptron:
    """
    Attributes
    ----------
    data: np.ndarray
        data that entered the perceptron.
    weights: np.ndarray
        Numpy array withn the weights.
    localfield: float
        Local field of the weights and the data.
    output: float
        output of the perceptron.
    phiF: str
        Name of the activation function.
    """

    def __init__(self, phiF: str = "linear") -> None:
        """
        Parameters
        ----------
        phi: str
            Name of the activation function.
        """

        self.data: np.ndarray
        self.weights: np.ndarray
        self.localfield: float
        self.output: float
        self.phiF: str = phiF

    def loss(self, target: float, output: float) -> float:
        """Loss function

        Parameters
        ----------
        target: float
            Desired value for the perceptron output.
        output: float
            Perceptron output.

        Returns
        -------
        float
           The error between the target and the output.
        """

        return (target - output) ** 2 / 2

    def diff_loss(self, target: float, output: float) -> float:
        """Derivative of the loss function.

        Parameters
        ----------
        target: float
            Desired value for the perceptron output.
        output: float
            Perceptron output.

        Returns
        -------
        float
            The derivative of the error between the target and the output
        """

        return -(target - output)

    def phi(self, localfield) -> float:
        """Activation function.

        Parameters
        ----------
        localfield: float
            Localfield between the weights and the data.

        Returns
        -------
        The activation function evaluated in the localfield
        """

        if self.phiF == "linear":
            return localfield
        elif self.phiF == "sigmoid":
            return 1 / (1 + np.exp(-localfield))

    def diff_phi(self, localfield: float) -> float:
        """Derivative of the activation function.

        Parameters
        ----------
        localfield: float
            Localfield between the weights and the data.

        Returns
        -------
        The derivative of the activation function evaluated in the localfield
        """

        if self.phiF == "linear":
            return 1
        elif self.phiF == "sigmoid":
            return self.phi(localfield) * (1 - self.phi(localfield))

    def gradient(self, target: float, output: float) -> np.ndarray:
        """Calculates the gradient for the current state of the perceptron.

        Parameters
        ----------
        target: float
            Desired value for the perceptron output.
        output: float
            Perceptron output.

        Returns
        -------
        np.ndarray
            Numpy array containing the gradient values for each variable.
        """

        return np.array(
            [
                self.diff_loss(target, output) * self.diff_phi(self.localfield) * data
                for data in self.data
            ]
        )

    def forward(self, data: np.ndarray, weights: np.ndarray) -> float:
        """Forward pass off the perceptron.

        Parameters
        ----------
        data: np.ndarray
            data to forward in the perceptron.
        weights: np.ndarray
            weights to calculate the ouput.

        Returns
        -------
        float
            Output of the perceptron.
        """
        self.data = data
        self.localfield = np.dot(weights, self.data)
        self.output = self.phi(self.localfield)
        return self.output
