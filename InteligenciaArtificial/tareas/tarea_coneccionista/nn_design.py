import NNpy.nn as nn
import NNpy.functional as F
import numpy as np
from tqdm import tqdm


class NeuralNetwork:
    """Design 1 of the neural network.

    Attributes
    ----------
    fcn: nn.Sequential
        Neural network.
    """

    def __init__(
        self,
        in_features,
        hidden_layers: int = 1,
        hidden_neurons: int = 1,
        phi: str = "sigmoid",
    ) -> None:
        """
        Parameters
        ----------
        hidden_layers: int
            Number of hidden layers.
        hidden_neurons: int
            Number of neurons in each hidden layer.
        phi: str
            Activation function.
        """

        if phi == "sigmoid":
            self.fcn = nn.Sequential(
                nn.Sigmoid(in_features, hidden_neurons),
                *[
                    nn.Sigmoid(hidden_neurons, hidden_neurons)
                    for _ in range(hidden_layers)
                ],
                nn.Sigmoid(hidden_neurons, 1, last=True),
            )

        elif phi == "linear":
            self.fcn = nn.Sequential(
                nn.Linear(in_features, hidden_neurons, 1, 0.5),
                *[
                    nn.Linear(hidden_neurons, hidden_neurons, 1, 0.5)
                    for _ in range(hidden_layers)
                ],
                nn.Linear(hidden_neurons, 1, True, 1, 0.5),
            )

    def forward(self, x) -> np.ndarray:
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
        return self.fcn.forward(x)

    def fit(self, x, y, epochs, lr, tolerance: float = 1e-2) -> tuple:
        """Fit the neural network to the data.

        Parameters
        ----------
        x: np.ndarray
            Input data.
        y: np.ndarray
            Output data.
        epochs: int
            Number of epochs.
        lr: float
            Learning rate.
        tolerance: float
            Tolerance to stop the training.

        Returns
        -------
        tuple
            Tuple with the mean losses and the mean gradients.
        """
        meanlosses = []
        meangradients = []
        for epoch in tqdm(range(epochs), desc="Epochs", leave=False):
            losses = []
            gradients = []
            for i in range(len(x)):
                y_hat = self.fcn.forward(x[i])
                loss = F.mse(y[i], y_hat)
                grad = self.fcn.backward(dloss=y[i] - y_hat)
                # Update the weights.
                self.fcn.step(lr)
                # Store the loss and the gradient.
                losses.append(loss)
                gradients.append(grad)
            # Calculate the mean loss
            meanloss = sum(losses) / len(losses)
            meanlosses.append(meanloss)
            # Calculate the mean gradients
            meangradient = sum(gradients[-1]) / len(gradients[-1])
            meangradients.append(meangradient)

        return np.array(meanlosses), np.array(meangradients)
