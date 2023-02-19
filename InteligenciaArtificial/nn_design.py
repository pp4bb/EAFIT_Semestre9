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

    def __init__(self, hidden_layers: int = 1, hidden_neurons: int = 1) -> None:
        """
        Parameters
        ----------
        hidden_layers: int
            Number of hidden layers.
        hidden_neurons: int
            Number of neurons in each hidden layer.
        """
        self.fcn = nn.Sequential(
            nn.Sigmoid(2, hidden_neurons),
            *[nn.Sigmoid(hidden_neurons, hidden_neurons) for _ in range(hidden_layers)],
            nn.Sigmoid(hidden_neurons, 1, last=True),
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
        meangradients = [[] for layer in self.fcn.layers]
        for epoch in tqdm(range(epochs), desc="Epochs", leave=False):
            losses = []
            gradients = []
            for i in range(len(x)):
                y_hat = self.fcn.forward(x[i])
                loss = F.insta_energy(y[i], y_hat)
                dloss = F.dinsta_energy(y[i], y_hat)
                grad = self.fcn.backward(dloss=dloss)
                # Update the weights.
                self.fcn.step(lr)
                # Store the loss and the gradient.
                losses.append(loss)
                gradients.append(grad)
            # Calculate the mean loss
            meanloss = sum(losses) / len(losses)
            meanlosses.append(meanloss)
            # Calculate the mean gradients for each layer.
            for i in range(len(meangradients)):
                meangradients[i].append(
                    np.sum([grad[i] for grad in gradients]) / len(gradients)
                )
            # Check if the loss is less than the tolerance.
            if losses[-1] < tolerance:
                break
            # Check if the gradients are still decreasing.
            if len(meangradients[0]) > 1 and all(
                [
                    meangradients[i][-1] > meangradients[i][-2]
                    for i in range(len(meangradients))
                ]
            ):
                break

        return np.array(meanlosses), np.array(meangradients)
