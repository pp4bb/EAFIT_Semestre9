from perceptron import Perceptron
import numpy as np


class MultilayerPerceptron:
    """
    Attributes
    ----------
    layer_num: int
        Number of layer.
    neurons_by_layer: list[int]
        List of length `layer_num` that indicates the number of neurons for each layer
        (represented by the index).
    phiF: str
        Name of the activation function.
    perceptrons: list
        list of lists containg the perceptrons for each layer.
    """

    def __init__(
        self,
        layer_num: int = 3,
        neurons_by_layer: list[int] = [2, 1, 1],
        phiF: str = "sigmoid",
        hiddenF: str = "sigmoid",
    ) -> None:
        """
        Parameters
        ----------
        layer_num: int
            Number of layer.
        neurons_by_layer: list[int]
            List of length `layer_num` that indicates the number of neurons for each layer
            (represented by the index).
        phiF: str
            Name of the activation function.
        """

        assert layer_num == len(neurons_by_layer)
        self.layer_num = layer_num
        self.neurons_by_layer = neurons_by_layer
        self.phiF = phiF
        self.hiddenF = hiddenF
        self.perceptrons: list = []
        self.__init__perceptrons__(layer_num, neurons_by_layer)

    def __init__perceptrons__(
        self, layer_num: int = 3, neurons_by_layer: list[int] = [2, 1, 1]
    ) -> None:
        """Initialices the percetrons list

        Parameters
        ----------
        layer_num: int
            Number of layer.
        neurons_by_layer: list[int]
            List of length `layer_num` that indicates the number of neurons for each layer
            (represented by the index).
        """

        for layer in range(layer_num):
            perceptron_array = []
            for _ in range(neurons_by_layer[layer]):
                if layer == 1:
                    perceptron_array.append(Perceptron(phiF=self.hiddenF))
                else:
                    perceptron_array.append(Perceptron(phiF=self.phiF))

            self.perceptrons.append(perceptron_array)

    def forward(self, data: np.ndarray, weights: list[np.ndarray]) -> float:
        """Forward pass

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

        # First layer
        x3 = self.perceptrons[0][0].forward(data, weights[0][:, 0])
        x4 = self.perceptrons[0][1].forward(data, weights[0][:, 1])
        # Second layer
        data = np.array([x3, x4])
        x5 = [self.perceptrons[1][0].forward(data, weights[1])]  # As list
        # Third layer
        output = self.perceptrons[2][0].forward(x5, weights[2])
        return output

    def fit(
        self,
        all_inputs: list,
        all_outputs: list,
        epochs: int = 10000,
        lr: float = 0.01,
    ):
        """
        Fits the model to the data and the targets, performs gradient descend

        Parameters
        ----------
        all_inputs: list
            The input data
        """

        weights = [np.random.rand(2, 2), np.random.rand(2), np.random.rand(1)]
        meanlosses = []
        deltas = []
        for _ in range(epochs):
            losses = []
            gradients = []
            for data, target in zip(all_inputs, all_outputs):
                output = self.forward(data, weights)
                loss = self.perceptrons[2][0].loss(target, output)
                gradient = self.perceptrons[2][0].gradient(target, output)
                # Store loss and gradient
                losses.append(loss)
                gradients.append(gradient)

            # Calculate means by epoch
            meanloss = sum(losses) / len(losses)
            meanlosses.append(meanloss)
            meangrad = np.array(
                [
                    sum([gradient[i] for gradient in gradients]) / len(gradients)
                    for i in range(len(gradients[0]))
                ]
            )

            delta = lr * -meangrad
            weights[-1] += delta

            deltas.append(-meangrad)

        return weights, meanlosses, deltas
