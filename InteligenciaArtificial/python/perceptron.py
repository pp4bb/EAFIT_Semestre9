import numpy as numpy


class Perceptron:
    """
    Attributes
    ----------
    inputs: np.ndarray
        Numpy array with the inputs
    weights: np.ndarray
        Numpy array withn the weights
    phi: function
        Activation function
    """

    def __init__(self, inputs: np.ndarray, weights: np.ndarray, phi: np.ndarray):
        """
        Parameters
        ----------
        inputs: np.ndarray
            Numpy array with the inputs
        weights: np.ndarray
            Numpy array withn the weights
        phi: function
            Activation function
        """

        self.inputs: np.ndarray = inputs
        self.weights: np.ndarray = weights
        self.phi: function = phi
