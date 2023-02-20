from nn_design import NeuralNetwork
import numpy as np


def load_data() -> tuple:
    """Load the data."""
    data = np.loadtxt("data/DATOS.txt", comments="#", delimiter=",", unpack=False)
    all_inputs = []
    all_outputs = []
    for i in range(len(data)):
        all_inputs.append(data[i][1:])
        all_outputs.append(data[i][0])
    # Split in train, test and validation.
    train_inputs = all_inputs[: int(0.6 * len(all_inputs))]
    train_outputs = all_outputs[: int(0.6 * len(all_outputs))]
    test_inputs = all_inputs[int(0.6 * len(all_inputs)) : int(0.8 * len(all_inputs))]
    test_outputs = all_outputs[
        int(0.6 * len(all_outputs)) : int(0.8 * len(all_outputs))
    ]
    validation_inputs = all_inputs[int(0.8 * len(all_inputs)) :]
    validation_outputs = all_outputs[int(0.8 * len(all_outputs)) :]
    # Assemble the data.
    train = (train_inputs, train_outputs)
    test = (test_inputs, test_outputs)
    validation = (validation_inputs, validation_outputs)
    return train, test, validation


def main():
    train, test, validation = load_data()
    MLP = NeuralNetwork(hidden_layers=3, hidden_neurons=5)
    losses, deltas = MLP.fit(train[0], train[1], epochs=1, lr=0.2, tolerance=1e-2)


if __name__ == "__main__":
    main()
