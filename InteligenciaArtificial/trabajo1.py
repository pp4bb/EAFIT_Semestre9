import matplotlib.pyplot as plt
from nn_design import NeuralNetwork
import numpy as np
from tqdm import tqdm


def load_data() -> tuple:
    """Load the data."""
    data = np.loadtxt("data/DATOS.txt", comments="#", delimiter=",", unpack=False)
    all_inputs, all_outputs = [], []
    # Store the desired data.
    for i in range(300):
        all_inputs.append(np.array([data[i, 0], data[i, 1]]))
        all_outputs.append(data[i, 2])
    return all_inputs, all_outputs


def plot_results(experiments_L, experiments_G) -> None:
    """Plot the results."""
    # Initialize the lists.
    hidden_layers = [1, 2, 3]
    hidden_neurons = [1, 2, 3, 4, 5]
    learning_rates = [0.2, 0.5, 0.9]
    # Iterate over the parameters.
    for hidden_layer in hidden_layers:
        for hidden_neuron in hidden_neurons:
            for learning_rate in learning_rates:
                # Initialize the lists.
                losses = []
                gradients = []
                # Iterate over the experiments.
                for experiment_L, experiment_G in zip(experiments_L, experiments_G):
                    # Append the results.
                    losses.append(experiment_L)
                    gradients.append(experiment_G)
                # Plot the results.
                plt.figure()
                plt.subplot(211)
                plt.plot(losses)
                plt.title(
                    f"Losses for hidden layers: {hidden_layer}, hidden neurons: {hidden_neuron}, learning rate: {learning_rate}"
                )
                plt.xlabel("epochs")
                plt.ylabel("losses")
                plt.subplot(212)
                plt.plot(gradients)
                plt.title(
                    f"Gradients for hidden layers: {hidden_layer}, hidden neurons: {hidden_neuron}, learning rate: {learning_rate}"
                )
                plt.xlabel("epochs")
                plt.ylabel("gradients")
                plt.show()


def main() -> None:
    # Load the data.
    all_inputs, all_outputs = load_data()
    print(all_inputs[0])
    # Initialize the parameters.
    hidden_layers = [1]
    hidden_neurons = [3]
    learning_rates = [0.2]
    epochs = 1
    # Initialize the lists.
    experiments_L = []
    experiments_G = []
    # Iterate over the parameters.
    for hidden_layer in hidden_layers:
        for hidden_neuron in hidden_neurons:
            for learning_rate in learning_rates:
                # print combination
                print(
                    f"hidden layers: {hidden_layer}, hidden neurons: {hidden_neuron}, learning rate: {learning_rate}"
                )
                # Initialize the neural network.
                MLP = NeuralNetwork(
                    hidden_layers=hidden_layer, hidden_neurons=hidden_neuron
                )
                # Fit the neural network.
                losses, gradients = MLP.fit(
                    all_inputs, all_outputs, epochs=epochs, lr=learning_rate
                )
                # Store the results.
                experiments_L.append(losses)
                experiments_G.append(gradients)


if __name__ == "__main__":
    main()
