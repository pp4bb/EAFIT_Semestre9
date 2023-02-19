import matplotlib.pyplot as plt
from nn_design import NeuralNetwork
import numpy as np
from tqdm import tqdm
from NNpy import functional as F

# Global variables
hidden_layers = [1, 2, 3]
hidden_neurons = [1, 2, 3, 4, 5]
learning_rates = [0.2, 0.5, 0.9]
epochs = 10
tolerance = 1e-2


def load_data() -> tuple:
    """Load the data."""
    data = np.loadtxt("data/DATOS.txt", comments="#", delimiter=",", unpack=False)
    all_inputs = []
    all_outputs = []
    for i in range(len(data)):
        all_inputs.append(data[i][0:2])
        all_outputs.append(data[i][2])
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


def iterate_over_parameters(train: tuple) -> None:
    """Iterate over the parameters.

    Parameters
    ----------
    train: tuple
        Tuple with the training data.
    """
    # Initialize the lists.
    experiments_L = []
    experiments_G = []
    models_dict = {"models": [], "parameters": []}
    # Iterate over the parameters.
    for hidden_layer in tqdm(hidden_layers, desc="Hidden layers", leave=False):
        for hidden_neuron in tqdm(hidden_neurons, desc="Hidden neurons", leave=False):
            for learning_rate in tqdm(
                learning_rates, desc="Learning rates", leave=False
            ):
                # Initialize the neural network.
                MLP = NeuralNetwork(
                    hidden_layers=hidden_layer, hidden_neurons=hidden_neuron
                )
                # Fit the neural network.
                losses, gradients = MLP.fit(
                    train[0],
                    train[1],
                    epochs=epochs,
                    lr=learning_rate,
                    tolerance=tolerance,
                )
                # Store the model.
                models_dict["models"].append(MLP)
                models_dict["parameters"].append(
                    (hidden_layer, hidden_neuron, learning_rate)
                )
                # Store the training results.
                experiments_L.append(losses)
                experiments_G.append(gradients)

    return experiments_L, experiments_G, models_dict


def get_best_worst_median(models: dict, test: tuple) -> tuple:
    """Get the best, worst and median experiments.

    Parameters
    ----------
    models: dict
        Dictionary with the models and their parameters.
    test: tuple
        Tuple with the test data.
    """
    # Initialize the lists.
    instant_energy = []
    # Forwards pass.
    for model in models["models"]:
        errors = []
        for i in range(len(test[0])):
            y = model.forward(test[0][i])
            error = F.mse(y, test[1][i])
            errors.append(error)
        instant_energy.append(np.mean(errors))
    # Get the best, worst and median experiments.
    best_idx = np.argmin(instant_energy)
    best_L = models["models"][best_idx], models["parameters"][best_idx]
    worst_idx = np.argmax(instant_energy)
    worst_L = models["models"][worst_idx], models["parameters"][worst_idx]
    median_idx = np.argsort(instant_energy)[len(instant_energy) // 2]
    median_L = models["models"][median_idx], models["parameters"][median_idx]

    return best_L, worst_L, median_L


def plot_gradients(experiments_G: list) -> None:
    """Plot the gradients.

    Parameters
    ----------
    experiments_G: list
        List with the gradients of the experiments.
    """
    for lr in learning_rates:
        fig, ax = plt.subplots(
            len(hidden_layers), len(hidden_neurons), figsize=(30, 20)
        )
        # print dims of ax
        aux = 0
        for i in range(len(hidden_layers)):
            for j in range(len(hidden_neurons)):
                for layer in range(len(experiments_G[aux])):
                    ax[i, j].plot(experiments_G[aux][layer], label=f"Layer {layer}")
                aux += 1
                ax[i, j].set_title(
                    f"Hidden layers: {hidden_layers[i]}, hidden neurons: {hidden_neurons[j]}, learning rate: {lr}"
                )
                ax[i, j].set_xlabel("Epochs")
                ax[i, j].set_ylabel("Gradient")
                ax[i, j].legend()
        plt.savefig(f"Gradients_lr_{lr}.png")


def main() -> None:
    # Load the data.
    train, test, validation = load_data()
    # Perform the experiments.
    experiments_L, experiments_G, models = iterate_over_parameters(train)
    # Get the best, worst and median experiments.
    # best_L, worst_L, median_L = get_best_worst_median(models, test)
    # best_G, worst_G, median_G = get_best_worst_median(experiments_G)
    plot_gradients(experiments_G)


if __name__ == "__main__":
    main()
