import matplotlib.pyplot as plt
from nn_design import NeuralNetwork
import numpy as np
from tqdm import tqdm
from NNpy import functional as F
from sklearn import preprocessing
import warnings

warnings.filterwarnings("ignore")

# Global variables
hidden_layers = [1, 2, 3]
hidden_neurons = [1, 2, 3, 4, 5]
learning_rates = [0.2, 0.5, 0.9]
epochs = 50
tolerance = 1e-2


def normalize_data(data: np.ndarray, method: str) -> np.ndarray:
    """Normalize the data.

    Parameters
    ----------
    data: np.ndarray
        Numpy array with the data.
    method: str
        Method to normalize the data.
    """
    if method == "minmax":
        return preprocessing.minmax_scale(data)
    elif method == "standard":
        return preprocessing.scale(data)


def load_data() -> tuple:
    """Load the data."""
    data = np.loadtxt("data/examen.csv", comments="#", delimiter=",", unpack=False)
    # Normalize the data by column.
    data = normalize_data(data, "minmax")
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


def iterate_over_parameters(train: tuple, phi: str) -> None:
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
                    in_features=train[0][0].shape[0],
                    hidden_layers=hidden_layer,
                    hidden_neurons=hidden_neuron,
                    phi=phi,
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


def plot_losses(experiments_L: list, phi: str) -> None:
    """Plot the losses.

    Parameters
    ----------
    experiments_L: list
        List with the losses of the experiments.
    phi: str
        Activation function.
    """
    fig, ax = plt.subplots(3, 1, figsize=(30, 20))
    aux = 0
    for hidden_layer in hidden_layers:
        for hidden_neuron in hidden_neurons:
            for learning_rate in range(len(learning_rates)):
                ax[learning_rate].plot(
                    experiments_L[aux], label=f"L: {hidden_layer}, N: {hidden_neuron}"
                )
                aux += 1
    ax[0].set_title("Learning rate: 0.2")
    ax[0].set_xlabel("Epochs")
    ax[0].set_ylabel("Loss")
    ax[1].set_title("Learning rate: 0.5")
    ax[1].set_xlabel("Epochs")
    ax[1].set_ylabel("Loss")
    ax[2].set_title("Learning rate: 0.9")
    ax[2].set_xlabel("Epochs")
    ax[2].set_ylabel("Loss")
    ax[0].legend(loc="upper right")
    ax[1].legend(loc="upper right")
    ax[2].legend(loc="upper right")
    plt.savefig(f"losses_{phi}.png")


def plot_gradients(experiments_G: list, phi: str) -> None:
    """Plot the gradients.

    Parameters
    ----------
    experiments_G: list
        List with the gradients of the experiments.
    phi: str
        Activation function.
    """
    fig, ax = plt.subplots(3, 1, figsize=(30, 20))
    aux = 0
    for hidden_layer in hidden_layers:
        for hidden_neuron in hidden_neurons:
            for learning_rate in range(len(learning_rates)):
                ax[learning_rate].plot(
                    experiments_G[aux], label=f"L: {hidden_layer}, N: {hidden_neuron}"
                )
                aux += 1
    ax[0].set_title("Learning rate: 0.2")
    ax[0].set_xlabel("Epochs")
    ax[0].set_ylabel("Gradient")
    ax[1].set_title("Learning rate: 0.5")
    ax[1].set_xlabel("Epochs")
    ax[1].set_ylabel("Gradient")
    ax[2].set_title("Learning rate: 0.9")
    ax[2].set_xlabel("Epochs")
    ax[2].set_ylabel("Gradient")
    ax[0].legend(loc="upper right")
    ax[1].legend(loc="upper right")
    ax[2].legend(loc="upper right")
    plt.savefig(f"gradients_{phi}.png")


def plot_best_worst_median(best, worst, median, test, name: str):
    """Plot the best, worst and median experiments.

    Parameters
    ----------
    best: tuple
        Tuple with the best model and its parameters.
    worst: tuple
        Tuple with the worst model and its parameters.
    median: tuple
        Tuple with the median model and its parameters.
    test: tuple
        Tuple with the test data.
    name: str
        Name of the experiment.
    """
    plt.figure()
    b = []
    w = []
    m = []
    r = []
    for i in range(len(test[0])):
        b.append(best[0].forward(test[0][i]))
        w.append(worst[0].forward(test[0][i]))
        m.append(median[0].forward(test[0][i]))
        r.append(test[1][i])
    plt.plot(b, label="Best")
    plt.plot(w, label="Worst")
    plt.plot(m, label="Median")
    plt.plot(r, label="Real")
    plt.legend()
    plt.savefig(f"best_worst_median_{name}.png")


def plot_comparison(best_S, best_L, test, name: str):
    """Plot the best experiments with sigmoid and linear activation functions.

    Parameters
    ----------
    best_S: tuple
        Tuple with the best model and its parameters with sigmoid activation function.
    best_L: tuple
        Tuple with the best model and its parameters with linear activation function.
    test: tuple
        Tuple with the test data.
    name: str
        Name of the experiment.
    """
    plt.figure()
    b_S = []
    b_L = []
    r = []
    for i in range(len(test[0])):
        b_S.append(best_S[0].forward(test[0][i]))
        b_L.append(best_L[0].forward(test[0][i]))
        r.append(test[1][i])
    plt.plot(b_S, label="Sigmoid")
    plt.plot(b_L, label="Linear")
    plt.plot(r, label="Real")
    plt.legend()
    plt.savefig(f"best_comparison_{name}.png")


def main() -> None:
    # Load the data.
    train, test, validation = load_data()
    # Perform the experiments for the sigmoid activation function.
    experiments_LS, experiments_GS, models_S = iterate_over_parameters(
        train, phi="sigmoid"
    )
    # Perform the experiments for the linear activation function.
    experiments_LL, experiments_GL, models_L = iterate_over_parameters(
        train, phi="linear"
    )
    # Get the best, worst and median experiments for the sigmoid activation function.
    best_LS, worst_LS, median_LS = get_best_worst_median(models_S, validation)
    # print the parameters of the best, worst and median experiments.
    print("Sigmoid activation function:")
    print("Best experiment:")
    print(best_LS[1])
    print("Worst experiment:")
    print(worst_LS[1])
    print("Median experiment:")
    print(median_LS[1])
    # Plot the best, worst and median experiments for the sigmoid activation function.
    plot_best_worst_median(best_LS, worst_LS, median_LS, validation, "validationS")
    plot_best_worst_median(best_LS, worst_LS, median_LS, test, "testS")
    # Plot the losses for the sigmoid activation function.
    plot_losses(experiments_LS, "sigmoid")
    # Plot the gradients for the sigmoid activation function.
    plot_gradients(experiments_GS, "sigmoid")
    # Get the best, worst and median experiments for the linear activation function.
    best_LL, worst_LL, median_LL = get_best_worst_median(models_L, validation)
    # print the parameters of the best, worst and median experiments.
    print("Linear activation function:")
    print("Best experiment:")
    print(best_LL[1])
    print("Worst experiment:")
    print(worst_LL[1])
    print("Median experiment:")
    print(median_LL[1])
    # Plot the best, worst and median experiments for the linear activation function.
    plot_best_worst_median(best_LL, worst_LL, median_LL, validation, "validationL")
    plot_best_worst_median(best_LL, worst_LL, median_LL, test, "testL")
    # Plot the losses for the linear activation function.
    plot_losses(experiments_LL, "linear")
    # Plot the gradients for the linear activation function.
    plot_gradients(experiments_GL, "linear")
    # Plot the best experiments with sigmoid and linear activation functions.
    plot_comparison(best_LS, best_LL, test, "Best")
    # Plot the worst experiments with sigmoid and linear activation functions.
    plot_comparison(worst_LS, worst_LL, test, "Worst")
    # Plot the median experiments with sigmoid and linear activation functions.
    plot_comparison(median_LS, median_LL, test, "Median")


if __name__ == "__main__":
    main()
