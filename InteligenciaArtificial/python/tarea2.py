"""Use a fully conected multilayer perceptron in its most basic way to classify the
AND, OR, and XOR logic gates.
"""
from multilayer_perceptron import MultilayerPerceptron
import matplotlib

#matplotlib.use("TKAgg")
import matplotlib.pyplot as plt


def main():
    all_inputs = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
    all_outputs = [0.0, 1.0, 1.0, 1.0]
    # all_outputs = [0.0, 0.0, 0.0, 1.0]
    # all_outputs = [0.0, 1.0, 1.0, 0.0]
    # Multilayer perceptron
    MLP = MultilayerPerceptron(phiF="linear")
    weights, losses, deltas = MLP.fit(all_inputs, all_outputs)
    print("Weights: ", weights)
    for ipt in all_inputs:
        print(f"Input: {ipt}")
        y = MLP.forward(ipt, weights)
        print(f"Output {y}")

    # Plot

    plt.plot(deltas)
    plt.xlabel("epochs")
    plt.ylabel("deltas")
    plt.show()


if __name__ == "__main__":
    main()
