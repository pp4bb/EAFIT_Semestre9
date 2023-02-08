import numpy as np
import matplotlib.pyplot as plt
from perceptron import Perceptron


def fit(
    perceptron: Perceptron,
    all_inputs: list,
    all_outputs: list,
    epochs: int = 10000,
    lr: float = 0.01,
):
    weights = np.random.rand(2)
    meanlosses = []
    for _ in range(epochs):
        losses = []
        gradients = []
        for data, target in zip(all_inputs, all_outputs):
            y = perceptron.forward(data, weights)
            loss = perceptron.loss(target, y)
            gradient = perceptron.gradient(target, y)
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
        weights += delta

    print(weights)
    return weights, meanloss


def main():
    all_inputs = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
    all_outputs = [0.0, 1.0, 1.0, 1.0]
    P = Perceptron()
    weights, losses = fit(P, all_inputs, all_outputs)
    for ipt in all_inputs:
        print(f"Input: {ipt}")
        y = P.forward(ipt, weights)
        print(f"Output {y}")

    # Plot the resulting lines
    fig, ax = plt.subplot(1, 4, 4)
    for i in range(3):
        ax[i]


if __name__ == "__main__":
    main()
