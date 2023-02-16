"""Use a perceptron to classify the AND, OR and XOR logic gates"""
from NNpy.perceptron import Perceptron


def main():
    all_inputs = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
    all_outputs = [0.0, 1.0, 1.0, 1.0]
    # all_outputs = [0.0, 0.0, 0.0, 1.0]
    # all_outputs = [0.0, 1.0, 1.0, 0.0]
    P = Perceptron()
    weights, losses = P.fit(all_inputs, all_outputs)
    for ipt in all_inputs:
        print(f"Input: {ipt}")
        y = P.forward(ipt, weights)
        print(f"Output {y}")


if __name__ == "__main__":
    main()
