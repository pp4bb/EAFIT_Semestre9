import NNpy.nn as nn
import NNpy.functional as F
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt


class design1:
    def __init__(self):
        self.fcn = nn.Sequential(
            nn.Linear(2, 2, args=(1, 0)),
            nn.Linear(2, 1, args=(1, 0)),
            nn.Linear(1, 1, last=True, args=(1, 0)),
        )

    def forward(self, x):
        return self.fcn.forward(x)

    def fit(self, x, y, epochs, lr):
        meanlosses = []
        meangradients = []
        for epoch in tqdm(range(epochs), desc="Epochs"):
            losses = []
            grads = []
            for i in range(len(x)):
                y_hat = self.fcn.forward(x[i])
                loss = F.insta_energy(y[i], y_hat)
                losses.append(loss)
                self.fcn.backward(loss=loss)
                grad = self.fcn.layers[-1].gradient
                grads.append(grad)
                self.fcn.step(lr)
            meanloss = sum(losses) / len(losses)
            meanlosses.append(meanloss)
            meangrad = sum(grads) / len(grads)
            meangradients.append(meangrad)

        return meanlosses, meangradients


def main():
    all_inputs = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    all_outputs = np.array([0.0, 1.0, 1.0, 1.0])
    # all_outputs = [0.0, 0.0, 0.0, 1.0]
    # all_outputs = [0.0, 1.0, 1.0, 0.0]
    MLP = design1()
    losses, gradients = MLP.fit(all_inputs, all_outputs, 10, 0.01)
    for ipt in all_inputs:
        print(f"Input: {ipt}")
        y = MLP.forward(ipt)
        print(f"Output {y}")

    print(MLP.fcn.layers[-1].weights)


if __name__ == "__main__":
    main()
