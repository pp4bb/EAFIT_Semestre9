from numpy import loadtxt
import matplotlib.pyplot as plt
from multilayer_perceptron import MultilayerPerceptron

data = loadtxt("DATOS.txt", comments="#", delimiter=",", unpack=False)
all_inputs = []
all_outputs = []

for i in range(300):
    all_inputs.append([data[i, 1], data[i, 2]])
    all_outputs.append(data[i, 0])


epochs = 1000
MLP = MultilayerPerceptron(phiF="sigmoid", hiddenF="sigmoid")
weights, losses, deltas = MLP.fit(all_inputs, all_outputs, epochs=epochs)

MLP2 = MultilayerPerceptron(phiF="sigmoid", hiddenF="linear")
weights2, losses2, deltas2 = MLP.fit(all_inputs, all_outputs, epochs=epochs)


plt.plot(deltas, label="sigmoid")
plt.xlabel("epochs")
plt.ylabel("gradients")

plt.plot(deltas2, label="linear")

plt.show()
