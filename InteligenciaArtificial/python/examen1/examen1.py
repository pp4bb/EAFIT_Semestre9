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

plt.figure()
plt.subplot(211)
plt.plot(deltas)
plt.title("hiddden layer activation function: sigmoid")
plt.xlabel("epochs")
plt.ylabel("gradients")

plt.subplot(212)
plt.plot(deltas2)
plt.title("hiddden layer activation function: x + 0.5")
plt.xlabel("epochs")
plt.ylabel("gradients")
plt.show()
