{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "def perceptron(inputs:np.ndarray, weights:np.ndarray, phi:function):\n",
    "    localfield = np.dot(weights, inputs)\n",
    "    output = phi(localfield)\n",
    "    return output"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "def loss(output: np.ndarray, label: np.ndarray):\n",
    "    (label - output)**2 / 2\n",
    "\n",
    "def diff_loss(output: np.ndarray, label: np.ndarray):\n",
    "    return -(label - output)\n",
    "\n",
    "def phi(localfield:float):\n",
    "    return 1 / (1 + np.exp(-localfield))\n",
    "\n",
    "def diff_phi(phi:function, localfield:float):\n",
    "    return phi(localfield) * (1 - phi(localfield))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "def gradient_descend(\n",
    "    inputs:np.ndarray,\n",
    "    output:np.ndarray,\n",
    "    localfield:float,\n",
    "    label:np.ndarray,\n",
    "    diff_loss:function,\n",
    "    diff_phi:function\n",
    "    ):\n",
    "    return [\n",
    "        diff_loss(label, output) * diff_phi(localfield) * ipt for ipt in inputs\n",
    "        ]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "def multi_layer_perceptron(input, ):\n",
    "    pass\n"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.10.2"
  },
  "orig_nbformat": 4,
  "vscode": {
   "interpreter": {
    "hash": "0ed17916cfe6f9c87218e6dae1b9cc0f032f6e36f680b2e47ca8c8872f167137"
   }
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
