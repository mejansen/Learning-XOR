from PerceptronLayer import PerceptronLayer
import numpy as np


class MLP:

    def __init__(self, sizes):
        self.layers = [None]
        for i in range(len(sizes) - 1):
            layer = PerceptronLayer(sizes[i], sizes[i + 1])
            self.layers.append(layer)

    def forward_step(self, x):
        """Function for the forward pass. Input activation for each neuron is multiplied with the weights and given through activation function."""
        for layer in self.layers[1:]:
            x = layer.activate(x)
        return x

    def backprop_step(self, x, t, epsilon):

        # Forward step saving weights
        activations = [x]
        for layer in self.layers[1:]:
            x = activations[-1]
            y = layer.activate(x)
            activations.append(y)

        # Initial delta
        outputLayer = self.layers[-1]
        y = activations[-1]
        delta = (y - t) * (y * (1 - y))
        outputLayer.adapt(activations[-2], delta, epsilon)

        # Propagation error backwards
        num_layers = len(self.layers)
        for l in reversed(range(1, num_layers - 1)):
            # Calc delta
            weights_withoutBias = self.layers[l + 1].getWeightMatrix()[:, :-1]  # ignore Bias
            sigmoidPrime = activations[l] * (np.ones_like(activations[l]) - activations[l])
            delta = (delta.T @ weights_withoutBias) * sigmoidPrime

            self.layers[l].adapt(activations[l - 1], delta, epsilon)