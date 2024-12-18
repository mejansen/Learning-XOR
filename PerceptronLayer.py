from Perceptron import Perceptron
import numpy as np

class PerceptronLayer:
    def __init__(self, dim_input, dim_out):
        self.perceptrons = [Perceptron(dim_input) for _ in range(dim_out)]

    def activate(self, x):
        """Functino calculates the activation for an entire layer referencing back to the activation func in Perceptron.py"""
        activation = np.array([p.activate(x) for p in self.perceptrons])
        return activation

    def adapt(self, x, deltas, epsilon):
        """This function can be seen as the middleware function between the backprop algorithm and the perceptron."""
        for perceptron, delta in zip(self.perceptrons, deltas):
            perceptron.adapt(x, delta, epsilon)

    def getWeightMatrix(self):
        """Weight matrices are always important, so let's get them here."""
        return np.array([p.weights for p in self.perceptrons])