import numpy as np
from Useful import *


class Perceptron:
    def __init__(self, dim_input):
        self.weights = np.random.normal(size = dim_input + 1)

    def activate(self, x):
        """Function returns the activation of a single unit on the basis of its input x."""
        x = np.append(x, 1)  # add the bias
        self.drive = self.weights @ x # multiply to get the drive
        return sigmoid(self.drive) # return the activation

    def adapt(self, x, delta, epsilon):
        """Function implements the adaptation after backpropagation values were given."""
        x = np.append(x, 1)  # bias
        self.weights -= epsilon * delta * x