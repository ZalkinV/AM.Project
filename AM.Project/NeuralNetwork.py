import math
import numpy as np



def sigmoid(x):
    return (1 / (1 + math.e**(-x)))


def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))


class NeuralNetwork:
    def __init__(self, input_count, neurons_count, learning_rate):
        self.learning_rate = np.array(learning_rate)
        self.weights_input_hidden = np.random.randn(neurons_count, input_count)
        self.weights_hidden_output = np.random.randn(1, neurons_count)
