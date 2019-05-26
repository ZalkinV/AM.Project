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


    def predict(self, input):
        hidden_in = np.dot(self.weights_input_hidden, input)
        hidden_out = np.array(list(map(sigmoid, hidden_in)))

        output_in = np.dot(self.weights_hidden_output, hidden_out)
        output_out = np.array(list(map(sigmoid, output_in)))

        return output_out
