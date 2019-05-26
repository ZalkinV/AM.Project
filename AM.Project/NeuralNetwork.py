import math
import numpy as np



def sigmoid(x):
    return (1 / (1 + math.e**(-x)))


def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))


class NeuralNetwork:
    def __init__(self, input_count, neurons_count, learning_rate):
        np.random.seed(0)
        self.learning_rate = np.array(learning_rate)
        self.weights_input_hidden = np.random.randn(neurons_count, input_count)
        self.weights_hidden_output = np.random.randn(1, neurons_count)
        self.activation = np.vectorize(sigmoid)
        self.gradient = np.vectorize(lambda x_sigmoided: x_sigmoided * (1 - x_sigmoided))


    def predict(self, input):
        hidden_in = np.dot(self.weights_input_hidden, input)
        hidden_out = self.activation(hidden_in)

        output_in = np.dot(self.weights_hidden_output, hidden_out)
        output_out = self.activation(output_in)

        return output_out


    def train(self, input, expected_output):
        hidden_in = np.dot(self.weights_input_hidden, input)
        hidden_out = self.activation(hidden_in)

        output_in = np.dot(self.weights_hidden_output, hidden_out)
        output_out = self.activation(output_in)


        error_output = output_out - expected_output
        weights_hidden_output_delta = error_output * self.gradient(output_out)
        self.weights_hidden_output -= np.dot(weights_hidden_output_delta, hidden_out.reshape(1, len(hidden_out))) * self.learning_rate
        
        error_hidden = self.weights_hidden_output * weights_hidden_output_delta
        weights_input_hidden_delta = error_hidden * self.gradient(hidden_out)
        self.weights_input_hidden -= np.dot(input.reshape(len(input), 1), weights_input_hidden_delta).T * self.learning_rate
