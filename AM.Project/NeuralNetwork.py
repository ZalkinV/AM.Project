import math
import numpy as np



def sigmoid(x):
    return (1 / (1 + math.e**(-x)))


def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))


class NeuralNetwork:
    def __init__(self, layers, learning_rate):
        np.random.seed(0)
        self.learning_rate = np.array(learning_rate)
        self.activation = sigmoid
        self.gradient = np.vectorize(lambda x_sigmoided: x_sigmoided * (1 - x_sigmoided))

        self.weights = []
        for i_layer in range(1, len(layers)):
            current_layer_weights = np.random.randn(layers[i_layer], layers[i_layer - 1])
            self.weights.append(current_layer_weights)


    def get_layers(self, input):
        layers = [input]
        for layer_weights in self.weights:
            current_layer = []
            for neuron_weights in layer_weights:
                neuron_value = self.activation(np.sum(neuron_weights * layers[-1]))
                current_layer.append(neuron_value)
            layers.append(current_layer)

        return layers

    
    def predict(self, input):
        layers_values = self.get_layers(input)
        return layers_values[-1]


    def train(self, input, expected_output):
        layers = self.get_layers(input)
        layers = [np.array(layer) for layer in layers]
        

        error_output = layers[-1] - expected_output
        weights_hidden_output_delta = error_output * self.gradient(layers[2])
        self.weights[1] -= np.dot(weights_hidden_output_delta, layers[1].reshape(1, len(layers[1]))) * self.learning_rate
        
        error_hidden = self.weights[1] * weights_hidden_output_delta
        weights_input_hidden_delta = error_hidden * self.gradient(layers[1])
        self.weights[0] -= np.dot(input.reshape(len(layers[0]), 1), weights_input_hidden_delta).T * self.learning_rate
