import numpy as np

from NeuralNetwork import NeuralNetwork




def MSE(actual, expected):
    return np.mean((actual - expected)**2)


if __name__ == "__main__":
    network = NeuralNetwork(3, 2, 1)
    result = network.predict([0,1,1])
    train = network.train(np.array([0,1,1]), 1)
