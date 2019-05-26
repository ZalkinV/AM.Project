import numpy as np

from NeuralNetwork import NeuralNetwork
import data_generator as DG



def MSE(actual, expected):
    return np.mean((actual - expected)**2)


if __name__ == "__main__":
    epochs = 1000

    inputs_count = 3
    neurons_count = 2
    learning_rate = 1

    logic_data = np.array(DG.read_csv("logical.csv", int), np.int32)
    regr_raw_data = np.array(DG.read_csv("regression.csv", float), np.float)


    network = NeuralNetwork(inputs_count, neurons_count, learning_rate)
    result = network.predict([0,1,1])
    train = network.train(np.array([0,1,1]), 1)
