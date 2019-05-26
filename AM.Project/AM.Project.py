import numpy as np

from NeuralNetwork import NeuralNetwork
import data_generator as DG



def MSE(actual, expected):
    return np.mean((actual - expected)**2)


if __name__ == "__main__":
    logic_data = np.array(DG.read_csv("logical.csv", int), np.int32)
    regr_raw_data = np.array(DG.read_csv("regression.csv", float), np.float)


    network = NeuralNetwork(3, 2, 1)
    result = network.predict([0,1,1])
    train = network.train(np.array([0,1,1]), 1)
