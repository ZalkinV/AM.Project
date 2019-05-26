import numpy as np

from NeuralNetwork import NeuralNetwork
import data_generator as DG



def MSE(actual, expected):
    return np.mean((actual - expected)**2)


if __name__ == "__main__":
    logic_raw_data = DG.read_csv("logical.csv")
    logic_data = []
    for row in logic_raw_data:
        logic_data.append(list(map(int, row)))
    logic_data = np.array(logic_data)

    regr_raw_data = DG.read_csv("regression.csv")
    regr_data = []
    for row in regr_raw_data:
        regr_data.append(list(map(float, row)))
    regr_data = np.array(regr_data)


    network = NeuralNetwork(3, 2, 1)
    result = network.predict([0,1,1])
    train = network.train(np.array([0,1,1]), 1)
