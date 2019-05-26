import numpy as np
import matplotlib.pyplot as plt

from NeuralNetwork import NeuralNetwork
import data_generator as DG
import task_info as TI



def MSE(actual, expected):
    return np.mean((actual - expected)**2)


def train_network(network, data, epochs):
    mse_losses = []
    for epoch in range(epochs):
        mse_sum = 0
        for row in logic_data:
            X, y = row[:-1], row[-1]
            network.train(X, y)
            mse_sum += MSE(network.predict(X), y)
        
        mse_losses.append(mse_sum / len(row))
        print(f"Epoch {epoch + 1}/{epochs}: MSE = {mse_losses[-1]}", end="\r")
    print()
    
    plt.plot(range(epochs), mse_losses)
    plt.xlabel("Epochs")
    plt.ylabel("Mean Squared Error")


def test_net_on_logic_func(net, func, parameters_count):
    max_value = 2**parameters_count - 1
    for value in range(max_value):
        X = []
        for i in range(parameters_count):
            X.append(value % 2)
            value = value >> 1

        actual = round(net.predict(X)[0], 2)
        expected = func(*X)

        print(f"{X}: {actual}/{expected}")



if __name__ == "__main__":
    epochs = 10
    logic_layers = [3, 2, 1]
    learning_rate = 1

    logic_data = np.array(DG.read_csv("logical.csv", int), np.int32)
    regr_raw_data = np.array(DG.read_csv("regression.csv", float), np.float)


    network = NeuralNetwork(logic_layers, learning_rate)
   
    train_network(network, logic_data, epochs)
    print("Results:")
    test_net_on_logic_func(network, TI.second_function, 3)


    plt.show()
