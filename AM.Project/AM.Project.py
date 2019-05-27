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
        for row in data:
            X, y = row[:-1], row[-1]
            network.train(X, y)
            mse_sum += MSE(network.predict(X), y)
        
        mse_losses.append(mse_sum / len(row))
        print(f"Epoch {epoch + 1}/{epochs}: MSE = {mse_losses[-1]}", end="\r")
    print()
    
    plt.plot(range(epochs), mse_losses)
    plt.xlabel("Epochs")
    plt.ylabel("Mean Squared Error")
    plt.show()


def test_net_on_logic_func(net, func, parameters_count):
    max_value = 2**parameters_count
    for value in range(max_value):
        X = []
        for i in range(parameters_count):
            X.append(value % 2)
            value = value >> 1

        actual = round(net.predict(X)[0], 2)
        expected = func(*X)

        print(f"{X}: {actual}/{expected}")



def network_work(epochs, layers, data, learning_rate, func, parameters_count):
    network = NeuralNetwork(layers, learning_rate)
   
    train_network(network, data, epochs)
    print("Results:")
    test_net_on_logic_func(network, func, parameters_count)
    print()


if __name__ == "__main__":
    epochs = 50
    logic_layers = [3, 2, 1]
    logic_layers_2 = [2, 1, 1]
    learning_rate = 1

    logic_data = np.array(DG.read_csv("logical.csv", int), np.int32)
    logic_data_2 = np.array(DG.read_csv("logical_2.csv", int), np.int32)


    network_work(epochs, logic_layers, logic_data, learning_rate, TI.second_function, 3)
    network_work(epochs, logic_layers_2, logic_data_2, learning_rate, TI.third_function, 2)

