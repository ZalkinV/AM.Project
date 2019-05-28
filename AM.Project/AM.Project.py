import numpy as np
import matplotlib.pyplot as plt

from NeuralNetwork import NeuralNetwork
import data_generator as DG
import task_info as TI



def MSE(actual, expected):
    return np.mean((actual - expected)**2)


def train_network(network, data, epochs, is_plot):
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
    
    if is_plot:
        plt.plot(range(1, epochs + 1), mse_losses)
        plt.xlabel("Epochs")
        plt.ylabel("Mean Squared Error")
    return mse_losses[-1]
    


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



def network_work(epochs, layers, data, learning_rate, func, parameters_count, is_plot=True):
    network = NeuralNetwork(layers, learning_rate)
   
    mse_loss = train_network(network, data, epochs, is_plot)
    print("Results:")
    test_net_on_logic_func(network, func, parameters_count)
    print()
    
    return mse_loss


if __name__ == "__main__":
    epochs = 100
    logic_layers = [3, 2, 1]
    logic_layers_2 = [2, 2, 1]
    learning_rate = 1

    logic_data = np.array(DG.read_csv("logical.csv", int), np.int32)
    logic_data_2 = np.array(DG.read_csv("logical_2.csv", int), np.int32)


    network_work(epochs, logic_layers, logic_data, learning_rate, TI.second_function, 3)
    network_work(epochs, logic_layers_2, logic_data_2, learning_rate, TI.third_function, 2)
    plt.legend(["First function", "Second functon"])
    plt.show()


    
    plt.xticks(range(1, 20))
    plt.xlabel("Neurons count")
    plt.ylabel("Mean Squared Error")

    print("\n\nNeurons count test for first function:")
    epochs = 50
    mse_losses = []
    max_neuron = 17
    for i in range(1, max_neuron):
        logic_layers[1] = i
        print(f"Hidden layer neurons count = {i}")
        mse_losses.append(round(network_work(epochs, logic_layers, logic_data, learning_rate, TI.second_function, 3, False), 4))
    plt.plot(range(1, max_neuron), mse_losses)

    print("\nNeurons count test for second function:")
    mse_losses.clear()
    for i in range(1, max_neuron):
        logic_layers[0] = 2
        logic_layers[1] = i
        print(f"Hidden layer neurons count = {i}")
        mse_losses.append(round(network_work(epochs, logic_layers, logic_data_2, learning_rate, TI.third_function, 2, False), 4))
    plt.plot(range(1, max_neuron), mse_losses)

    plt.legend(["First function", "Second functon"])
    plt.show()
