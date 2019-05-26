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
    
    for epoch in range(epochs):
        mse_sum = 0

        for row in logic_data:
            X, y = row[:-1], row[-1]
            network.train(X, y)
            mse_sum += MSE(network.predict(X), y)
        
        mse_loss = mse_sum / len(row)
        print(f"Epoch {epoch + 1}/{epochs}: MSE = {mse_loss}", end="\r")
