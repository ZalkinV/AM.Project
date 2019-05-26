import math



def sigmoid(x):
    return (1 / (1 + math.e**(-x)))


def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))
