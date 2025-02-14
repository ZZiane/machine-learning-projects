import numpy as np


class NeuralNetwork:

    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.weights1 = np.random.randn(self.input_size, self.hidden_size)
        self.bias1 = np.zeros((1, self.hidden_size))
        self.weights2 = np.random.randn(self.hidden_size, self.output_size)
        self.bias2 = np.zeros((1, self.output_size))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def forward(self, X):
        self.hidden_layer_activation = np.dot(X, self.weights1) + self.bias1
        self.hidden_layer_output = self.sigmoid(self.hidden_layer_activation)
        self.output_layer_activation = np.dot(
            self.hidden_layer_output, self.weights2) + self.bias2
        self.predicted_output = self.sigmoid(self.output_layer_activation)
        return self.predicted_output

    def backward(self, X, y, output):
        self.error = y - output
        self.output_delta = self.error * self.sigmoid_derivative(output)
        self.hidden_error = self.output_delta.dot(self.weights2.T)
        self.hidden_delta = self.hidden_error * \
            self.sigmoid_derivative(self.hidden_layer_output)
        self.weights1 += X.T.dot(self.hidden_delta)
        self.bias1 += np.sum(self.hidden_delta, axis=0, keepdims=True)
        self.weights2 += self.hidden_layer_output.T.dot(self.output_delta)
        self.bias2 += np.sum(self.output_delta, axis=0, keepdims=True)

    def train(self, X, y, epochs):
        for epoch in range(epochs):
            output = self.forward(X)
            self.backward(X, y, output)

    def predict(self, X):
        return self.forward(X)
