#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import argparse

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Initialize weights and biases
        self.W1 = np.random.randn(self.input_size, self.hidden_size)
        self.b1 = np.zeros((1, self.hidden_size))
        self.W2 = np.random.randn(self.hidden_size, self.output_size)
        self.b2 = np.zeros((1, self.output_size))
    
    def tanh_derivative(self, z):
        return 1 - np.tanh(z) ** 2
    
    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = np.tanh(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.z2  # Linear activation for output layer
        
        return self.a2
    
    def backward(self, X, y, output, learning_rate):
        self.output_error = output - y
        self.output_delta = self.output_error
        
        self.z1_error = self.output_delta.dot(self.W2.T)
        self.z1_delta = self.z1_error * self.tanh_derivative(self.a1)
        
        self.W2 -= self.a1.T.dot(self.output_delta) * learning_rate
        self.b2 -= np.sum(self.output_delta, axis=0, keepdims=True) * learning_rate
        self.W1 -= X.T.dot(self.z1_delta) * learning_rate
        self.b1 -= np.sum(self.z1_delta, axis=0, keepdims=True) * learning_rate
    
    def train_batch(self, X, y, epochs, learning_rate):
        for _ in range(epochs):
            output = self.forward(X)
            self.backward(X, y, output, learning_rate)
    
    def train_online(self, X, y, epochs, learning_rate):
        for _ in range(epochs):
            for i in range(len(X)):
                output = self.forward(X[i].reshape(1, -1))
                self.backward(X[i].reshape(1, -1), y[i].reshape(1, -1), output, learning_rate)
    
    def predict(self, X):
        return self.forward(X)

def load_data(filename, split_ratio=0.8):
    data = np.loadtxt(filename)
    X = data[:, 0].reshape(-1, 1)
    y = data[:, 1].reshape(-1, 1)
    
    split_index = int(split_ratio * len(X))
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]
    
    return X_train, X_test, y_train, y_test

def batch_train(nn):
    nn.train_batch(X_train, y_train, epochs, learning_rate)

    train_predictions = nn.predict(X_train)
    test_predictions = nn.predict(X_test)

    train_mse = np.mean((train_predictions - y_train) ** 2)
    test_mse = np.mean((test_predictions - y_test) ** 2)

    print(f'Training MSE: {train_mse}')
    print(f'Test MSE: {test_mse}')

    return train_predictions, test_predictions

def online_train(nn: NeuralNetwork):
    nn.train_online(X_train, y_train, epochs, learning_rate)

    train_predictions_online = nn.predict(X_train)
    test_predictions_online = nn.predict(X_test)

    train_mse_online = np.mean((train_predictions_online - y_train) ** 2)
    test_mse_online = np.mean((test_predictions_online - y_test) ** 2)

    print(f'Training MSE (Online): {train_mse_online}')
    print(f'Test MSE (Online): {test_mse_online}')

    return train_predictions_online, test_predictions_online

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('data_num', type=str, help='The number of the dataset')
    args = parser.parse_args()

    input_size = 1
    hidden_size = 10
    output_size = 1
    epochs = 1000
    learning_rate = 0.01

    X_train, X_test, y_train, y_test = load_data(f'Dane/dane{args.data_num}.txt')

    nn = NeuralNetwork(input_size, hidden_size, output_size)

    train_predictions, test_predictions               = batch_train(nn)
    train_predictions_online, test_predictions_online = online_train(nn)

    plt.figure()
    plt.plot(X_train, y_train, 'ro', label='Training data')
    plt.plot(X_train, train_predictions, 'b-', label='Batch predictions')
    plt.plot(X_train, train_predictions_online, 'g--', label='Online predictions')
    plt.legend()
    plt.show()
    
    plt.figure()
    plt.plot(X_test, y_test, 'ro', label='Test data')
    plt.plot(X_test, test_predictions, 'b-', label='Batch predictions')
    plt.plot(X_test, test_predictions_online, 'g--', label='Online predictions')
    plt.legend()
    plt.show()

