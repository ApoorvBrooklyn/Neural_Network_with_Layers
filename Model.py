import time
import random
import numpy as np
import matplotlib.pyplot as plt
# from tensorflow.keras.datasets import mnist
# from keras.utils import to_categorical



X_train = np.loadtxt('dataset/cat_train_x.csv', delimiter = ',')/255.0
Y_train = np.loadtxt('dataset/cat_train_y.csv', delimiter = ',').reshape(1, X_train.shape[1])
X_test = np.loadtxt('dataset/cat_test_x.csv', delimiter = ',')/255.0
Y_test = np.loadtxt('dataset/cat_test_y.csv', delimiter = ',').reshape(1, X_test.shape[1])


print(X_train.shape)
print(Y_train.shape)
print(X_test.shape)
print(Y_test.shape)

def sigmoid(Z):
    A = 1/(1+np.exp(-Z))
    return A

def softmax(z):
    expZ = np.exp(z)
    return expZ/(np.sum(expZ, 0))

def relu(Z):
    A = np.maximum(0,Z)
    return A

def tanh(x):
    return np.tanh(x)

def derivative_relu(Z):
    return np.array(Z > 0, dtype = 'float')

def derivative_tanh(x):
    return (1 - np.power(x, 2))

layer_dims = [X.shape[1], 100, 200, Y.shape[0]]

def initialize_parameter(layer_dims):
    L = len(layer_dims) - 1
    parameters = {}
    for l in range(1, L+1):
        parameters["W"+str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1])/ np.sqrt(layer_dims[l-1])
        



