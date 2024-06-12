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

# def softmax(z):
#     expZ = np.exp(z)
#     return expZ/(np.sum(expZ, 0))
def softmax(z):
    z_max = np.max(z, axis=0, keepdims=True)
    expZ = np.exp(z - z_max)  # Subtract max for numerical stability
    return expZ / np.sum(expZ, axis=0, keepdims=True)

def relu(Z):
    A = np.maximum(0,Z)
    return A

def tanh(x):
    return np.tanh(x)

def derivative_relu(Z):
    return np.array(Z > 0, dtype = 'float')

def derivative_tanh(x):
    return (1 - np.power(x, 2))

layer_dims = [X_train.shape[1], 100, 200, Y_train.shape[0]]

def initialize_parameter(layer_dims):
    L = len(layer_dims) - 1
    parameters = {}
    for l in range(1, L+1):
        parameters["W"+str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1])/ np.sqrt(layer_dims[l-1])
        parameters["b"+str(l)] = np.zeros((layer_dims[l],1))

    return parameters

def forward_propogation(X,parameters, activation ):
    forward_cache = {}
    L = len(parameters) // 2

    forward_cache["A0"] = X

    for l in range(1,L):
        forward_cache["Z" + str(l)] = parameters["W" + str(l)].dot(forward_cache["A" + str(l-1)]) + parameters["b"+str(l)]
        forward_cache["A"+str(l)] = relu(forward_cache["Z"+str(l)])

    forward_cache["Z" + str(L)] = parameters["W" + str(L)].dot(forward_cache["A" + str(L-1)]) + parameters["b"+str(L)]
    if forward_cache["Z"+str(L)].shape[0] == 0:
        forward_cache["A"+str(L)] = sigmoid(forward_cache["Z"+str(L)])
    else:
        forward_cache["A"+str(L)] = softmax(forward_cache["Z"+str(L)])

    return forward_cache['A' + str(L)], forward_cache


def compute_cost(AL, Y):
    m = Y.shape[1]

    if Y.shape[0] == 1:
        cost = (1./m) * (-np.dot(Y,np.log(AL).T) - np.dot(1-Y, np.log(1-AL).T))
    else:
        cost = -(1./m) * np.sum(Y * np.log(AL))
        
    cost = np.squeeze(cost)  ## Returns a single digit. [[7 ]] fun will return 7
    return cost

def backwar_propogation(AL, Y, parameters, forward_cache, activation):
    grads = {}
    L = len(parameters)//2
    m = Y.shape[1]

    grads["dZ"+str(L)] = AL - Y
    grads["dW"+ str(L)] = (1/m)*np.dot(grads["dZ"+str(L)], forward_cache["A"+str(L-1)].T)
    grads["db"+str(L)] = (1/m)*np.sum(grads["dZ"+str(L)], axis=1, keepdims=True)

    for l in reversed(range(1, L)):
        if activation == 'tanh':
            grads["dZ" + str(l)] = np.dot(parameters['W' + str(l+1)].T,grads["dZ" + str(l+1)])*derivative_tanh(forward_cache['A' + str(l)])
        else:
            grads["dZ" + str(l)] = np.dot(parameters['W' + str(l+1)].T,grads["dZ" + str(l+1)])*derivative_relu(forward_cache['A' + str(l)])
            
        grads["dW" + str(l)] = 1./m * np.dot(grads["dZ" + str(l)],forward_cache['A' + str(l-1)].T)
        grads["db" + str(l)] = 1./m * np.sum(grads["dZ" + str(l)], axis = 1, keepdims = True)

    return grads

def update_parameters(parameters, grads, learning_rate):
    L = len(parameters)//2
    for l in range(1, L+1):
        parameters["W"+str(l)] = parameters["W"+str(l)] - learning_rate * grads["dW" + str(l)]
        parameters["b"+str(l)] = parameters["b"+str(l)] - learning_rate * grads["db" + str(l)]

    return parameters

def predict(X, y, parameters, activation):

    m = X.shape[1]
    y_pred, caches = forward_propogation(X, parameters, activation)
    
    if y.shape[0] == 1:
        y_pred = np.array(y_pred > 0.5, dtype = 'float')
    else:
        y = np.argmax(y, 0)
        y_pred = np.argmax(y_pred, 0)
    
    return np.round(np.sum((y_pred == y)/m), 2)

def Model(X, Y, layer_dims, learning_rate, activation = 'relu', num_iteration = 1000 ):
    parameters = initialize_parameter(layer_dims)

    for i in range (0, num_iteration):
        AL, forward_cache =  forward_propogation(X, parameters, activation='relu')
        cost = compute_cost(AL, Y)
        grads = backwar_propogation(AL, Y, parameters, forward_cache, activation)
        parameters = update_parameters(parameters, grads, learning_rate)

        if i % (num_iteration/10) == 0:
            print("\niter:{} \t cost: {} \t train_acc:{} \t test_acc:{}".format(i, np.round(cost, 2), predict(X_train, Y_train, parameters, activation), predict(X_test, Y_test, parameters, activation)))
        
        if i % 10 == 0:
            print("==", end = '')


    return parameters 


layer_dims = [X_train.shape[0], 20, 7, 5, Y_train.shape[0]]
lr = 0.0075
intr = 1000

parameters = Model(X_train, Y_train, layer_dims, learning_rate= lr, activation='relu', num_iteration=intr)


       




