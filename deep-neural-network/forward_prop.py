import numpy as np
from activation_functions import *


##---------------------------------------------------------
# FORWARD PROPAGATION
##---------------------------------------------------------

# Implement the linear part of a layer's forward propagation.
# Arguments:    A -- activations from previous layer (or input data): (size of previous layer, number of examples)
#               W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
#               b -- bias vector, numpy array of shape (size of the current layer, 1)
# Returns:      Z -- the input of the activation function, also called pre-activation parameter
#               cache -- a python dictionary containing "A", "W" and "b" ; stored for computing the backward pass efficiently
def linear_forward(A, W, b):
    Z = W.dot(A) + b

    assert(Z.shape == (W.shape[0], A.shape[1]))
    cache = (A, W, b)

    return Z, cache


# Implement the forward propagation for the LINEAR->ACTIVATION layer
# Arguments:    A_prev -- activations from previous layer (or input data): (size of previous layer, number of examples)
#               W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
#               b -- bias vector, numpy array of shape (size of the current layer, 1)
#               activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"
# Returns:      A -- the output of the activation function, also called the post-activation value
#               cache -- a python dictionary containing "linear_cache" and "activation_cache";
#                   stored for computing the backward pass efficiently
def linear_activation_forward(A_prev, W, b, activation):

    if activation == "sigmoid":
        # Inputs: "A_prev, W, b". Outputs: "A, activation_cache".
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = sigmoid(Z)

    elif activation == "relu":
        # Inputs: "A_prev, W, b". Outputs: "A, activation_cache".
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu(Z)

    assert (A.shape == (W.shape[0], A_prev.shape[1]))
    cache = (linear_cache, activation_cache)

    return A, cache


# Implement forward propagation for the [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID computation
# Arguments:    X -- data, numpy array of shape (input size, number of examples)
#               parameters -- output of initialize_parameters_deep()
# Returns:      AL -- last post-activation value
#               caches -- list of caches containing:
#                   every cache of linear_relu_forward() (there are L-1 of them, indexed from 0 to L-2)
#                   the cache of linear_sigmoid_forward() (there is one, indexed L-1)
def L_model_forward(X, parameters):

    caches = []
    A = X
    L = len(parameters) // 2                  # number of layers in the neural network

    # Implement [LINEAR -> RELU]*(L-1). Add "cache" to the "caches" list.
    for l in range(1, L):
        A_prev = A
        A, cache = linear_activation_forward(A_prev, parameters['W' + str(l)], parameters['b' + str(l)], activation = "relu")
        caches.append(cache)

    # Implement LINEAR -> SIGMOID. Add "cache" to the "caches" list.
    AL, cache = linear_activation_forward(A, parameters['W' + str(L)], parameters['b' + str(L)], activation = "sigmoid")
    caches.append(cache)

    assert(AL.shape == (1,X.shape[1]))

    return AL, caches
