import numpy as np
from activation_functions import *


##---------------------------------------------------------
# BACKWARD PROPAGATION
##---------------------------------------------------------

# Implement the linear portion of backward propagation for a single layer (layer l)
# Arguments:    dZ -- Gradient of the cost with respect to the linear output (of current layer l)
#               cache -- tuple of values (A_prev, W, b) coming from the forward propagation in the current layer
# Returns:      dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
#               dW -- Gradient of the cost with respect to W (current layer l), same shape as W
#               db -- Gradient of the cost with respect to b (current layer l), same shape as b
def linear_backward(dZ, cache):

    A_prev, W, b = cache
    m = A_prev.shape[1]

    dW = (1/m) * np.dot(dZ, cache[0].T)
    db = (1/m) * np.sum(dZ, axis = 1, keepdims = True)
    dA_prev = np.dot(cache[1].T, dZ)

    assert (dA_prev.shape == A_prev.shape)
    assert (dW.shape == W.shape)
    assert (db.shape == b.shape)

    return dA_prev, dW, db


# Implement the backward propagation for the LINEAR->ACTIVATION layer.
# Arguments:    dA -- post-activation gradient for current layer l
#               cache -- tuple of values (linear_cache, activation_cache) we store for computing backward propagation efficiently
#               activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"
# Returns:      dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
#               dW -- Gradient of the cost with respect to W (current layer l), same shape as W
#               db -- Gradient of the cost with respect to b (current layer l), same shape as b
def linear_activation_backward(dA, cache, activation):

    linear_cache, activation_cache = cache

    if activation == "relu":
        dZ = relu_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)

    elif activation == "sigmoid":
        dZ = sigmoid_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)

    return dA_prev, dW, db


# Implement the backward propagation for the [LINEAR->RELU] * (L-1) -> LINEAR -> SIGMOID group
# Arguments:    AL -- probability vector, output of the forward propagation (L_model_forward())
#               Y -- true "label" vector (containing 0 if non-cat, 1 if cat)
#               caches -- list of caches containing:
#                   every cache of linear_activation_forward() with "relu" (it's caches[l], for l in range(L-1) i.e l = 0...L-2)
#                   the cache of linear_activation_forward() with "sigmoid" (it's caches[L-1])
# Returns:      grads -- A dictionary with the gradients
#                   grads["dA" + str(l)] = ...
#                   grads["dW" + str(l)] = ...
#                   grads["db" + str(l)] = ...
def L_model_backward(AL, Y, caches):

    grads = {}
    L = len(caches) # the number of layers
    m = AL.shape[1]
    Y = Y.reshape(AL.shape) # after this line, Y is the same shape as AL

    # Initializing the backpropagation
    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL)) # derivative of cost with respect to AL

    # Lth layer (SIGMOID -> LINEAR) gradients. Inputs: "AL, Y, caches". Outputs: "grads["dAL"], grads["dWL"], grads["dbL"]
    current_cache = caches[L - 1]
    grads["dA" + str(L)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL, current_cache, "sigmoid")

    for l in reversed(range(L-1)):
        # lth layer: (RELU -> LINEAR) gradients.
        # Inputs: "grads["dA" + str(l + 2)], caches". Outputs: "grads["dA" + str(l + 1)] , grads["dW" + str(l + 1)] , grads["db" + str(l + 1)]
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l + 2)], current_cache, "relu")
        grads["dA" + str(l + 1)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp

    return grads
