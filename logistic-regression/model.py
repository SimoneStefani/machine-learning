# Package imports
import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
from utilities import load_dataset


##---------------------------------------------------------
# PREPARE DATASET
##---------------------------------------------------------

# Loading the data (cat/non-cat)
train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()

# Reshape the training and test examples
train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T

# Standardize dataset
train_set_x = train_set_x_flatten / 255.
test_set_x = test_set_x_flatten / 255.


##---------------------------------------------------------
# SETUP LINEAR REGRESSION ALGORITHM
##---------------------------------------------------------

# Compute the sigmoid function of z
# Arguments:    z -- A scalar or numpy array of any size.
# Return:       s -- sigmoid(z)
def sigmoid(z):
    s = 1 / (1 + np.exp(-z))
    return s

# This function creates a vector of zeros of shape (dim, 1) for w and initializes b to 0.
# Argument:     dim -- size of the w vector we want (or number of parameters in this case)
# Returns:      w -- initialized vector of shape (dim, 1)
#               b -- initialized scalar (corresponds to the bias)
def initialize_with_zeros(dim):
    w = np.zeros((dim, 1))
    b = 0

    assert(w.shape == (dim, 1))
    assert(isinstance(b, float) or isinstance(b, int))

    return w, b

# Implement the cost function and its gradient for the propagation explained above
# Arguments:    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
#               b -- bias, a scalar
#               X -- data of size (num_px * num_px * 3, number of examples)
#               Y -- true "label" vector (containing 0 if non-cat, 1 if cat) of size (1, number of examples)
# Return:       cost -- negative log-likelihood cost for logistic regression
#               dw -- gradient of the loss with respect to w, thus same shape as w
#               db -- gradient of the loss with respect to b, thus same shape as b
def propagate(w, b, X, Y):

    m = X.shape[1]

    # FORWARD PROPAGATION (FROM X TO COST)
    A = sigmoid(np.dot(w.T, X) + b)                                     # compute activation
    cost = - (1 / m) * np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A))     # compute cost

    # BACKWARD PROPAGATION (TO FIND GRAD)
    dw = (1 / m) * np.dot(X, (A - Y).T)
    db = (1 / m) * np.sum(A - Y)

    assert(dw.shape == w.shape)
    assert(db.dtype == float)
    cost = np.squeeze(cost)
    assert(cost.shape == ())

    grads = {"dw": dw, "db": db}

    return grads, cost

# This function optimizes w and b by running a gradient descent algorithm
# Arguments:    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
#               b -- bias, a scalar
#               X -- data of shape (num_px * num_px * 3, number of examples)
#               Y -- true "label" vector (containing 0 if non-cat, 1 if cat), of shape (1, number of examples)
#               num_iterations -- number of iterations of the optimization loop
#               learning_rate -- learning rate of the gradient descent update rule
#               print_cost -- True to print the loss every 100 steps
# Returns:      params -- dictionary containing the weights w and bias b
#               grads -- dictionary containing the gradients of the weights and bias with respect to the cost function
#               costs -- list of all the costs computed during the optimization, this will be used to plot the learning curve.
def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost = False):

    costs = []

    for i in range(num_iterations):

        # Cost and gradient calculation (≈ 1-4 lines of code)
        grads, cost = propagate(w, b, X, Y)

        # Retrieve derivatives from grads
        dw = grads["dw"]
        db = grads["db"]

        # update rule (≈ 2 lines of code)
        w = w - learning_rate * dw
        b = b - learning_rate * db

        # Record the costs
        if i % 100 == 0:
            costs.append(cost)

        # Print the cost every 100 training examples
        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))

    params = {"w": w, "b": b}

    grads = {"dw": dw, "db": db}

    return params, grads, costs

# Predict whether the label is 0 or 1 using learned logistic regression parameters (w, b)
# Arguments:    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
#               b -- bias, a scalar
#               X -- data of size (num_px * num_px * 3, number of examples)
# Returns:      Y_prediction -- a numpy array (vector) containing all predictions (0/1) for the examples in X
def predict(w, b, X):

    m = X.shape[1]
    Y_prediction = np.zeros((1,m))
    w = w.reshape(X.shape[0], 1)

    # Compute vector "A" predicting the probabilities of a cat being present in the picture
    A = sigmoid(np.dot(w.T, X) + b)

    Y_prediction = np.where(A > 0.5, 1, 0)

    assert(Y_prediction.shape == (1, m))

    return Y_prediction


##---------------------------------------------------------
# SETUP NEURAL NETWORK MODEL
##---------------------------------------------------------

# Builds the logistic regression model by calling the function you've implemented previously
# Arguments:    X_train -- training set represented by a numpy array of shape (num_px * num_px * 3, m_train)
#               Y_train -- training labels represented by a numpy array (vector) of shape (1, m_train)
#               X_test -- test set represented by a numpy array of shape (num_px * num_px * 3, m_test)
#               Y_test -- test labels represented by a numpy array (vector) of shape (1, m_test)
#               num_iterations -- hyperparameter representing the number of iterations to optimize the parameters
#               learning_rate -- hyperparameter representing the learning rate used in the update rule of optimize()
#               print_cost -- Set to true to print the cost every 100 iterations
# Returns:      d -- dictionary containing information about the model.
def model(X_train, Y_train, X_test, Y_test, num_iterations = 2000, learning_rate = 0.5, print_cost = False):

    # initialize parameters with zeros
    w, b = initialize_with_zeros(np.shape(X_train)[0])

    # Gradient descent
    parameters, grads, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost)

    # Retrieve parameters w and b from dictionary "parameters"
    w = parameters["w"]
    b = parameters["b"]

    # Predict test/train set examples
    Y_prediction_test = predict(w, b, X_test)
    Y_prediction_train = predict(w, b, X_train)

    # Print train/test Errors
    print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))


    d = {"costs": costs,
         "Y_prediction_test": Y_prediction_test,
         "Y_prediction_train" : Y_prediction_train,
         "w" : w,
         "b" : b,
         "learning_rate" : learning_rate,
         "num_iterations": num_iterations}

    return d


##---------------------------------------------------------
# RUN NEURAL NETWORK MODEL
##---------------------------------------------------------

d = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations = 2000, learning_rate = 0.005, print_cost = True)

# Plot learning curve (with costs)
costs = np.squeeze(d['costs'])
plt.plot(costs)
plt.ylabel('cost')
plt.xlabel('iterations (per hundreds)')
plt.title("Learning rate =" + str(d["learning_rate"]))
plt.show()
