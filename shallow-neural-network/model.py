# Package imports
import numpy as np
import matplotlib.pyplot as plt
#from testCases import *
import sklearn
import sklearn.datasets
import sklearn.linear_model
from utilities import plot_decision_boundary, sigmoid, load_planar_dataset

##---------------------------------------------------------
# GENERATE DATASET
##---------------------------------------------------------

# Seed NumPy random generator for consistent data
np.random.seed(1)

X, Y = load_planar_dataset()

# Visualize the data
plt.scatter(X[0, :], X[1, :], c = Y, s = 40, cmap = plt.cm.Spectral);
#plt.show()


##---------------------------------------------------------
# LOGISTIC REGRESSION
##---------------------------------------------------------

# Train a logistic regression model to fit the dataset
clf = sklearn.linear_model.LogisticRegressionCV();
clf.fit(X.T, Y.T);

# Plot the dataset with the classification boundary found
# through logistic regression
plot_decision_boundary(lambda x: clf.predict(x), X, Y)
plt.title("Logistic Regression")
#plt.show()


##---------------------------------------------------------
# SETUP NEURAL NETWORK STRUCTURE
##---------------------------------------------------------

# Arguments:    X -- input dataset of shape (input size, number of examples)
#               Y -- labels of shape (output size, number of examples)
# Returns:      n_x -- the size of the input layer
#               n_h -- the size of the hidden layer
#               n_y -- the size of the output layer
def layer_sizes(X, Y):
    n_x = np.shape(X)[0] # size of input layer
    n_h = 4
    n_y = np.shape(Y)[0] # size of output layer

    return (n_x, n_h, n_y)

# Arguments:    n_x -- size of the input layer
#               n_h -- size of the hidden layer
#               n_y -- size of the output layer
# Returns:      params -- python dictionary containing your parameters:
#                   W1 -- weight matrix of shape (n_h, n_x)
#                   b1 -- bias vector of shape (n_h, 1)
#                   W2 -- weight matrix of shape (n_y, n_h)
#                   b2 -- bias vector of shape (n_y, 1)
def initialize_parameters(n_x, n_h, n_y):

    np.random.seed(2)

    W1 = np.random.randn(n_h, n_x) * 0.01
    b1 = np.zeros((n_h, 1))
    W2 = np.random.randn(n_y, n_h) * 0.01
    b2 = np.zeros((n_y, 1))

    assert (W1.shape == (n_h, n_x))
    assert (b1.shape == (n_h, 1))
    assert (W2.shape == (n_y, n_h))
    assert (b2.shape == (n_y, 1))

    parameters = {"W1": W1, "b1": b1, "W2": W2, "b2": b2}

    return parameters
