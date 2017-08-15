# Package imports
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import sklearn.datasets
import sklearn.linear_model
import matplotlib
from utilities import plot_decision_boundary

##---------------------------------------------------------
# GENERATE DATASET
##---------------------------------------------------------

# Seed NumPy random generator
np.random.seed(0)

# Generate 200 data points distributed in the shape of two
# interleaving half circles with given noise
X, y = sklearn.datasets.make_moons(200, noise = 0.20)

# Generate scatter plot of the dataset
plt.scatter(X[:, 0], X[:, 1], s = 10, c = y, cmap = plt.cm.Spectral)
#plt.show()


##---------------------------------------------------------
# LOGISTIC REGRESSION
##---------------------------------------------------------

# Train a logistic regression model to fit the dataset
log_reg_clf = sklearn.linear_model.LogisticRegressionCV()
log_reg_clf.fit(X, y)

# Plot the dataset with the classification boundary found
# through logistic regression
plot_decision_boundary(lambda x: log_reg_clf.predict(x), X, y)
plt.title("Logistic Regression")
#plt.show()


##---------------------------------------------------------
# BUILD 3-LAYERS NEURAL NETWORK MODEL
##---------------------------------------------------------

# Model parameters
nn_input_dim = 2 # input layer dimensionality
nn_output_dim = 2 # output layer dimensionality
alpha = 0.01 # learning rate for gradient descent
reg_lambda = 0.01 # regularization strength

# Helper function to evaluate the total loss on the dataset
def calculate_loss(model):
    W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']

    # Forward propagation to calculate predictions
    z1 = X.dot(W1) + b1
    a1 = np.tanh(z1)
    z2 = a1.dot(W2) + b2
    exp_scores = np.exp(z2)
    probs = exp_scores / np.sum(exp_scores, axis = 1, keepdims = True)

    # Calculating the loss
    #corect_logprobs = -np.log(probs[range(len(X)), y])
    data_loss = np.sum(probs)

    # Add regulatization term to loss (optional)
    #data_loss += reg_lambda/2 * (np.sum(np.square(W1)) + np.sum(np.square(W2)))
    return 1./len(X) * data_loss

# Helper function to predict an output (0 or 1)
def predict(model, x):
    W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']

    # Forward propagation
    z1 = x.dot(W1) + b1
    a1 = np.tanh(z1)
    z2 = a1.dot(W2) + b2
    exp_scores = np.exp(z2)
    probs = exp_scores / np.sum(exp_scores, axis = 1, keepdims = True)
    return np.argmax(probs, axis = 1)

# This function learns parameters for the neural network and returns the model.
# - nn_hdim: Number of nodes in the hidden layer
# - num_passes: Number of passes through the training data for gradient descent
# - print_loss: If True, print the loss every 1000 iterations
def build_model(nn_hdim, num_passes = 20000, print_loss = False):

    # Initialize the parameters to random values. We need to learn these.
    np.random.seed(0)
    W1 = np.random.randn(nn_input_dim, nn_hdim) / np.sqrt(nn_input_dim)
    b1 = np.zeros((1, nn_hdim))
    W2 = np.random.randn(nn_hdim, nn_output_dim) / np.sqrt(nn_hdim)
    b2 = np.zeros((1, nn_output_dim))

    # This is what we return at the end
    model = {}

    # Gradient descent. For each batch...
    for i in range(0, num_passes):

        # Forward propagation
        z1 = X.dot(W1) + b1
        a1 = np.tanh(z1)
        z2 = a1.dot(W2) + b2
        exp_scores = np.exp(z2)
        probs = exp_scores / np.sum(exp_scores, axis = 1, keepdims = True)

        # Backpropagation
        delta3 = probs
        delta3[range(len(X)), y] -= 1
        dW2 = (a1.T).dot(delta3)
        db2 = np.sum(delta3, axis = 0, keepdims = True)
        delta2 = delta3.dot(W2.T) * (1 - np.power(a1, 2))
        dW1 = np.dot(X.T, delta2)
        db1 = np.sum(delta2, axis = 0)

        # Add regularization terms (b1 and b2 don't have regularization terms)
        dW2 += reg_lambda * W2
        dW1 += reg_lambda * W1

        # Gradient descent parameter update
        W1 += -alpha * dW1
        b1 += -alpha * db1
        W2 += -alpha * dW2
        b2 += -alpha * db2

        # Assign new parameters to the model
        model = { 'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}

        # Optionally print the loss.
        # This is expensive because it uses the whole dataset, so we don't want to do it too often.
        if print_loss and i % 1000 == 0:
          print('Loss after iteration {0:10d}: {1:10f}'.format(i, calculate_loss(model)))

    return model

# Build a model with a 3-dimensional hidden layer
model = build_model(3, print_loss = True)

# Plot the decision boundary
plot_decision_boundary(lambda x: predict(model, x), X, y)
plt.title("Decision Boundary for hidden layer size 3")
plt.show()
