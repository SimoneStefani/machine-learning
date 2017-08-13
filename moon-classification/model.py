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
