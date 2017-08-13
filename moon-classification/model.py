# Package imports
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import sklearn.datasets
import sklearn.linear_model
import matplotlib


##---------------------------------------------------------
# GENERATE DATASET
##---------------------------------------------------------

# Seed NumPy random generator
np.random.seed(0)

# Generate 200 data points distributed in the shape of two
# interleaving half circles with given noise
X, y = sklearn.datasets.make_moons(200, noise = 0.20)

# Generate scatter plot of the dataset
plt.scatter(X[:, 0], X[:, 1], s = 40, c = y, cmap = plt.cm.Spectral)
