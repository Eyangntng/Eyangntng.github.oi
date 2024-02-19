import numpy as np
import matplotlib.pyplot as plt

# Generate data
np.random.seed(0)
x = np.random.normal(50, 10, 100)
y = np.random.normal(50, 10, 100)

# Create a figure and axis
fig, ax = plt.subplots()

# Scatter plot
ax.scatter(x, y, color='black', alpha=0.5)

# Create a grid of points
x_grid = np.linspace(min(x), max(x), 100)
y_grid = np.linspace(min(y), max(y), 100)

# Create a meshgrid
X, Y = np.meshgrid(x_grid, y_grid)

# Fit a Gaussian Mixture Model (GMM)
from sklearn.mixture import GaussianMixture
gmm = GaussianMixture(n_components=2, random_state=0)
gmm.fit(np.column_stack((x, y)))

# Calculate the density
Z = gmm.predict_proba(np.column_stack((X, Y)))

# Plot the density
ax.contour(X, Y, Z[:, 1].reshape(100, 100), levels=[0.5], colors='red', linestyles='solid')

# Plot the data points
ax.scatter(x, y, color='black', alpha=0.5)

# Show the plot
plt.show()