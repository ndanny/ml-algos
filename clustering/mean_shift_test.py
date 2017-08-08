import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import MeanShift
from sklearn.datasets.samples_generator import make_blobs

# --------
# This experiment tests and visulizes 3-dimensional data using
# sklearn's MeanShift algorithm
# --------

# Create some blobs
X, y = make_blobs(n_samples=100, centers=3, n_features=3, cluster_std=1)

# Create the MeanShift() classifier
classifier = MeanShift()
classifier.fit(X)

# Plot the blobs and their centers
color_list = ['red', 'blue', 'green']

centers = classifier.cluster_centers_
print('Cluster Centers:', centers)

figure = plt.figure()
graph = figure.add_subplot(111, projection='3d')

for i in range(len(centers)):
    graph.scatter(centers[i][0], centers[i][1], centers[i][2], color='orange', marker='x', linewidth='2')

for i in range(len(X)):
    graph.scatter(X[i][0], X[i][1], X[i][2], color=color_list[y[i]], marker='.')

plt.show()
