import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Load iris.data into a dataframe
df = pd.read_csv('iris.data')

# Declare X and create a KMeans classifier
X = df.iloc[0:100, [2, 3]].values
clf = KMeans(n_clusters=2)
clf.fit(X)
centroids, labels = clf.cluster_centers_, clf.labels_

print('Centroids:', centroids)

# Plot data for visualization
# Red x's denote the centroids while the two class labels are depicted in pink and cyan
plt.scatter(centroids[:, 0], centroids[:, 1], color='red', marker='x')
plt.scatter(X[:50, 0], X[:50, 1], color='pink', marker='.')
plt.scatter(X[50:, 0], X[50:, 1], color='cyan', marker='.')
plt.show()