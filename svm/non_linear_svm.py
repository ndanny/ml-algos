import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from plot_decision_regions import plot_decision_regions

# Create nonlinear data that has the form of an XOR gate
np.random.seed(0)
X_xor = np.random.randn(200, 2)
y_xor = np.logical_xor(X_xor[:, 0] > 0, X_xor[:, 1] > 0)
y_xor = np.where(y_xor, 1, -1)

# Graph xor dataset that we just created
plt.scatter(X_xor[y_xor==1, 0], X_xor[y_xor==1, 1], color='blue', marker='x', label='1')
plt.scatter(X_xor[y_xor==-1, 0], X_xor[y_xor==-1, 1], color='red', marker='s', label='-1')
plt.legend()
plt.ylim(-4.0, 4.0)
plt.xlim(-4.0, 4.0)
plt.show()

# Create classifier with a RBF kernel and graph decision regions
clf = SVC(kernel='rbf', random_state=0, gamma=0.1, C=10.0)
clf.fit(X_xor, y_xor)
print('Created classifier with gamma = 0.1 and C = 10.0')
plot_decision_regions(X_xor, y_xor, classifier=clf)
plt.legend()
plt.show()

# Re-create classifier with different parameters and graph decision regions
clf2 = SVC(kernel='rbf', random_state=0, gamma=0.2, C=1.0)
clf2.fit(X_xor, y_xor)
print('Created classifier with gamma = 0.2 and C = 1.0')
plot_decision_regions(X_xor, y_xor, classifier=clf2)
plt.legend()
plt.show()