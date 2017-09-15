import numpy as np
import plot_decision_regions
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Disable warnings
import warnings
warnings.filterwarnings("ignore")

# Load dataset
iris = datasets.load_iris()
X, y = iris.data[:, [2,3]], iris.target

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Scale data for efficiency
sc = StandardScaler()
sc.fit(X_train)
X_train_sc = sc.transform(X_train)
X_test_sc = sc.transform(X_test)
print('Before Scaling:', X_train[0])
print('After Scaling:', X_train_sc[0])

# Train data
clf = SVC(C=1.0, random_state=0, kernel='linear', probability=True)
clf.fit(X_train_sc, y_train)

# Print accuracy
print('Accuracy:', clf.score(X_test_sc, y_test))
print('Classes:', tuple(np.unique(y_test)))
print('Probability per class:', clf.predict_proba(np.array(X_test_sc[9])))
print('Prediction of value 10:', clf.predict(X_test_sc[9]))

# Plot decision regions
plot_decision_regions.plot_decision_regions(X_train_sc, y_train, classifier=clf)
plt.legend(loc='upper left')
plt.xlabel('Petal length (Scaled)')
plt.ylabel('Petal width (Scaled)')
plt.show()