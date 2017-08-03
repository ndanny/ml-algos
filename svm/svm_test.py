import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from plot_decision_regions import plot_decision_regions
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

# Load iris.data into a dataframe
df = pd.read_csv('iris.data')

# Declare X (training data) and y (labels)
X = df.iloc[0:100, [2, 3]].values
y = df.iloc[0:100, 4 ].values
y = np.where(y == 'Iris-setosa', 1, -1)

# Split data into train and test sets with the test set being 20% of the overall data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Create a Support Vector Machine classifier and train it
clf = SVC(kernel='linear', C=1.0, random_state=0)
clf.fit(X_train, y_train)

# Print accuracy (results are consistently 1.0 for the most part)
print('Accuracy:', clf.score(X_test, y_test))

# Plot data for visualization purposes=
plot_decision_regions(X, y, classifier=clf)
plt.xlabel('Petal Length')
plt.ylabel('Petal Width')
plt.legend(loc='upper left')
plt.suptitle('SVM - Petal Length vs. Petal Width')
plt.show()