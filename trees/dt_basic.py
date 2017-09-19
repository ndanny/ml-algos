import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.tree import export_graphviz
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from plot_decision_regions import plot_decision_regions

# Load the iris dataset
iris = datasets.load_iris()
X, y = iris.data[:, [2,3]], iris.target

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Create a decision tree classifier
clf = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=0)
clf.fit(X_train, y_train)

# Print accuracy
print('Decision Tree Accuracy:', clf.score(X_test, y_test))

# Plot the decision regions of the tree (rectangles)
plot_decision_regions(X_train, y_train, classifier=clf)
plt.xlabel('Petal Length')
plt.ylabel('Petal Width')
plt.legend()
plt.show()

# Visualize tree using GraphViz (must export file)
export_graphviz(clf, out_file='dt_basic.dot', feature_names=['petal length', 'petal_width'])