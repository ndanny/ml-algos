import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.model_selection import train_test_split

# Load breast-cancer-wisconsin.data into a dataframe
df = pd.read_csv('breast-cancer-wisconsin.data')
df.replace('?', -99999, inplace=True)
df.drop(['id'], 1, inplace=True)

# Declare X and y
X = np.array(df.drop('class', 1))
y = np.array(df['class'])

# Split data into train and test sets with the test set being 20% of the overall data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Create a SVM classifier and train the classifier with the test data
clf = svm.SVC()
clf.fit(X_train, y_train)

# Find accuracy with test set
accuracy = clf.score(X_test, y_test)
print("Accuracy:", accuracy)