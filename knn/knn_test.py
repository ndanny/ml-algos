import numpy as np
import pandas as pd
from sklearn import preprocessing, neighbors
from sklearn.model_selection import train_test_split

"""
This uses the Breast Cancer Data from UCI's
Machine Learning Repository to fit a KNN classifier
and to predict data entries given the attribute information.

Attribute Information: (class attribute has been moved to last column)

    #  Attribute                     Domain
    -- -----------------------------------------
    1. Sample code number            id number
    2. Clump Thickness               1 - 10
    3. Uniformity of Cell Size       1 - 10
    4. Uniformity of Cell Shape      1 - 10
    5. Marginal Adhesion             1 - 10
    6. Single Epithelial Cell Size   1 - 10
    7. Bare Nuclei                   1 - 10
    8. Bland Chromatin               1 - 10
    9. Normal Nucleoli               1 - 10
    10. Mitoses                      1 - 10
    11. Class:                       2 for benign, 4 for malignant
"""

# Load breast-cancer-wisconsin.data into a dataframe
df = pd.read_csv('breast-cancer-wisconsin.data')
df.replace('?', -99999, inplace=True)
df.drop(['id'], 1, inplace=True)

# Declare X and y
X = np.array(df.drop('class', 1))
y = np.array(df['class'])

# Split data into train and test sets with the test set being 20% of the overall data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Create a KNN classifier
clf = neighbors.KNeighborsClassifier()
clf.fit(X_train, y_train)

# Find accuracy with test set
accuracy = clf.score(X_test, y_test)
print("Accuracy:", accuracy)

# Create a data entry to predict a result (experimental phase)
data_entry = np.array([5, 1, 1, 3, 2, 1, 1, 3, 1]).reshape(1, -1)
print("Test data entry:", data_entry)
print("Result:", clf.predict(data_entry))
