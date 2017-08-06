import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn import preprocessing

# The purpose of this script is to gain insight on the Titanic dataset.
# The individuals abroad the Titanic has a unique background which we can use to
# determine if there was a correlation between their data and wether they survived.
#
# Additionally, this code will exemplify using unsupervised clustering and KMeans to work
# on non-numeric data.
#
# The features are:
# ['pclass', 'survived', 'sex', 'age', 'sibsp', 'parch', 'ticket', 'fare', 'cabin', 'embarked', 'boat', 'home.dest']

def to_numeric(dataframe, show_conversion_dict=True):
    """Converts all non-numeric column data to numeric values"""

    # Create a dictionary with dataframe columns as keys and a defaultdict as value
    columns_to_transform = dict()

    # Load into columns_to_transform the columns in the dataframe that are not of
    # type float64 or int64
    for column in dataframe:
        if dataframe[column].dtype not in (np.float64, np.int64):
            columns_to_transform[column] = dict()

    # For each column, we want to make an integer value for each unique entry
    for column in columns_to_transform.keys():
        for item in dataframe[column]:
            if item not in columns_to_transform[column].keys():
                columns_to_transform[column][item] = -1
                columns_to_transform[column][item] = max(columns_to_transform[column].values()) + 1

    # Transform dataframe column entries to return
    for column in columns_to_transform:
        for i in range(len(dataframe[column])):
            dataframe.set_value(i, column, columns_to_transform[column][dataframe[column][i]])

    # Show the conversion dictionary if applicable
    if show_conversion_dict:
        print('Conversion Dictionary:')
        for column in columns_to_transform:
            print(column, '->', columns_to_transform[column])

    return dataframe

def main():
    # Load titanic excel data into a dataframe object, drop the irrelevant columns,
    # fill missing data and convert non-numeric data to meaningful numeric data
    df = pd.read_excel('titanic.xls')
    df.drop(['name', 'body', 'boat'], 1, inplace=True)
    df.fillna(0, inplace=True)
    df = to_numeric(df)

    # Create X and y
    X = np.array(df.drop(['survived'], 1).astype(float))
    X = preprocessing.scale(X)
    y = np.array(df['survived'])

    # Create classifier
    classifier = KMeans(n_clusters=2)
    classifier.fit(X)

    # Run the prediction to test for accuracy
    correct, total = 0, len(X)
    for i in range(total):
        to_predict = np.array(X[i].astype(float))
        to_predict = to_predict.reshape(-1, len(to_predict))
        prediction = classifier.predict(to_predict)
        correct = correct + prediction[0]

    print()
    print('Correct:     ', correct)
    print('Total:       ', total)
    print('Wrong:       ', total - correct)
    print('Accuracy:    ', correct/total)

if __name__ == '__main__':
    main()