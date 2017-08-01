import warnings
import numpy as np
import pandas as pd
import random
from collections import Counter
from time import time

def euclidean_distance(x, y):
    """Calculates the euclidean distance between two points x and y.
    Each point lay in the same dimension"""
    return np.linalg.norm(np.array(x) - np.array(y))

def k_nearest_neighbors(dataset, sample, k=3):
    """Determines the proper class of a single sample (of type list)
    given a dataset (of type dict) and value of k.

    Does not account for errors when dataset entries and sample are different shapes
    """
    if len(dataset) >= k: warnings.warn('k is less than the total number of classified groups.')
    distances = []

    # Calculate euclidean distance between sample point and data points.
    # Append to the distances list
    for group in dataset:
        for entry in dataset[group]:
            distances.append((euclidean_distance(entry, sample), group))

    # Create a list of the k nearest neighbors
    closest_k = [entry[1] for entry in sorted(distances)[0:k]]

    # Return the most common neighbor in closest_k
    return Counter(closest_k).most_common(1)[0][0]

if __name__ == '__main__':
    # Start timer right before loading the breast cancer data
    start_time = time()

    # Create and modify a dataframe that reads the breast cancer data
    df = pd.read_csv('breast-cancer-wisconsin.data')
    df.replace('?', -99999, inplace=True)
    df.drop(['id'], 1, inplace=True)

    # Convert column entries to type float and shuffle list elements
    df_float = df.astype(float).values.tolist()
    random.shuffle(df_float)

    # Create a set to train (80% of data) and a set to test (20% of data)
    # Class labels are: 2 for benign and 4 for malignant
    train_set = {2.0: [], 4.0: []}
    test_set = {2.0: [], 4.0: []}

    train_data = df_float[:int(len(df_float) * .8)]
    test_data = df_float[int(len(df_float) * .8):]

    for i in train_data:
        train_set[i[-1]].append(i[:-1])

    for j in test_data:
        test_set[j[-1]].append(j[:-1])

    # Determine the accuracy/score of our test set
    score, total = 0, len(test_data)

    for group in test_set:
        for entry in test_set[group]:
            if k_nearest_neighbors(train_set, entry, k=5) == group:
                score = score + 1

    print('Total Correct:', score)
    print('Total Test Size:', total)
    print('Accuracy:', score/total)
    print('Total Time Taken:', time() - start_time)








