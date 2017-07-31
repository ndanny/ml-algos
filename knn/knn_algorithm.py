import warnings
import numpy as np
from collections import Counter

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
    test_dataset = {'p': [[1, 2], [2, 3], [3, 1]], 'r': [[6, 5], [7, 7], [8, 6]]}
    test_sample = [5, 7]

    print('Your test sample', test_sample, 'is classified as', end=' ')
    print(k_nearest_neighbors(test_dataset, test_sample))