import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
from statistics import mean

def best_fit_slope(x, y):
    """Returns the best fit slope given parameters x and y"""
    return ((mean(x) * mean(y)) - mean(x * y)) / ((mean(x) ** 2) - mean(x ** 2))

def best_fit_yint(x, y):
    """Returns the y-intercept value given parameters x and y"""
    return mean(y) - best_fit_slope(x, y) * mean(x)

def squared_error (y1, y2):
    """Returns the squared error
    -----
    Parameters:
        y1: The points of y of the dataset
        y2: The points of y that lie on the regression line
    -----
    """
    return sum((y2 - y1) ** 2)

def co_det(y1, y2):
    """Returns the coefficient of determination"""
    y_mean = [mean(y1) for _ in y1]
    squared_error_reg = squared_error(y1, y2)
    squared_error_y = squared_error(y1, y_mean)
    return 1 - (squared_error_reg / squared_error_y)

if __name__ == '__main__':
    # Declare x, y, m, b and regression_line
    x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype = np.float64)
    y = np.array([2, 4, 5, 8, 1, 12, 14, 25, 15, 20], dtype = np.float64)
    m = best_fit_slope(x, y)
    b = best_fit_yint(x, y)
    regression_line = [((m * i) + b) for i in x]
    r_squared = co_det(y, regression_line)

    # Print results
    print('Regression Line: y = ' + str(m) + 'x + ' + str(b))
    print('R Squared:', r_squared)

    # Plot data and regression line
    style.use('fivethirtyeight')
    plt.scatter(x, y)
    plt.plot(regression_line)
    plt.show()
