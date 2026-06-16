import numpy as np

def euclidean_distance(x, y):
    """
    Compute the Euclidean (L2) distance between vectors x and y.
    Must return a float.
    """
    x_arr = np.asarray(x)
    y_arr = np.asarray(y)

    dist = np.sqrt(np.sum((x_arr - y_arr)**2))
    return float(dist)