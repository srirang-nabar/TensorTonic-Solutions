import numpy as np

def manhattan_distance(x, y):
    """
    Compute the Manhattan (L1) distance between vectors x and y.
    Must return a float.
    """
    x_arr = np.asarray(x)
    y_arr = np.asarray(y)

    dist = np.sum(np.abs(x_arr - y_arr))
    return float(dist)