import numpy as np

def expected_value_discrete(x, p):
    """
    Returns: float expected value
    """
    x = np.asarray(x)
    p = np.asarray(p)
    ones = np.ones(np.shape(p))
    tot_prob = np.dot(ones, p) 

    if (np.shape(x) != np.shape(p)) or not (np.isclose(np.sum(p), 1.0)):
        raise ValueError
    return np.dot(x, p)