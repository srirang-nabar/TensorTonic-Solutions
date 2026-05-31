import numpy as np

def dropout(x, p=0.5, rng=None):
    """
    Apply dropout to input x with probability p.
    Return (output, dropout_pattern).
    """
    x = np.asarray(x)
    if rng is not None:
        matrix = rng.random(x.shape)
    else:
        matrix = np.random.random(x.shape)

    mask = matrix >= p
    scale = 1/(1-p)

    pattern = mask*scale
    output = x* pattern
    return output, pattern