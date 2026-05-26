import numpy as np

def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    """
    Train logistic regression via gradient descent.
    Return (w, b).
    """
    # X as a list of lists
    D = len(X[0])
    N = len(X)
    w = np.zeros(D)
    b = np.asarray(0.0)

    X = np.asarray(X)
    y = np.asarray(y)

    for step in range(steps):

        p = _sigmoid(np.dot(X, w) + b)

        delta_w = -1*(1/N)*(np.matmul(X.T, (p-y)))
        delta_b = -1*np.mean(p-y)
        
        w+= lr*delta_w
        b+= lr*delta_b

    return (w,b)