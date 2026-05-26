import numpy as np

def matrix_transpose(A):
    """
    Return the transpose of matrix A (swap rows and columns).
    """
    n, m = np.shape(np.asarray(A))
    AT = np.zeros((m,n), dtype=int)
    for i in range(m):
        for j in range(n):
            AT[i][j] = A[j][i]
    return AT
