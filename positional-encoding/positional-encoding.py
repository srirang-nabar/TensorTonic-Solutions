import numpy as np

def positional_encoding(seq_len, d_model, base=10000.0):
    """
    Return PE of shape (seq_len, d_model) using sin/cos formulation.
    Odd d_model -> last column is sin.
    """
    pos_enc = np.zeros((seq_len, d_model))

    for pos in range(seq_len):
        for dim in range(d_model):
            if dim%2 == 0:
                pos_enc[pos][dim] = np.sin(pos/(base**(dim/d_model)))
            else:
                pos_enc[pos][dim] = np.cos(pos/(base**((dim-1)/d_model)))
    return pos_enc