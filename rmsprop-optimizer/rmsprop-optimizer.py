import numpy as np

def rmsprop_step(w, g, s, lr=0.001, beta=0.9, eps=1e-8):
    """
    Perform one RMSProp update step.
    """
    s_arr = np.asarray(s)
    g_arr = np.asarray(g)
    w_arr = np.asarray(w)

    s_new = beta*s_arr + (1-beta)*(g_arr*g_arr)
    w_new = w_arr - (lr*g_arr)/(np.sqrt(s_new +eps))
    return w_new, s_new