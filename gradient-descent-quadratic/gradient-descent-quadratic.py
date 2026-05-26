def gradient_descent_quadratic(a, b, c, x0, lr, steps):
    """
    Return final x after 'steps' iterations.
    """
    # for f(x) = ax**2 + bx + c
    # f'(x) = 2ax + b

    for step in range(steps):
        x0 -= lr*(2*a*x0 +b)
    return x0