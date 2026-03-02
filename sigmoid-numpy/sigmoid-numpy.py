import numpy as np

def sigmoid(x):
    """Vectorized sigmoid function.

    Works on scalars, Python lists, and NumPy arrays.
    Always returns a NumPy array of float64.
    Numerically stable via branch masking — no Python loops.
    """
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x)
    pos = x >= 0
    out[pos]  = 1.0 / (1.0 + np.exp(-x[pos]))              # stable for x >= 0
    out[~pos] = np.exp(x[~pos]) / (1.0 + np.exp(x[~pos]))  # stable for x < 0
    return out