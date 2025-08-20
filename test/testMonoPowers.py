# small test program, only to understand
# 1) _monomial_powers (borrowed from SciPy's RBFInterpolator) and
# 2) k-fold cross validation (from https://neptune.ai/blog/cross-validation-in-machine-learning-how-to-do-it-right and
#    https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html )

import numpy as np
from itertools import combinations_with_replacement
from scipy.special import comb

from sklearn.model_selection import KFold


def _monomial_powers(ndim, degree):
    """Return the powers for each monomial in a polynomial.

    Parameters
    ----------
    ndim : int
        Number of variables in the polynomial.
    degree : int
        Degree of the polynomial.

    Returns
    -------
    (nmonos, ndim) int ndarray
        Array where each row contains the powers for each variable in a
        monomial.

    """
    nmonos = comb(degree + ndim, ndim, exact=True)
    out = np.zeros((nmonos, ndim), dtype=np.dtype("long"))
    count = 0
    for deg in range(degree + 1):
        for mono in combinations_with_replacement(range(ndim), deg):
            # `mono` is a tuple of variables in the current monomial with
            # multiplicity indicating power (e.g., (0, 1, 1) represents x1*x2**2,
            # where '0' <--> x1, '1' <--> x2, '2' <--> x3, and so on)
            for var in mono:
                out[count, var] += 1

            count += 1

    return out

def cross_vali():
    """
    Example how to do k-fold cross validation

    from https://neptune.ai/blog/cross-validation-in-machine-learning-how-to-do-it-right

    see also https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html
    """
    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    y = np.array([1, 2, 3, 4])
    kf = KFold(n_splits=3)

    for train_index, test_index in kf.split(X):
        print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        dummy = 0

# print(_monomial_powers(3, 2))
cross_vali()
dummy = 0
