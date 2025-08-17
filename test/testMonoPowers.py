from itertools import combinations_with_replacement
from scipy.special import comb

import numpy as np

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
            # multiplicity indicating power (e.g., (0, 1, 1) represents x*y**2,
            # where '0' <--> x, '1' <--> y, '2' <--> z, and so on)
            for var in mono:
                out[count, var] += 1

            count += 1

    return out

powers = _monomial_powers(3, 2)
print(powers)
dummy = 0
