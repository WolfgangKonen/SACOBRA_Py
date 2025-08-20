import numpy as np
from sklearn.metrics.pairwise import euclidean_distances        # scikit-learn
from itertools import combinations_with_replacement
from scipy.special import comb


def svd_inv(M):
    U, D, Vh = np.linalg.svd(M, full_matrices=True)
    eps = 1e-14
    maxd = np.max(D)
    invD = 1 / D
    invD[np.abs(D)/maxd < eps] = 0
    invM = np.matmul(Vh.T, np.matmul(np.diag(invD), U.T))
    return invM


def dist_line(x: np.ndarray, xp: np.ndarray):
    """
    Euclidean distance of ``x`` to all points ``xp``

    It is perhaps easier to use
    `euclidean_distances <https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.euclidean_distances.html>`_
    from scikit-learn. But measurements have shown that it is 6 - 10 times (!) *slower*.

    :param x:   vector of dimension d
    :param xp:  n points x_i of dimension d are arranged in (n x d) matrix ``xp``.
                If ``xp`` is a vector, it is interpreted as (n x 1) matrix, i.e. d=1.
    :return:    vector of length n, the Euclidean distances
    """
    if xp.ndim == 1:
        xp = xp.reshape(xp.size, 1)
    assert x.ndim == 1 and x.shape[0] == xp.shape[1]
    z = np.tile(x, (xp.shape[0], 1)) - xp
    z = np.sqrt(np.sum(z*z, axis=1))
    return z


def calc_rhs(U: np.ndarray, d2: int) -> np.ndarray:
    if d2 == 0:         # no polynomial tail
        rhs = U
    else:               # with polynomial tail
        if U.ndim == 1:
            rhs = np.hstack((U, np.zeros(d2)))
        elif U.ndim == 2:
            rhs = np.vstack((U, np.zeros((d2, U.shape[1]))))
        else:
            raise RuntimeError("[calcRHS] Matrix U is neither 1D nor 2D")
    return rhs


def _monomial_powers(ndim, degree) -> np.ndarray:
    """
    Return the powers for each monomial in a polynomial.

    Borrowed from
    `SciPy's RBFInterpolator <https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.RBFInterpolator.html>`_
    and extended with special case degree=1.5.

    Parameters
    ----------
    ndim : int
        Number of variables in the polynomial.
    degree : float
        Degree of the polynomial (with 1.5 as special case).

    Returns
    -------
    (nmonos, ndim) int ndarray
        Array where each row contains the powers for each variable in a
        monomial.

    """
    if degree == 1.5:
        # SACOBRA special case: pure squares (ptail=T, squares=T)
        nmonos = 2 * ndim + 1
        out = np.zeros((nmonos, ndim), dtype=np.dtype("long"))
        for i in range(ndim):
            out[     i+1, i] = 1
            out[ndim+i+1, i] = 2
    else:
        # the general case borrowed from SciPy's RBFInterpolator
        degree = int(degree)
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


def polynomial_matrix(x, powers):
    """Evaluate monomials, with exponents from `powers`, at `x`."""
    out = np.empty((x.shape[0], powers.shape[0]), dtype=float)
    for i in range(x.shape[0]):
        for j in range(powers.shape[0]):
            out[i, j] = np.prod(x[i]**powers[j])   # e.g.: x[i] = [2, 5], powers[j] = [0, 2] --> out[i,j] = 2**0 + 5**2
    return out


def polynomial_vector(x, powers):
    """Evaluate monomials, with exponents from `powers`, at the point `x`."""
    out = np.empty((powers.shape[0]), dtype=float)
    for i in range(powers.shape[0]):
        out[i] = np.prod(x**powers[i])
    return out


class RBFsacob:
    """
    RBF model implemented in SACOBRA. In contrast to SciPy's RBFInterpolator implementation, it allows ``degree=1.5``
    which means linear polynomial tail plus **pure** square terms ``x1**2, x2**2, ...``
    """

    def __init__(self, xobs, yobs, kernel="cubic", degree=1.5, rho=0, width=-1, test_pmat=False):
        """
        Create RBF model(s) from observations ``(xobs,yobs)``. Shape m of ``yobs`` controls whether one RBF
        model (m=1) or several RBF models (m>1) are formed.

        :param xobs:    (n x d)-matrix of n d-dimensional vectors :math:`\\vec{x}_i,\\, i=0,...,n-1`
        :param yobs:    vector of shape (n,) with observations :math:`f(\\vec{x}_i)` - or -
                        matrix of shape (n,m) with observations :math:`f_j(\\vec{x}_i)` for :math:`m` functions
                        :math:`f_j,\\, j=0,...,m-1`
        :param kernel:  RBF kernel type, see :class:`.RBFoptions` for details. The names should match to SciPy's
                        `RBFInterpolator <https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.RBFInterpolator.html>`_
        :param degree:  degree of polynomial tail, see :ref:`below <degree_label>`
        :param rho:     smoothing parameter. If = 0, we have interpolating RBFs, if > 0, we have approximating RBFs (the
                        larger ``rho``, the more approximating)
        :param width:   optional width parameter for Gaussian or MQ RBFs, see :class:`.RBFoptions` for details
        :param test_pmat: only for testing the RBFsacob implementation (degree=1, 1.5)

        .. _degree_label:

        Parameter ``degree`` controls whether RBF models are augmented with a polynomial tail. Allowed values:

        - 0 or -1: no polynomial tail
        - 1: linear polynomial tail
        - 1.5: linear plus **pure** squares tail (e.g. x1*x1, x2*x2) (option ``squares=T`` in R's SACOBRA)
        - 2: linear plus quadratic polynomial tail (**all** monomials of degree 1 and 2)

        """
        self.xp = xobs
        self.npts = xobs.shape[0]
        self.d = xobs.shape[1]
        self.degree = degree
        self.type = kernel
        self.rho = rho
        self.width = width
        self.coef = None        # will be set in trainRBF
        self.powers = None      # will be set in trainRBF
        self.d2 = None          # will be set in trainRBF, no. of columns of polynomial tail matrix
        self.shift = 0.0        # shift for x in polynomial tail
        self.scale = 1.0        # scale for x in polynomial tail
        self.test_pmat = test_pmat

        switcher = {
            "cubic": self.trainCubicRBF,
            "gaussian": self.trainGaussianRBF,
            "multiquadric": self.trainMQRBF,
            "quintic": self.trainQuinticRBF,
            "thin_plate_spline": self.trainThinRBF,
        }
        mdl_func = switcher.get(kernel, lambda xobs, yobs, degree, rho, width: "not implemented")
        mdl = mdl_func(xobs, yobs, degree, rho, width)
        assert mdl != "not implemented", f"Model type {kernel} is not (yet) implemented"

    def trainCubicRBF(self, xp, U, degree, rho, width):
        edist = euclidean_distances(xp, xp)     # distance between rows of xp
        phi = edist * edist * edist             # cubic RBF (npts x npts matrix)

        self.trainRBF(phi, U, degree, xp, rho)
        return "cubic"

    def trainGaussianRBF(self, xp, U, degree, rho, width):
        edist = euclidean_distances(xp, xp)     # distance between rows of xp
        phi = np.exp(-0.5 * (edist * edist) / width)

        self.trainRBF(phi, U, degree, xp, rho)
        return "gaussian"

    def trainMQRBF(self, xp, U, degree, rho, width):
        edist = euclidean_distances(xp, xp)     # distance between rows of xp
        phi = -np.sqrt(1 + 0.5 * (edist * edist) / width)
        # the multiquadric function with "-" in front and factor 0.5 differs slightly from that in R,
        # however, this way it matches exactly with SciPy's RBFinterpolator.

        self.trainRBF(phi, U, degree, xp, rho)
        return "multiquadric"

    def trainQuinticRBF(self, xp, U, degree, rho, width):
        edist = euclidean_distances(xp, xp)     # distance between rows of xp
        phi = edist ** 5                # quintic RBF (npts x npts matrix)

        self.trainRBF(phi, U, degree, xp, rho)
        return "quintic"

    def trainThinRBF(self, xp, U, degree, rho, width):
        edist = euclidean_distances(xp, xp)     # distance between rows of xp
        nu = 1e-10
        phi = edist * edist * np.log(edist + nu)           # thin-plate-spline RBF (npts x npts matrix)

        self.trainRBF(phi, U, degree, xp, rho)
        return "thin_plate"

    def trainRBF(self, phi, U, degree: float, xp, rho=0):
        """
        Internal function to train RBFs.

        After matrix ``phi`` is built in methods trainCubicRBF, trainGaussianRBF or trainMQRBF, ...,
        it will be passed to this function where it becomes augmented by the polynomial tail (if ``degree > 0``).
        Then, the RBF model coefficients are calculated by inverting the augmented matrix.

        Allowed values for ``degree``:

        - 0 or -1: no polynomial tail
        - 1: linear polynomial tail
        - 1.5: linear plus pure squares (e.g. x1*x1, x2*x2) (option ``squares`` in R's SACOBRA)
        - 2: linear plus quadratic polynomial tail (all monomials of degree 1 and 2)

        Return ``self`` with elements ``coef`` and ``d2`` filled appropriately:

        - ``self.d2``: number of columns in polynomial tail matrix
        - ``self.d2 = 0 | 1+d | 1+2*d | 1+d+d**2``  for ``degree = 0 | 1 | 1.5 | 2``
        - ``self.coef``: ``((npts+d2) x m)`` matrix holding in column ``m`` the coefficients for the ``m``'th model.

        ``d`` is the dimension of observation vectors :math:`\\vec{x}_i`, i.e. the number of columns of ``xobs``.
        """
        npts = xp.shape[0]
        assert phi.shape[0] == npts and phi.shape[1] == npts
        assert degree in {-1, 0, 1, 1.5, 2}, f"[RBFsacob.trainRBF] degree={degree} is not allowed"

        phi = phi + npts*rho*np.eye(npts)

        if degree == 0 or degree == -1:
            M = phi.copy()
            d2 = 0      # size of polynomial tail
        else:    # i.e degree in {1, 1.5, 2}:
            # for polynomial tail, scale all training points xp the same way as it is done in SciPy's RBFinterpolator:
            # Shift and scale the polynomial domain to be between -1 and 1 (for each dimension).
            mins = np.min(xp, axis=0)
            maxs = np.max(xp, axis=0)
            self.shift = (maxs + mins) / 2
            self.scale = (maxs - mins) / 2
            # The scale may be zero if there is a single point or all the points have
            # the same value for some dimension. Avoid division by zero by replacing
            # zeros with ones.
            self.scale[self.scale == 0.0] = 1.0
            p_x = (xp - self.shift) / self.scale

            # --- Bug before 2025/08/16: this old version was wrong for degree >= 2, because mixed monomials like x1*x2
            # --- would appear multiple times:
            # pMat = np.hstack((np.ones(npts).reshape(npts,1), p_x))   # linear tail LH(1,x1,x2,...)
            # if degree == 1.5:
            #     pMat = np.hstack((pMat, p_x*p_x))         # ... plus direct squares x1^2, x2^2, ...
            # if degree == 2:
            #     for i in range(self.d):                 # ... plus all quadratic terms x1*x1, x1*x2, ...:
            #         pMat = np.hstack((pMat, np.repeat(p_x[:,i], self.d).reshape(npts,self.d) * p_x))

            # --- Bug fix 2025/08/16: the new version (borrowed from SciPy's RBFInterpolator) works correctly for
            # --- degree >= 2 (and _monomial_powers is also extended for the special case degree == 1.5)
            self.powers = _monomial_powers(p_x.shape[1], degree)      # save powers to self for later use in interpRBF
            pMat = polynomial_matrix(p_x, self.powers)

            if self.test_pmat:      # this is just a test that degree=1 and =1.5 is the same as with the
                                    # old version (which was buggy for degree=2)
                pMat_old = np.hstack((np.ones(npts).reshape(npts, 1), p_x))   # linear tail LH(1,x1,x2,...)
                if degree == 1.5:
                    pMat_old = np.hstack((pMat_old, p_x*p_x))         # ... plus direct squares x1^2, x2^2, ...
                assert np.allclose(pMat, pMat_old)
                print(f"[RBFsacob: assert test_pmat passed] degree={degree}")

            d2 = pMat.shape[1]      # number of columns in polynomial tail matrix
            nMat = np.zeros((d2, d2))
            M = np.hstack((phi, pMat))

            QQ = pMat.T
            M = np.vstack((M, np.hstack((QQ, nMat))))

        invM = svd_inv(M)
        rhs = calc_rhs(U, d2)
        self.coef = np.matmul(invM, rhs)
        self.d2 = d2

    def interpRBF(self, x):
        """
        Apply RBF model(s) to a single data point ``x``.
        """
        assert x.size == self.d, f"x with size {x.size} is not a vector of length self.d={self.d}"

        ed = dist_line(x, self.xp)
        # The following equivalent call would be 6 - 10 times (!!) slower:
        # ed = euclidean_distances(x.reshape(1,self.d),self.xp)

        p_x = (x - self.shift) / self.scale
        ptail = np.array([])
        if self.degree >= 1:
            # --- Bug fix 2025/08/16: the new version (borrowed from SciPy's RBFInterpolator) works correctly for
            # --- all degrees
            ptail = polynomial_vector(p_x, self.powers)

            # --- Bug before 2025/08/16: this old version was wrong for degree >= 2, because mixed monomials like x1*x2
            # --- would appear multiple times:
            # ptail = np.append(1, x)
            # if self.degree == 1.5:
            #     ptail = np.append(ptail, x*x)
            # if self.degree == 2:
            #     for i in range(self.d):
            #         ptail = np.append(ptail, x[i]*x)

        def phi_cubic(ed, width):
            return ed*ed*ed

        def phi_gauss(ed, width):
            return np.exp(-0.5 * (ed * ed) / width)

        def phi_mq(ed, width):
            return -np.sqrt(1 + 0.5 * (ed * ed) / width)

        def phi_quintic(ed, width):
            return ed ** 5

        def phi_thin(ed, width):
            return ed * ed * np.log(ed)

        # TODO: think about whether we need for "gaussian" for each kernel a different width (as on the R side)
        # (lhs below could become a matrix in this case)

        switcher = {
            "cubic": phi_cubic,
            "gaussian": phi_gauss,
            "multiquadric": phi_mq,
            "quintic": phi_quintic,
            "thin_plate_spline": phi_thin,
        }
        phi_func = switcher.get(self.type, lambda ed, width: "not implemented")
        phi = phi_func(ed, self.width)
        assert isinstance(phi, np.ndarray), f"[interpRBF] kernel type {self.type} is not (yet) implemented"

        # if ptail.size == 0:
        #     lhs = phi
        # else:
        #     lhs = np.append(phi, ptail)
        lhs = np.append(phi, ptail)     # this includes lhs = phi for the case ptail.size == 0
        assert lhs.size == self.coef.shape[0]

        return np.matmul(lhs, self.coef)

    def __call__(self, xflat):
        """
        Apply RBF kernel(s) to data ``xflat``.

        :param xflat:   vector of length d  - or -  matrix of shape (n,d)
        :return:        response of kernel(s): See :meth:`__init__` for definition of parameter m.  If m=1, then the
            return value is either a number or a vector of length n, depending on size n of ``xflat``.
            If m>1, then it is either vector of shape m or matrix of shape (n,m), depending on size n of ``xflat``.

        .. hint:: The shape m refers to the size m of yobs in constructor :meth:`.__init__`.

        """
        if xflat.ndim == 1:
            xflat = xflat.reshape(1, xflat.shape[0])
        assert xflat.shape[1] == self.d, "[RBFsacob.__call__] xflat's number of columns differ from self.d"
        # Take each row of xflat in turn (each data point) and apply to it interpRBF.
        # This will be slow if xflat has many rows:
        return np.apply_along_axis(self.interpRBF, axis=1, arr=xflat)
