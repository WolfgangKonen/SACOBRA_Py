import numpy as np
from sklearn.metrics.pairwise import euclidean_distances        # scikit-learn

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

    It is probably easier (and faster?) to use
    `euclidean_distances <https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.euclidean_distances.html>`_
    from scikit-learn.

    :param x:   vector of dimension d
    :param xp:  n points x_i of dimension d are arranged in (n x d) matrix ``xp``.
#'                If ``xp`` is a vector, it is interpreted as (n x 1) matrix, i.e. d=1.
    :return:    vector of length n, the Euclidean distances
    """
    if xp.ndim == 1:
        xp = xp.reshape(xp.size,1)
    assert x.ndim == 1 and x.shape[0] == xp.shape[1]
    z = np.tile(x, (xp.shape[0],1)) - xp
    z = np.sqrt(np.sum(z*z, axis=1))
    return z


def calc_rhs(U: np.ndarray, d2: int) -> np.ndarray:
    if d2 == 0:         # no polynomial tail
        rhs = U
    else:               # with polynomial tail
        if U.ndim == 1:
            rhs = np.hstack((U,np.zeros(d2)))
        elif U.ndim == 2:
            rhs = np.vstack(((U, np.zeros((d2, U.shape[1])) )) )
        else:
            raise RuntimeError("[calcRHS] Matrix U is neither 1D nor 2D")
    return rhs


class RBFsacob:
    """
    RBF model implemented in SACOBRA. In contrast to the ScyPy RBF model implementation, it allows degree==1.5
    which means linear polynomial tail plus pure square terms x1**2, x2**2, ...
    """

    def __init__(self, xobs, yobs, kernel="cubic", degree=1.5, rho=0, width=-1):
        """
        Create RBF model(s) from observations ``(xobs,yobs)``. Shape m of ``yobs`` controls whether one RBF
        model (m=1) or several RBF models (m>1) are formed.

        :param xobs:    (n x d)-matrix of n d-dimensional vectors :math:`\\vec{x}_i,\, i=0,...,n-1`
        :param yobs:    vector of shape (n,) with observations :math:`f(\\vec{x}_i)` - or -
                        matrix of shape (n,m) with observations :math:`f_j(\\vec{x}_i)` for :math:`m` functions :math:`f_j,\, j=0,...,m-1`
        :param kernel:  RBF kernel type, currently only "cubic" implemented. The names should match to SciPy's
                        `RBFInterpolator <https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.RBFInterpolator.html>`_
        :param degree:  degree of polynomial tail, see below
        :param rho:     smoothing parameter. If = 0, we have interpolating RBFs, if > 0, we have approximating RBFs (the
                        larger ``rho``, the more approximating)
        :param width:   optional width parameter for Gaussian or MQ RBFs

        Parameter ``degree`` controls whether RBF models are augmented with a polynomial tail. Allowed values:

        - 0: no polynomial tail
        - 1: linear polynomial tail
        - 1.5: linear plus pure squares tail (e.g. x1*x1, x2*x2) (option ``squares`` in R's SACOBRA)
        - 2: linear plus quadratic polynomial tail (all monomials of degree 1 and 2)

        """
        self.xp = xobs
        self.npts = xobs.shape[0]
        self.d = xobs.shape[1]
        self.degree = degree
        self.type = kernel
        self.rho = rho
        self.width = width
        self.coef = None        # will be set in trainRBF
        self.d2 = None          # will be set in trainRBF, no. of columns of polynomial tail matrix

        switcher = {
            "cubic": self.trainCubicRBF,
            "gaussian": self.trainGaussianRBF,
        }
        mdl_func = switcher.get(kernel, lambda xobs, yobs, degree, rho, width: "not implemented")
        mdl = mdl_func(xobs, yobs, degree, rho, width)
        assert mdl != "not implemented", f"Model type {kernel} is not (yet) implemented"

    def trainCubicRBF(self, xp, U, degree, rho, width):
        edist = euclidean_distances(xp, xp)     # distance between rows of xp
        phi = edist*edist*edist                 # cubic RBF (npts x npts matrix)

        self.trainRBF(phi, U, degree, xp, rho)
        return "cubic"

    def trainGaussianRBF(self, xp, U, degree, rho, width):
        raise NotImplementedError("[RBFsacob] kernel='gaussian' not (yet) implemented")
        # return "gaussian"

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

        Return ``self`` with elements ``coef`` and ``d2`` filled appropriately.

        ``self.coef`` is a (npts+d2 x m) matrix holding in column m the coefficients for the m'th model.
        """
        npts = xp.shape[0]
        assert phi.shape[0] == npts and phi.shape[1] == npts
        assert degree in {-1, 0, 1, 1.5, 2}, f"[RBFsacob.trainRBF] degree={degree} is not allowed"

        phi = phi + npts*rho*np.eye(npts)

        if degree == 0 or degree == -1:
            M = phi.copy()
            d2 = 0      # size of polynomial tail
        else:    # i.e degree in {1, 1.5, 2}:
            pMat = np.hstack((np.ones(npts).reshape(npts,1), xp))   # linear tail LH(1,x1,x2,...)
            if degree == 1.5:
                pMat = np.hstack((pMat, xp*xp))         # ... plus direct squares x1^2, x2^2, ...
            if degree == 2:
                for i in range(self.d):                 # ... plus all quadratic terms x1*x1, x1*x2, ...:
                    pMat = np.hstack((pMat, np.repeat(xp[:,i], self.d).reshape(npts,self.d) * xp))

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
        Apply RBF model(s) to a single point ``x``.
        """
        assert x.size == self.d, f"x with size {x.size} is not a vector of length self.d={self.d}"

        ed = dist_line(x, self.xp)
        # The following equivalent call would be 6 - 10 times (!!) slower:
        # ed = euclidean_distances(x.reshape(1,self.d),self.xp)

        ptail = np.array([])
        if self.degree >= 1:
            ptail = np.append(1, x)
            if self.degree == 1.5:
                ptail = np.append(ptail, x*x)
            if self.degree == 2:
                for i in range(self.d):
                    ptail = np.append(ptail, x[i]*x)

        def phi_cubic(ed, width):
            return ed*ed*ed

        def phi_gauss(ed,width):
            return np.exp(-0.5*(ed/width)**2)

        # TODO: think about whether we need for "gaussian" for each model a different width (as on the R side)
        # (lhs below could become a matrix in this case)

        switcher = {
            "cubic": phi_cubic,
            "gaussian": phi_gauss,
        }
        phi_func = switcher.get(self.type, lambda ed, width: "not implemented")
        phi = phi_func(ed, self.width)
        assert type(phi) == np.ndarray, f"[interpRBF] model type {self.type} is not (yet) implemented"

        if ptail.size==0:
            lhs = phi
        else:
            lhs = np.append(phi, ptail)
        assert lhs.size == self.coef.shape[0]

        return np.matmul(lhs, self.coef)

    def __call__(self, xflat):
        """
        Apply RBF model(s) to data ``xflat``.

        :param xflat:   vector of length d  - or -  matrix of shape (n,d)
        :return:        response of model(s): See :meth:`__init__` for definition of parameter m.  If m=1, then the
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


# if __name__ == '__main__':

