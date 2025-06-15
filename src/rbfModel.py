import numpy as np
from scipy.interpolate import RBFInterpolator


class RBFmodel:
    """
        Wrapper for
        `SciPy's RBFInterpolator <https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.RBFInterpolator.html>`_ to provide
        a syntax similar to SACOBRA's RbfInter.R.

        Usage:

        .. code-block::

          mdl = RBFmodel(xobs,yobs) # equivalent to trainCubicRBF
          yflat = mdl(xflat)        # apply the model to new observations xflat, with xflat.shape = [n,d]
    """
    def __init__(self, xobs: np.ndarray, yobs: np.ndarray, kernel="cubic", degree: int = None, rho=0.0):
        """
        Create RBF model(s) from observations ``(xobs,yobs)``. Shape m of ``yobs`` controls whether one RBF
        model (m=1) or several RBF models (m>1) are formed.

        :param xobs:    (n x d)-matrix of n d-dimensional vectors :math:`\\vec{x}_i,\, i=0,...,n-1`
        :param yobs:    vector of shape (n,) with observations :math:`f(\\vec{x}_i)` - or -
                        matrix of shape (n,m) with observations :math:`f_j(\\vec{x}_i)` for :math:`m` functions :math:`f_j,\, j=0,...,m-1`
        :param kernel:  the allowed kernels of `RBFInterpolator <https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.RBFInterpolator.html>`_
        :param degree:  the default None means, that the kernel-specific defaults specified in
                        `RBFInterpolator <https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.RBFInterpolator.html>`_
                        are taken (e.g. degree=1 for "cubic")
        :param rho:     set `RBFInterpolator <https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.RBFInterpolator.html>`_'s
                        parameter ``smoothing`` to ``N*rho`` where ``N=xobs.shape[0]``
        """
        self.d = xobs.shape[1]
        self.nmodels = 1 if yobs.ndim == 1 else yobs.shape[1]
        N = xobs.shape[0]
        try:
            self.model = RBFInterpolator(xobs, yobs, kernel=kernel, degree=degree, smoothing=N*rho)
        except np.linalg.LinAlgError:
            # LinAlgError ('Singular Matrix') is raised by RBFInterpolator if xobs contains identical rows
            # (identical infill points). We avoid this with cobra.for_rbf['A'] (instead of cobra.sac_res['A']),
            # where a new infill point is added in updateInfoAndCounters ONLY if min(xNewDist), the minimum distance
            # of the new infill points to all rows of cobra.for_rbf['A'] is greater than 0.
            print("[RBFmodel] LinAlgError --> probably identical points in rows of xobs")

    def __call__(self, xflat: np.ndarray):
        """
        Apply RBF model(s) to data ``xflat``.

        :param xflat:   vector of length d  - or -  matrix of shape (n,d)
        :return:        response of model(s): See :meth:`__init__` for definition of parameter m.  If m==1, then the
            return value is either a number or a vector of length n, depending on size n of xflat.
            If m>1, then it is either vector of shape m or matrix of shape (n,m), depending on size n of xflat.

        .. hint:: The shape m refers to the size m of yobs in constructor :meth:`.__init__`.

        """
        if xflat.ndim == 1:
            xflat = xflat.reshape(1, xflat.shape[0])
        assert xflat.shape[1] == self.d, "[RBFmodel.__call__] xflat's number of columns differ from self.d"
        return self.model(xflat)

    # with signature __call__(self, *args, **kwargs)), we would use xflat = args[0]

# # probably obsolete:
# class RBFmodelCubic(RBFmodel):
#     def __init__(self, xobs, yobs):
#         super().__init__(xobs, yobs, kernel="cubic")
