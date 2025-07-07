import time

import numpy as np
from opt.rbfOptions import RBFoptions
from scipy.interpolate import RBFInterpolator

from rbfSacobra import RBFsacob


class RBFmodel:
    """
        Wrapper for the RBF model which is contained in ``self.model``. ``self.model`` is either
        `SciPy's RBFInterpolator <https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.RBFInterpolator.html>`_
        or it is an object of class :class:`.RBFsacob`, which is SACOBRA's own implementation of RBF models (allows with
        ``degree=1.5`` the option equivalent to ``squares=T`` in R's SACOBRA, which means only pure squares in the
        polynomial tail).

        The wrapper's purpose is to provide a syntax similar to SACOBRA's RbfInter.R.

        Usage:

        .. code-block::

          mdl = RBFmodel(xobs,yobs) # equivalent to trainCubicRBF
          yflat = mdl(xflat)        # apply the model to new observations xflat, with xflat.shape = [n,d]

        It turns out that :class:`RBFsacob` is factor 20-50 slower in __init__ and factor 5-8 slower in __call__ than
        SciPy's RBFInterpolator (see test/results_time_RBF.txt for details). But it has the extra option ``degree=1.5``.
    """
    def __init__(self, xobs: np.ndarray, yobs: np.ndarray, interpolator="scipy",
                 kernel="cubic", degree: float = None, rho=0.0):
        """
        Create RBF model(s) from observations ``(xobs,yobs)``. Shape m of ``yobs`` controls whether one RBF
        model (m=1) or several RBF models (m>1) are formed.

        :param xobs:    (n x d)-matrix of n d-dimensional vectors :math:`\\vec{x}_i,\, i=0,...,n-1`
        :param yobs:    vector of shape (n,) with observations :math:`f(\\vec{x}_i)` - or -
                        matrix of shape (n,m) with observations :math:`f_j(\\vec{x}_i)` for :math:`m` functions :math:`f_j,\, j=0,...,m-1`
        :param interpolator: "scipy" or "sacobra", which interpolation method to use. See :class:`.RBFoptions`.
        :param kernel:  the allowed kernels of `SciPy's RBFInterpolator <https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.RBFInterpolator.html>`_
        :param degree:  the default None means, that the kernel-specific defaults specified in
                        `RBFInterpolator <https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.RBFInterpolator.html>`_
                        are taken (e.g. degree=1 for "cubic")
        :param rho:     set `RBFInterpolator <https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.RBFInterpolator.html>`_'s
                        parameter ``smoothing`` to ``N*rho`` where ``N=xobs.shape[0]`` - or - pass on to class
                        :class:`.RBFsacob`'s parameter ``rho``.
        """
        start = time.perf_counter()
        self.d = xobs.shape[1]
        self.nmodels = 1 if yobs.ndim == 1 else yobs.shape[1]
        N = xobs.shape[0]
        try:
            if interpolator == "scipy":
                self.model = RBFInterpolator(xobs, yobs, kernel=kernel, degree=degree, smoothing=N*rho)
            else:   # i.e. interpolator == "sacobra"
                self.model = RBFsacob(xobs, yobs, kernel=kernel, degree=degree, rho=rho)
        except np.linalg.LinAlgError:
            # LinAlgError ('Singular Matrix') is raised by RBFInterpolator if xobs contains identical rows
            # (identical infill points). We avoid this with cobra.for_rbf['A'] (instead of cobra.sac_res['A']),
            # where a new infill point is added in updateInfoAndCounters ONLY if min(xNewDist), the minimum distance
            # of the new infill points to all rows of cobra.for_rbf['A'] is greater than 0.
            print("[RBFmodel] LinAlgError --> probably identical points in rows of xobs")
        self.time_init = time.perf_counter() - start
        self.time_call = 0.0

    def __call__(self, xflat: np.ndarray):
        """
        Apply RBF model(s) to data ``xflat``.

        :param xflat:   vector of length d  - or -  matrix of shape (n,d)
        :return:        response of model(s): See :meth:`__init__` for definition of parameter m.  If m=1, then the
            return value is either a number or a vector of length n, depending on size n of ``xflat``.
            If m>1, then it is either vector of shape m or matrix of shape (n,m), depending on size n of ``xflat``.

        .. hint:: The shape m refers to the size m of yobs in constructor :meth:`.__init__`.

        """
        start = time.perf_counter()
        if xflat.ndim == 1:
            xflat = xflat.reshape(1, xflat.shape[0])
        assert xflat.shape[1] == self.d, "[RBFmodel.__call__] xflat's number of columns differ from self.d"
        y = self.model(xflat)
        self.time_call += (time.perf_counter() - start)
        return y

    # with signature __call__(self, *args, **kwargs)), we would use xflat = args[0]

# # probably obsolete:
# class RBFmodelCubic(RBFmodel):
#     def __init__(self, xobs, yobs):
#         super().__init__(xobs, yobs, kernel="cubic")
