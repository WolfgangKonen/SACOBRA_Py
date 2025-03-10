import numpy as np
from scipy.interpolate import RBFInterpolator

class RBFmodel:
    """
    Wrapper for RBFInterpolator to provide a syntax similar to RbfInter.R.
    Usage:
        myModel = RBFmodel(xobs,yobs)   # equivalent to trainCubicRBF
        yflat = myModel(xflat)          # combines interpRBF and predict.RBFinter, with xflat.shape = [d,n]
    """
    def __init__(self, xobs: np.ndarray, yobs: np.ndarray, kernel="cubic", degree: int = None):
        """
        Create RBF model(s) from observations (xobs,yobs)

        :param xobs:    (n x d)-matrix of n d-dimensional vectors x_i
        :param yobs:    vector of shape (n,) with observations f(x_i) - or -
                        matrix of shape (n,m) with observations f_j(x_i), j=1,...m, for m functions
        :param kernel:
        :param degree:  the default None means, that the kernel-specific defaults specified in
                        RBFInterpolator are taken (e.g. degree=1 for "cubic")
        """
        self.d = xobs.shape[1]
        self.nmodels = 1 if yobs.ndim == 1 else yobs.shape[1]
        try:
            self.model = RBFInterpolator(xobs, yobs, kernel=kernel, degree=degree)
        except np.linalg.LinAlgError:
            # LinAlgError ('Singular Matrix') is raised by RBFInterpolator if xobs contains identical rows
            # (identical infill points). We avoid this with cobra.for_rbf['A'] (instead of cobra.sac_res['A']),
            # where a new infill point is added in updateInfoAndCounters ONLY if min(xNewDist), the minimum distance
            # of the new infill points to all rows of cobra.for_rbf['A'] is greater than 0.
            print("[RBFmodel] LinAlgError --> probably identical points in rows of xobs")

    def __call__(self, xflat: np.ndarray):
        """
        Apply RBF model(s) to data xflat

        :param xflat:   vector of length d  - or -  matrix of shape (n,d)
        :return:        response of model(s), either vector of length n  - or -  matrix of shape (n,nmodels)
        """
        if xflat.ndim == 1:
            xflat = xflat.reshape(1, xflat.shape[0])
        assert xflat.shape[1] == self.d, "[RBFmodel.__call__] xflat's number of columns differ from self.d"
        return self.model(xflat)

    # with signature __call__(self, *args, **kwargs)), we would use xflat = args[0]

# probably obsolete:
class RBFmodelCubic(RBFmodel):
    def __init__(self, xobs, yobs):
        super().__init__(xobs, yobs, kernel="cubic")
