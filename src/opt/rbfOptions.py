import numpy as np


class RBFoptions:
    """
        Options for the RBF surrogate models

        :param model: RBF kernel type (currently only ``"cubic"``, but others will follow soon)
        :param degree: degree of polynomial tail for RBF kernel. If None, then ``RBFInterpolator`` will set it depending on kernel type. See SciPy's RBFInterpolator documentation for details.
        :param rho:  Smoothing parameter :math:`\\rho`. If :math:`\\rho=0` then *interpolating RBFs*: The surrogate model surface passes exactly through the points. If :math:`\\rho>0`: then *approximating RBFs* (spline-like). The larger :math:`\\rho`, the smoother the surrogate model.
        :param rhoDec: exponential decay factor for :math:`\\rho`
        :param rhoGrow: every ``rhoGrow`` (e.g. 100) iterations, re-enlarge :math:`\\rho`. If 0, then re-enlarge never
        :param width: only relevant for scalable (e.g. Gaussian) kernels. Determines the width :math:`\\sigma`
        :param widthFactor: only for scalable kernels. Additional constant factor applied to each width :math:`\\sigma`
        :param gaussRule: only relevant for Gaussian kernels, see ``trainGaussRBF``
        :param interpolator: "scipy" or "sacobra", which interpolation method to use. In case of "scipy", use
         `SciPy's RBFInterpolator <https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.RBFInterpolator.html>`_
         (faster and simpler in code). In case of  "sacobra", use an object of class :class:`.RBFsacob`, which is
         SACOBRA's own implementation of RBF models (allows with `degree=1.5` the option equivalent to ``squares=T``
         in SACOBRA R, which means only pure squares in the polynomial tail).
    """
    def __init__(self,
                 model="cubic",
                 degree= None,
                 rho=0.0,
                 rhoDec=2.0,        # exponential decay factor for rho
                 rhoGrow=0,
                 width=-1,
                 widthFactor=1.0,
                 gaussRule="One",
                 interpolator="scipy",
                 ):
        """

        """
        self.model = model
        self.degree = degree
        self.rho = rho
        self.rhoDec = rhoDec
        self.rhoGrow = rhoGrow
        self.width = width
        self.widthFactor = widthFactor
        self.gaussRule = gaussRule
        self.interpolator=interpolator
