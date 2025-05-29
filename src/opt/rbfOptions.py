import numpy as np


class RBFoptions:
    """
        Options for the RBF surrogate models

        :param model: RBF kernel type (currently only ``"cubic"``, but others will follow soon)
        :param degree: degree of polynomial tail for RBF kernel. If None, then ``RBFInterpolator`` will set it depending on kernel type. See SciPy's RBFInterpolator documentation for details.
        :param rho:  ``0``: interpolating RBFs, ``> 0``: approximating (spline-like) RBFs. The larger ``rho`` the smoother
        :param rhoDec: exponential decay factor for ``rho``
        :param rhoGrow: every ``rhoGrow`` (e.g. 100) iterations, re-enlarge ``rho``. If 0, then re-enlarge never
        :param width: only relevant for scalable (e.g. Gaussian) kernels. Determines the width :math:`\sigma`
        :param widthFactor: only for scalable kernels. Additional constant factor applied to each width :math:`\sigma`
        :param gaussRule: only relevant for Gaussian kernels, see ``trainGaussRBF``
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
