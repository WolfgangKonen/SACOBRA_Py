from enum import Enum


class W_RULE(Enum):
    ONE = 1
    THREE = 3


class RBFoptions:
    """
        Options for the RBF surrogate models

        :param kernel: RBF kernel type, see :ref:`below <kernel_label>`
        :param degree: degree of polynomial tail for RBF kernel. If None, then
                `SciPy's RBFInterpolator <https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.RBFInterpolator.html>`_
                will set it depending on kernel type. See SciPy's RBFInterpolator documentation for details.
        :param rho:  Smoothing parameter :math:`\\rho`. If :math:`\\rho=0`, use **interpolating RBFs**: The surrogate
                model surface passes exactly through the points. If :math:`\\rho>0`, use **approximating RBFs**
                (spline-like). The larger :math:`\\rho`, the smoother the surrogate model.
        :param rhoDec: exponential decay factor for :math:`\\rho`
        :param rhoGrow: every ``rhoGrow`` (e.g. 100) iterations, re-enlarge :math:`\\rho`. If 0, then re-enlarge never
        :param width: only relevant for the scale-variant kernel types. ``width =`` :math:`\\sigma` translates to
                shape parameter ``epsilon =`` :math:`\\epsilon = 1/\\sqrt{2\\sigma}` in
                `SciPy's RBFInterpolator <https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.RBFInterpolator.html>`_.
        :param widthRule: only relevant for the scale-variant kernels: If ``width=None``, calculate the appropriate
                width from the data by heuristic rule ``W_RULE.ONE`` or ``W_RULE.THREE``
        :param widthFactor: only for scale-variant kernels. Additional constant factor applied to each width :math:`\\sigma`
        :param interpolator: "scipy" or "sacobra", which interpolation method to use. In case of "scipy", use
         `SciPy's RBFInterpolator <https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.RBFInterpolator.html>`_
         (faster and simpler in code). In case of  "sacobra", use an object of class :class:`.RBFsacob`, which is
         SACOBRA's own implementation of RBF models (allows with ``degree=1.5`` the option equivalent to ``squares=T``
         in SACOBRA R, which means only pure squares in the polynomial tail).
        :param test_pmat: only for testing the RBFsacob implementation (degree=1, 1.5)

        .. _kernel_label:

        Parameter ``kernel`` specifies the kernel type and is one out of ``"cubic"``, ``"quintic"``,
        ``"thin_plate_spline"``, ``"gaussian"``, ``"multiquadric"``.

        - **scale-invariant** kernel types: ``"cubic"``, ``"quintic"``, ``"thin_plate_spline"``
        - **scale-variant** kernel types: ``"gaussian"``, ``"multiquadric"``
    """
    def __init__(self,
                 kernel="cubic",    # "cubic" | "quintic" | "thin_plate_spline" | "gaussian" | "multiquadric"
                 degree= None,
                 rho=0.0,
                 rhoDec=2.0,        # exponential decay factor for rho
                 rhoGrow=0,
                 width=None,
                 widthFactor=1.0,
                 widthRule=W_RULE.ONE,
                 interpolator="scipy",
                 test_pmat=False    # only for testing RBFsacob implementation (degree=1, 1.5)
                 ):
        """

        """
        self.kernel = kernel
        self.degree = degree
        self.rho = rho
        self.rhoDec = rhoDec
        self.rhoGrow = rhoGrow
        self.width = width
        self.widthFactor = widthFactor
        self.widthRule = widthRule
        self.interpolator=interpolator
        self.test_pmat=test_pmat
