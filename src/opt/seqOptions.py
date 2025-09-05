import numpy as np


class SEQoptions:
    """
    Options for the sequential optimization

    :param optimizer: string defining the optimization method for SACOBRA phase I and II.
            One out of ["COBYLA","ISRESN"] (see :class:`.SeqOptimizer`)
    :param feMax: maximum number of function evaluations on the surrogate model
    :param tol: convergence tolerance for sequential optimizer
    :param conTol: constraint violation tolerance
    :param penaF: (TODO)
    :param sigmaD: (TODO)
    :param epsilonInit: initial value for ``EPS``. ``EPS`` is a constant added to each constraint to maintain a certain
            margin to the boundary. If None, then ``epsilonInit`` is set to :math:`0.005 \ell`,
            where :math:`\ell` is the length of smallest side of search space
    :param epsilonMax: maximum value for ``EPS``. If None,
            then ``epsilonMax`` is set to :math:`2*0.005 \\ell`.
    :param finalEpsXiZero: if True, set in final iteration ``EPS`` and ``XI`` to zero for best exploitation
    :param Tfeas: threshold for count of feasible iterations in a row. If None, :class:`.CobraInitializer` will
            set it to :math:`floor(2\sqrt{d})`.
    :param Tinfeas: threshold for count of infeasible iterations in a row. If None, :class:`.CobraInitializer` will
            set it to :math:`floor(2\sqrt{d})`.
    :param trueFuncForSurrogates: if True, use the true (constraint & fitness) functions instead of surrogates (only for
            debug analysis)
    """
    def __init__(self,
                 optimizer="COBYLA",
                 feMax=1000,
                 tol=1e-6,
                 conTol=0.0,
                 penaF=[3.0, 1.7, 3e5],
                 sigmaD=[3.0, 2.0, 100],
                 epsilonInit=None,
                 epsilonMax=None,
                 finalEpsXiZero=True,  # if True, then set EPS=XI=0 in final iteration (full exploit, might require
                                       # SEQ.conTol=1e-7 instead of 0.0)
                 Tfeas=None,        # will be set in cobraInit
                 Tinfeas=None,      # will be set in cobraInit
                 trueFuncForSurrogates=False
                 ):
        self.optimizer = optimizer
        self.feMax = feMax
        self.tol = tol
        self.conTol = conTol
        self.penaF = penaF
        self.sigmaD = sigmaD
        self.epsilonInit = epsilonInit
        self.epsilonMax = epsilonMax
        self.finalEpsXiZero = finalEpsXiZero
        self.Tfeas = Tfeas
        self.Tinfeas = Tinfeas
        self.trueFuncForSurrogates = trueFuncForSurrogates
