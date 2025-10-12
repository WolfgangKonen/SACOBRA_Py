from enum import Enum


class RIoptions:
    """
        Options for :class:`.RepairInfeasibleRI2` (method to repair infeasible solutions)

        The **infeasibility** of a solution is its maximum constraint violation  (0 for a feasible solution).

        A solution :math:`x` is :math:`\\epsilon`-**feasible** for constraint function :math:`g` if
        :math:`g(x) + \\epsilon < 0`.

        :param RIMODE: one out of 0,1,2,3 with 0,1: deprecated older versions of RI2,
            2: the recommended RI2-case, see :class:`RepairInfeasibleRI2`,
            3: Chootinan's method (not implemented)
        :param eps1: include all constraints not eps1-feasible into the repair mechanism
        :param eps2: selects the solution with the shortest shift among all random
            realizations which are eps2-feasible
        :param q: draw coefficients :math:`\\alpha_k` from uniform distribution :math:`U[0, q]`
        :param mmax: draw ``mmax`` random realizations
        :param gradEps: stepsize for numerical gradient calculation
        :param repairMargin: repair only solutions whose infeasibility is less than this margin
        :param repairOnlyFresBetter: if True, then repair only iterates with
            ``fitness < so-far-best-fitness + marFres``
        :param marFres: only relevant if ``repairOnlyFresBetter==True``
        :param trueFuncForSurrogates: use true constraint functions instead of constraint surrogates

    """
    def __init__(self,
                 RIMODE=2,    # 0, 1 (deprecated older versions of RI2) | 2 (the recommended RI2-case) |
                              # 3 (Chootinan's method, TODO)
                 eps1=1e-4,
                 eps2=1e-4,
                 q=3.0,
                 mmax=1000,
                 gradEps=None,
                 repairMargin=1e-2,
                 repairOnlyFresBetter=False,
                 marFres=0.0,  # only relevant if repairOnlyFresBetter==True
                 trueFuncForSurrogates=False,
                 ):
        self.RIMODE = RIMODE
        self.eps1 = eps1
        self.eps2 = eps2
        self.q = q
        self.mmax = mmax
        self.gradEps = gradEps
        self.repairMargin = repairMargin
        self.repairOnlyFresBetter = repairOnlyFresBetter
        self.marFres = marFres
        self.trueFuncForSurrogates = trueFuncForSurrogates
