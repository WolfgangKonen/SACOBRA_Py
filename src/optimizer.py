import time
from abc import ABC, abstractmethod

import nlopt
import numpy as np
import scipy.optimize
from scipy.optimize import Bounds, differential_evolution, NonlinearConstraint

from cobraInit import CobraInitializer
from phase2Vars import Phase2Vars
from seqFuncFactory import SeqFuncFactory, ConstrFactory
from optimResult import OptimResult


class Optimizer(ABC):
    def __init__(self, xStart: np.ndarray, cobra: CobraInitializer, p2: Phase2Vars):
        self.s_opts = cobra.sac_opts
        self.s_res = cobra.sac_res

    @abstractmethod
    def get_result(self) -> OptimResult:
        pass


class NLoptOptimizer(Optimizer):
    """
    Optimizers that are based on package ``nlOpt``
    """
    def __init__(self, xStart: np.ndarray, cobra: CobraInitializer, p2: Phase2Vars):
        super().__init__(xStart, cobra, p2)
        start = time.perf_counter()

        # construct **now** sf_factory with methods subProb2 and gCOBRA, given the current state of cobra and p2
        sf_factory = SeqFuncFactory(cobra, p2)

        # cat("Starting optimizer ...\n");
        switcher = {
            "COBYLA": nlopt.opt(nlopt.LN_COBYLA, xStart.size),
            "ISRESN": nlopt.opt(nlopt.GN_ISRES, xStart.size)
        }
        opt = switcher.get(self.s_opts.SEQ.optimizer, "not implemented")
        assert opt != "not implemented", f"Optimizer {self.s_opts.SEQ.optimizer} is not (yet) implemented"

        lower = self.s_res['lower']
        upper = self.s_res['upper']
        assert (xStart >= lower).all(), "[SeqOptimizer] xStart >= lower violated"
        assert (xStart <= upper).all(), "[SeqOptimizer] xStart <= upper violated"
        opt.set_lower_bounds(lower)
        opt.set_upper_bounds(upper)
        opt.set_min_objective(sf_factory.subProb2)
        tol_i = sf_factory.get_tol_i()
        if tol_i.size > 0:  # this should always be the case (due to dist req constraint)
            opt.add_inequality_mconstraint(sf_factory.g_vec_c, tol_i)
        tol_e = sf_factory.get_tol_e()
        if tol_e.size > 0:
            opt.add_equality_mconstraint(sf_factory.h_vec_c, tol_e)
        opt.set_xtol_rel(self.s_opts.SEQ.tol)
        opt.set_maxeval(self.s_opts.SEQ.feMax)

        # --- debug only: ---
        # fs = cobra.rw.forward(cobra.solu)
        # xStart = fs

        try:
            x = opt.optimize(xStart)
        except nlopt.RoundoffLimited:
            print(f"WARNING: seqOpt [{p2.num}] nlopt.RoundoffLimited exception "
                  f"(result code {opt.last_optimize_result()})")
            x = xStart.copy()

        minf = opt.last_optimum_value()
        # only debug analysis:
        if self.s_opts.SEQ.checkTrueSolu and p2.num > 100:
            x, minf = sf_factory.check_if_solu_better(x, minf)

        self.time_ms = (time.perf_counter() - start) * 1000
        self.opt_res = OptimResult(x, minf,
                                   opt.last_optimize_result(),
                                   opt.get_numevals(),
                                   self.time_ms)

    def get_result(self) -> OptimResult:
        return self.opt_res


class DEoptOptimizer(Optimizer):
    """
    Optimizers that are based on ``scipy``'s differential evolution
    """
    def __init__(self, xStart: np.ndarray, cobra: CobraInitializer, p2: Phase2Vars):
        super().__init__(xStart, cobra, p2)
        start = time.perf_counter()

        # construct **now** sf_factory with methods subProb2 and gCOBRA, given the current state of cobra and p2
        sf_factory = SeqFuncFactory(cobra, p2)
        cf_factory = ConstrFactory(sf_factory)

        # cat("Starting optimizer ...\n");
        # switcher = {
        #     "COBYLA": nlopt.opt(nlopt.LN_COBYLA, xStart.size),
        #     "ISRESN": nlopt.opt(nlopt.GN_ISRES, xStart.size)
        # }
        # opt = switcher.get(self.s_opts.SEQ.optimizer, "not implemented")
        # assert opt != "not implemented", f"Optimizer {self.s_opts.SEQ.optimizer} is not (yet) implemented"

        lower = self.s_res['lower']
        upper = self.s_res['upper']
        bounds = Bounds(lower, upper)
        assert (xStart >= lower).all(), "[SeqOptimizer] xStart >= lower violated"
        assert (xStart <= upper).all(), "[SeqOptimizer] xStart <= upper violated"
        objective_function = sf_factory.subProb2_de

        constraints = []
        for i in range(sf_factory.ine_ind.size):
            if i != 0 or cobra.sac_opts.SEQ.DRC_active:
              constraints +=  [cf_factory.g_constr(i)]
        for j in range(sf_factory.equ_ind.size):
            constraints +=  [cf_factory.h_constr(j)]

        result = differential_evolution(
            objective_function,
            bounds=bounds,
            constraints=tuple(constraints),
            strategy='best2bin',
            x0=xStart,
            seed=42,
            maxiter=self.s_opts.SEQ.feMax,
            tol=self.s_opts.SEQ.tol,
        )

        time_ms = (time.perf_counter() - start) * 1000
        self.opt_res = OptimResult(result.x, result.fun,
                                   result.message,
                                   result.nfev,
                                   time_ms)

    def get_result(self) -> OptimResult:
        return self.opt_res
