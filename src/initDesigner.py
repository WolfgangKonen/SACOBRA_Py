from typing import Union
import numpy as np
import lhsmdu
from opt.sacOptions import SACoptions
from opt.idOptions import IDoptions         # needed for docstring


class InitDesigner:
    def __init__(self, x0: np.ndarray, fn,
                 lower: np.ndarray, upper: np.ndarray, s_opts: SACoptions):
        """
        Create matrix ``self.A`` with shape ``(P, d)`` of sample points in input space ``[lower, upper]``
        :math:`\subset \mathbb{R}^d`, where ``P = initDesPoints`` and ``d =`` input space dimension.

        Apply ``fn`` to these points and divide the result in objective
        function values ``self.Fres`` (shape ``(P,)``) and constraint function values ``self.Gres`` (shape
        ``(P,nC)`` with ``nC`` = number of constraints).

        :param x0:  the last point ``self.A[-1,:]`` is ``x0``
        :param fn:
        :param lower:
        :param upper:
        :param s_opts: object of class :class:`SACoptions`. Here we use from element ID of class :class:`IDoptions`
                       the elements ``initDesign`` and ``initDesPoints``.
        """
        self.val = s_opts.cobraSeed
        d = lower.size
        npts = s_opts.ID.initDesPoints
        if s_opts.ID.initDesign == "RANDOM":
            # Create self.A with shape (npts,d) where the first npts-1 points in R^d are uniform random from
            # [lower, upper] and the last point is x0.
            self.A = np.random.rand(npts-1, d)      # uniform random in [0,1)
        elif s_opts.ID.initDesign == "RAND_R":
            # Same as "RANDOM", but with reproducible random numbers (reproducible also on the R side).
            # The seed is s_opts.cobraSeed.
            self.A = self.my_rng(npts - 1, d, s_opts.cobraSeed)  # uniform random in [0,1)
        elif s_opts.ID.initDesign == "RAND_REP":
            # Same as "RAND_R", but with better self.my_rng2 (avoid cycles!).
            # The seed is s_opts.cobraSeed (set via initial value for self.val).
            self.A = self.my_rng2(npts - 1, d)  # uniform random in [0,1)
        elif s_opts.ID.initDesign == "LHS":
            # Latin Hypercube Sampling.
            # The seed is s_opts.cobraSeed.
            sam = lhsmdu.sample(d, npts - 1, randomSeed=s_opts.cobraSeed)
            self.A = np.array(sam).T   # shape=(npts-1,d), uniform random in [0,1)
        else:
            raise RuntimeError(f"[InitDesigner] Invalid value s_opts.initDesign = '{s_opts.ID.initDesign}' ")

        self.A = self.A @ np.diag(upper - lower) + np.tile(lower, (npts - 1, 1))
        self.A = np.vstack((self.A, x0))

        # TODO: other initial designs ("BIASED", "OPTIMIZED", "OPTBIASED", ...)

        # Apply fn to all points (rows) in matrix self.A. A point is a row in matrix self.A (axis=1).
        fnEval = np.apply_along_axis(fn, axis=1, arr=self.A)    # fnEval.shape = (initDesPoints, nConstraints+1)
        self.Fres = fnEval[:, 0]
        self.Gres = fnEval[:, 1:]

    def __call__(self, *args, **kwargs) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self.A, self.Fres, self.Gres

    def my_rng(self, n, d, seed):
        MOD = 10 ** 5 + 7
        val = seed
        x = np.zeros((n, d), dtype=np.float32)
        for n_ in range(n):
            for d_ in range(d):
                val = (val * val) % MOD
                x[n_, d_] = val / MOD   # map val to range [0,1[
        return x

    def my_rng2(self, n, d):
        MOD = 10 ** 5 + 7
        OFS = 10 ** 5 - 7
        x = np.zeros((n, d), dtype=np.float32)
        for n_ in range(n):
            for d_ in range(d):
                self.val = (self.val*self.val*np.sqrt(self.val)+OFS) % MOD    # avoid cycles (!)
                x[n_, d_] = self.val / MOD   # map val to range [0,1[
        return x
