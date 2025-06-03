from typing import Union
import numpy as np
import lhsmdu
from opt.sacOptions import SACoptions
from opt.idOptions import IDoptions         # needed for docstring


class InitDesigner:
    """
    The **initial design** is a set of ``P`` points from input space with dimension ``d``. The problem functions
    :math:`f,g,h` are evaluated at these ``P`` points and these evaluated sets form the basis of the later optimization:
    From the characteristics of the evaluated sets, the classes :class:`.CobraInitializer` and :class:`.CobraPhaseII`
    deduce decisions about certain adjustments:

    - whether to adjust constraint functions or not, see :meth:`.CobraInitializer.adCon`,
    - which :ref:`DRC <DRC-label>` to select, see :meth:`.CobraInitializer.adDRC`.
    - whether to apply :math:`plog(f)` or not, see :ref:`AdFitter <AdFitter-label>` (called in each iteration in :class:`.CobraPhaseII`),

    :class:`.CobraPhaseII` uses the evaluated sets to train the initial fitness and constraint surrogate models.

    In detail: Create matrix ``self.A`` with shape ``(P, d)`` of sample points in (potentially rescaled) input space
    ``[lower, upper]`` :math:`\\subset \mathbb{R}^d`, where ``P = initDesPoints`` and ``d =`` input space dimension.

    Apply ``fn`` to these points and split the result in objective function (:math:`f`) values ``self.Fres``
    with shape ``(P,)`` and constraint function (:math:`g,h`) values ``self.Gres`` with shape
    ``(P,nC)``, where ``nC`` = number of constraints.

    :param x0:  the last point ``self.A[-1,:]`` is ``x0``
    :param fn:  see parameter ``fn`` in :class:`cobraInit.CobraInitializer`
    :param lower: vector of shape ``(d,)``
    :param upper: vector of shape ``(d,)``
    :param s_opts: object of class :class:`SACoptions`. Here we use from element  :class:`.IDoptions` ``s_opts.ID``
                   the elements ``initDesign`` and ``initDesPoints``.
    :type s_opts: SACoptions
    """

    def __init__(self, x0: np.ndarray, fn, rng,
                 lower: np.ndarray, upper: np.ndarray, s_opts: SACoptions):
        self.val = s_opts.cobraSeed
        d = lower.size
        npts = s_opts.ID.initDesPoints
        if s_opts.ID.initDesign == "RANDOM":
            # Create self.A with shape (npts,d) where the first npts-1 points in R^d are uniform random from
            # [lower, upper] and the last point is x0.
            self.A = rng.random(size=(npts-1, d))      # uniform random in [0,1)
        elif s_opts.ID.initDesign == "RAND_R":
            # Same as "RANDOM", but with reproducible random numbers (reproducible also on the R side).
            # The seed is s_opts.cobraSeed.
            self.A = self._my_rng(npts - 1, d, s_opts.cobraSeed)  # uniform random in [0,1)
        elif s_opts.ID.initDesign == "RAND_REP":
            # Same as "RAND_R", but with better self.my_rng2 (avoid cycles!).
            # The seed is s_opts.cobraSeed (set via initial value for self.val).
            self.A = self._my_rng2(npts - 1, d)  # uniform random in [0,1)
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

    def __call__(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Return the three results ``A``, ``Fres`` and ``Gres`` of the initial design

        :return: (self.A, self.Fres, self.Gres)
        :rtype: (np.ndarray, np.ndarray, np.ndarray)
        """
        return self.A, self.Fres, self.Gres

    def _my_rng(self, n, d, seed):
        MOD = 10 ** 5 + 7
        val = seed
        x = np.zeros((n, d), dtype=np.float32)
        for n_ in range(n):
            for d_ in range(d):
                val = (val * val) % MOD
                x[n_, d_] = val / MOD   # map val to range [0,1[
        return x

    def _my_rng2(self, n, d):
        MOD = 10 ** 5 + 7
        OFS = 10 ** 5 - 7
        x = np.zeros((n, d), dtype=np.float32)
        for n_ in range(n):
            for d_ in range(d):
                self.val = (self.val*self.val*np.sqrt(self.val)+OFS) % MOD    # avoid cycles (!)
                x[n_, d_] = self.val / MOD   # map val to range [0,1[
        return x
