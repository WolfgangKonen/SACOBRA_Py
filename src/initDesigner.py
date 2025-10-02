from typing import Union

import nlopt
import numpy as np

from innerFuncs import distLine
from opt.sacOptions import SACoptions
from opt.idOptions import IDoptions         # needed for docstring
from fnArchiveFact import FnArchiveFactory
# import lhsmdu
from scipy.stats.qmc import LatinHypercube


class FnNloptFactory:
    """
    Factory for the functions that are optimized in OPTIMIZED, OPTCOBYLA or OPTBIASED initial runs
    """
    def __init__(self, x0: np.ndarray, fn_arch: FnArchiveFactory, is_equ: np.ndarray, tol: float):
        self.fn_arch = fn_arch
        # self.fn_x0 = fn_arch(x0)
        self.is_equ = is_equ
        self.n_constraints = is_equ.size
        # equ_ind is an index to the equality constraints in gCOBRA:
        self.equ_ind = np.flatnonzero(self.is_equ)
        # ine_ind is an index to the inequality constraints in gCOBRA:
        self.ine_ind = np.flatnonzero(self.is_equ == False)
        # DON'T change here to 'self.is_equ is False' as the PEP hint suggest --> strange error in NLopt (!)
        self.tol_e = np.repeat(tol, self.equ_ind.size)      # we need tol_e and tol_i
        self.tol_i = np.repeat(tol, self.ine_ind.size)      # just to transport the sizes

    def subProb2(self, x, grad):
        """
        surrogate evaluation of 'f' for constrained optimization methods
        """
        return self.fn_arch(x)[0]

    def g_vec_c(self, result, x, grad):
        """ Vector-valued **inequality** constraints for nlopt

            Note the special signature with ``result`` which has to be a vector of size self.ine_ind.size
        """
        g = self._gCOBRA(x, grad)
        result[:] = g[self.ine_ind]

    def h_vec_c(self, result, x, grad):
        """ Vector-valued **equality** constraints for nlopt

            Note the special signature with ``result`` which has to be a vector of size self.equ_ind.size
        """
        h = self._gCOBRA(x, grad)
        result[:] = h[self.equ_ind]

    def _gCOBRA(self, x, grad) -> np.array:
        """
        surrogate evaluation of '\vec{g}' for constrained optimization methods
        """
        if self.n_constraints > 0:
            # if np.allclose(x, self.fn_arch.soluArchive[-1, :]):
            if np.all(np.equal(x, self.fn_arch.soluArchive[-1, :])):
                # if there was a preceding call to fn_arch with this x, re-use the last row of funcArchive
                constraintPrediction = self.fn_arch.funcArchive[-1, 1:]
                DBG = False
                if DBG:
                    # this assertion will double the lines in self.fn_arch.funcArchive, but it checks correctness:
                    assert np.allclose(constraintPrediction, self.fn_arch(x)[1:])
            else:
                constraintPrediction = self.fn_arch(x)[1:]

            h = constraintPrediction
            assert constraintPrediction.size == self.n_constraints
        else:
            h = None
        return h

    def get_tol_e(self):
        """ Tolerance vector for equality constraints, length = # eq. constr.
        :return: tolerance vector
        """
        return self.tol_e

    def get_tol_i(self):
        """ Tolerance vector for inequality constraints, length = # problem ineq. constr.
        :return: tolerance vector
        """
        return self.tol_i


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

    In detail, the constructor of ``InitDesigner`` does the following: Create the **initial design** in matrix
    **self.A** with shape ``(P, d)`` of sample points in (potentially rescaled) input space
    ``[lower, upper]`` :math:`\\subset \\mathbb{R}^d`, where ``P = s_opts.ID.initDesPoints`` and ``d =`` input space dimension.
    The recipe how to select the sample points is prescribed by the :ref:`type of initial design <initDesign-label>`
    ``s_opts.ID.initDesign``.

    Apply ``fn`` to these points and split the result in objective function (:math:`f`) values **self.Fres**
    with shape ``(P,)`` and constraint function (:math:`g,h`) values **self.Gres** with shape
    ``(P,nC)``, where ``nC`` = number of constraints.

    :param x0:  the last point ``self.A[-1,:]`` is ``x0`` (potentially rescaled)
    :param fn:  see parameter ``fn`` in :class:`cobraInit.CobraInitializer`
    :param rng: RNG (random number generator) from :class:`cobraInit.CobraInitializer`
    :param lower: vector of shape ``(d,)``
    :param upper: vector of shape ``(d,)``
    :param s_opts: the options. Here we use ``s_opts.cobraSeed`` and from element  :class:`.IDoptions` ``s_opts.ID``
                   the elements ``initDesign`` and ``initDesPoints``.
    :type s_opts: :class:`SACoptions`
    """

    def __init__(self, x0: np.ndarray, fn, rng,
                 lower: np.ndarray, upper: np.ndarray, is_equ: np.ndarray, s_opts: SACoptions):
        self.val = s_opts.cobraSeed
        d = lower.size
        npts = s_opts.ID.initDesPoints
        if s_opts.ID.initDesign == "RANDOM":
            # Create self.A with shape (npts,d) where the first npts-1 points in R^d are uniform random from
            # [lower, upper].
            self.A = rng.random(size=(npts-1, d))      # uniform random in [0,1)

        elif s_opts.ID.initDesign == "RAND_R":   # deprecated, better use "RAND_REP"
            # -- DEPRECATED, use RAND_REP --
            # Same as "RANDOM", but with reproducible random numbers (reproducible also on the R side).
            # The seed is s_opts.cobraSeed.
            self.A = self._my_rng(npts - 1, d, s_opts.cobraSeed)  # uniform random in [0,1)

        elif s_opts.ID.initDesign == "RAND_REP":
            # Same as "RANDOM", but with reproducible random numbers (reproducible also on the R side).
            # Uses self.my_rng2 for better random numbers than in "RAND_R" (avoid cycles!).
            # The seed is s_opts.cobraSeed (set via initial value for self.val).
            self.A = self._my_rng2(npts - 1, d)  # uniform random in [0,1)
        elif s_opts.ID.initDesign == "LHS":
            n = npts -1
            # Latin Hypercube Sampling via SciPy
            engine = LatinHypercube(d=d, rng=s_opts.cobraSeed)
            sam = engine.random(n=n)
            self.A = np.array(sam)   # shape=(n,d), uniform random in [0,1)

            # # Latin Hypercube Sampling via lhsmdu --- now deprecated
            # sam = lhsmdu.sample(d, n, randomSeed=s_opts.cobraSeed)
            # A_old = np.array(sam).T   # shape=(npts-1,d), uniform random in [0,1)

        elif s_opts.ID.initDesign == "OPTCOBYLA":
            self.fnArchiveF = FnArchiveFactory(fn, x0)
            A_for_rbf = np.zeros((0,x0.size))  # dummy
            feval = int(npts * 1.3)
            while A_for_rbf.shape[0] < npts:
                self._cobyla_run(x0, self.fnArchiveF, feval, lower, upper, is_equ, s_opts)
                self.A = self.fnArchiveF.getSoluArchive()
                A_for_rbf = self._build_A_for_rbf()    # reduce self.A to non-identical rows
                # only diagnostics:
                assert self.A.shape[0] == feval, f"A.shape[0] = {self.A.shape[0]} and feval = {feval} differ!"
                print(f"A.shape = {self.A.shape}, {A_for_rbf.shape} = A_for_rbf.shape")
                # setup for possible next pass through while-loop:
                feval = int(feval * 1.1)
                self.fnArchiveF = FnArchiveFactory(fn, x0)
            self.A = A_for_rbf[0:npts, :]

        elif s_opts.ID.initDesign == "BIASED":
            # Create self.A with shape (npts,d) where the first npts-1 points in R^d are normal random from
            # N(x0, initBias).
            sd = np.repeat(s_opts.ID.initBias, x0.size)
            self.A = rng.normal(x0, sd, size=(npts - 1, d))  # normal distributed around x0
            self._clip_lower_upper_A(lower, upper)

        else:
            raise RuntimeError(f"[InitDesigner] Invalid value s_opts.initDesign = '{s_opts.ID.initDesign}' ")

        # TODO: other initial designs ("OPTIMIZED", "OPTBIASED", ...)
        # (Note that the MOPTA runs from 2016-2018 used initDesign = "OPTIMIZED")

        zero_one_distributed = ["RANDOM", "RAND_R", "RAND_REP", "LHS"]
        if s_opts.ID.initDesign in zero_one_distributed:
            # rescale to [lower, upper]:
            self.A = self.A @ np.diag(upper - lower) + np.tile(lower, (npts - 1, 1))
        if s_opts.ID.initDesign != "OPTCOBYLA":
            # each initial design != "OPTCOBYLA" gets x0 added as the last point (which is already rescaled to
            # [lower,upper]). For "OPTCOBYLA we do not add x0, because it is already the first point
            self.A = np.vstack((self.A, x0))

        # Apply fn to all points (rows) in matrix self.A. The points are the rows of matrix self.A (axis=1).
        fnEval = np.apply_along_axis(fn, axis=1, arr=self.A)    # fnEval.shape = (initDesPoints, nConstraints+1)
        self.Fres = fnEval[:, 0]
        self.Gres = fnEval[:, 1:]

    def __call__(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Return the three results ``A``, ``Fres`` and ``Gres`` of the initial design

        :return: tuple (**self.A**, **self.Fres**, **self.Gres**)

        :rtype: (np.ndarray, np.ndarray, np.ndarray)

        The return tuple contains:

        - **self.A**: ``(P,d)``-matrix with initial design
        - **self.Fres**: ``(P,)``-vector with objective evaluated at each initial design point
        - **self.Gres**: ``(P,nC)``-matrix with constraints evaluated at each initial design point

        """
        return self.A, self.Fres, self.Gres

    def _cobyla_run(self, x0, fn_arch, feval, lower, upper, is_equ, s_opts: SACoptions):
        tol = s_opts.SEQ.tol
        fnNloptFact = FnNloptFactory(x0, fn_arch, is_equ, tol)
        optimizer = "COBYLA"
        switcher = {
            "COBYLA": nlopt.opt(nlopt.LN_COBYLA, x0.size),
            "ISRESN": nlopt.opt(nlopt.GN_ISRES, x0.size)
        }
        opt = switcher.get(optimizer, "not implemented")
        assert opt != "not implemented", f"Optimizer {optimizer} is not (yet) implemented"

        assert (x0 >= lower).all(), "[InitDesigner.cobyla_run] x0 >= lower violated"
        assert (x0 <= upper).all(), "[InitDesigner.cobyla_run] x0 <= upper violated"
        opt.set_lower_bounds(lower)
        opt.set_upper_bounds(upper)
        opt.set_min_objective(fnNloptFact.subProb2)
        tol_i = fnNloptFact.get_tol_i()
        if tol_i.size > 0:  # this should always be the case (due to dist req constraint)
            opt.add_inequality_mconstraint(fnNloptFact.g_vec_c, tol_i)
        tol_e = fnNloptFact.get_tol_e()
        if tol_e.size > 0:
            opt.add_equality_mconstraint(fnNloptFact.h_vec_c, tol_e)
        opt.set_xtol_rel(tol)
        opt.set_maxeval(feval)

        try:
            x = opt.optimize(x0)
        except nlopt.RoundoffLimited:
            print(f"WARNING: seqOpt [{0}] nlopt.RoundoffLimited exception "
                  f"(result code {opt.last_optimize_result()})")
            x = x0.copy()

        minf = opt.last_optimum_value()
        feMax = opt.get_numevals()
        return minf, feMax

    def _build_A_for_rbf(self):
        """
        :return: a matrix ``A_for`` which is a copy of ``self.A``, but identical rows are removed
        """
        A_for = self.A[0:2,:].copy()
        for k in range(2,self.A.shape[0]):
            xNew = self.A[k,:]
            xNewDist = distLine(xNew, A_for)
            if min(xNewDist) > 1e-9:  # 0.0:     # a value 1e-9 is needed by G04 to avoid LinAlgError
                A_for = np.vstack((A_for, xNew))
        return A_for

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

    def _clip_lower_upper_A(self, lower, upper):
        self.A = np.apply_along_axis(lambda x: np.maximum(x, lower), axis=1, arr=self.A)
        self.A = np.apply_along_axis(lambda x: np.minimum(x, upper), axis=1, arr=self.A)
