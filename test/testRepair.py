import time
import unittest
import numpy as np

from cobraInit import CobraInitializer
from cobraPhaseII import CobraPhaseII
from opt.idOptions import IDoptions
from opt.riOptions import RIoptions
from opt.sacOptions import SACoptions
from phase2Vars import Phase2Vars
from repairInfeasRI2 import RI2
from surrogator import Surrogator

verb = 1


class TestRepair(unittest.TestCase):
    """
        Several tests for repair infeasible (class ``RI2 ri2``)
    """


    def test_eps_feas(self):
        """
        Test ``ri2.is_epsilon_feasible``: Given a COP with two linear constraints, the intersection
        of the two constraint lines is the point ``x_active`` where both constraints are active. If we move from
        ``x_active`` to the left, we move into the feasible region. If both constraint values are ``-eps2`` or below
        then we have ``eps2``-feasibility where ``eps2=RI.eps2``. If we move to the right, we have infeasibility.

        We test the following: We form the points ``x=x_active+[delta,0]`` for different values of ``delta``.

        - If all constraint values for a given ``x`` are ``-eps2`` or below (which coincides for the specific
          constraints with ``delta <= -eps2``) then we have ``eps2``-feasibility. We check that for the coded ``delta``
          and ``eps2`` the is_eps_feas-pattern is [True, True, False, False, False, False].
        - We test that this holds whether we use true functions for surrogates or the surrogate models. The models are
          trained from 20 initial design points.
        - We test that this holds whether we do rescaling in cobraInit or not.
        - We test that all four cases produce the same vector ``con_s`` for a given ``delta`` in vector ``deltas``.
        - As a side test we check that ``fn(x) = cobra.fn(x_rescaled)`` for all ``x`` and all values of ``ID.rescale``.
          (If ``ID.rescale==False`` then ``fn=cobra.fn`` and ``x=x_rescaled`` so that the check is self-understood. But
          for ``ID.rescale==True`` the check is a real test and will only succeed if ``x_rescaled=rw.forward(x)`` is not
          subject to clipping.)
        """

        def fn(x):
            """  A simple COP with sphere objective and two linear constraints """
            return np.array([3 * np.sum(x ** 2), 10 * (np.sum(x) - 1), -(x[1] - x[0] + 10)])
        x_active = np.array([5.5, -4.5])
        # x_active is the point where both constraints of fn are active (intersection of constraint lines).
        # Be sure that x_active is within search space [lower,upper] and not directly at the border (!)

        ieq1 = False   # constraint 1: equality (ieq1=True) or inequality (ieq1=False) constraint
        self.first_pass = True
        for rsc in [False, True]:
            for tfs in [False, True]:
                print(f"\n*** ieq={ieq1},  rescale={rsc}, trueFunc={tfs} ***")
                self.inner_is_eps_feas(fn, x_active, rsc, tfs, ieq1)
        print("[test_eps_feas] All assertions passed.")

    def inner_is_eps_feas(self, fn, x_active, rsc, tfs, ieq1):
        x0 = np.array([2.5, 2.4])
        dim = x_active.size
        idp = 20   # (dim + 1) * (dim +2) // 2
        idesign = "OPTCOBYLA"     # "OPTCOBYLA" | "OPTBIASED"
        lower = np.array([-10, -10])    # be sure that all points x below fall into search space [lower, upper],
        upper = np.array([+10, 10])     # otherwise con_s values can differ across cases (!)
        is_equ = np.array([False, ieq1])

        ID = IDoptions(initDesign=idesign, initDesPoints=idp, rescale=rsc)
        RI = RIoptions(eps2=2e-4, trueFuncForSurrogates=tfs)
        cobra = CobraInitializer(x0, fn, "f_name", lower, upper, is_equ,
                                 s_opts=SACoptions(verbose=verb, cobraSeed=42,
                                                   ID=ID, RI=RI))
        cobra.sac_res['muVec'][-1] = 8e-2   # currentMu
        p2 = Phase2Vars(cobra)
        p2 = Surrogator.trainSurrogates(cobra, p2)

        ri2 = RI2(cobra)
        constrSurr = p2.constraintSurrogates
        deltas=[-0.1, -0.01, -1e-4, 0, 1e-4, 0.01]
        is_feas = np.zeros(len(deltas), dtype=bool)
        if ieq1 == False:        # the inequality constraint case
            tr_feas = np.array([True, True, False, False, False, False])  # condition: delta <= -RI.eps2
        else:                   # the one-equality constraint, currentMu = 0.08
            tr_feas = np.array([False, True, True, False, False, False])  # condition: delta <= -RI.eps2
        if self.first_pass:
            self.con_s = np.zeros((len(deltas), cobra.sac_res['nConstraints']), dtype=float)
        act_con_s = np.zeros((len(deltas), cobra.sac_res['nConstraints']), dtype=float)
        for i, delta in enumerate(deltas):
            x = x_active + np.array([delta, 0.0])       # apply delta always to non-rescaled x_active ...
            if ID.rescale:                              #
                fn1 = fn(x)                             #
                x = ri2.rw.forward(x)                   # ... then do optional rescale
                fn2 = ri2.fn(x)
                assert np.allclose(fn1, fn2)
            is_feas[i] = ri2.is_epsilon_feasible(x, RI.eps2, constrSurr)
            act_con_s[i, :] = ri2.con_s
            np.set_printoptions(precision=5)
            print(f"delta={delta:.0e}, x={x}, {is_feas[i]}, con_s={ri2.con_s}")
        assert all(np.equal(is_feas, tr_feas))
        if self.first_pass:
            self.con_s = act_con_s.copy()
            self.first_pass = False
        else:
            assert np.allclose(self.con_s, act_con_s, atol=1e-6)


    def test_find_best(self):
        """
        Test ``ri2.find_best_feasible``: Given a COP with two axis-parallel linear constraints, the intersection
        of the two constraint lines is the point ``x_active`` where both constraints are active.
        Form a matrix ``deltaMat`` such that ``x=x_active+deltaMat`` are random points around ``x_active``.
        Select with ``ri2.find_best_feasible`` the best feasible point ``x_best``.

        We test the following:

        If ``x_best`` is ``eps2``-feasible:

        - ``x_best`` is always a point where both components of ``deltaMat`` are negative
        - If we calculate ``ri2.is_eps_feasible(xbest)`` then the side-computation ``ri2.con_s`` has in all components
          a value ``-ri2.eps2`` or lower (as a consequence of ``eps2``-feasibility)
        - All other points ``x`` have a higher maximum violation than ``xbest``

        Else (if ``x_best`` is not ``eps2``-feasible):

        - No other point ``x`` has a lower number ``num_nef`` of **not** ``eps2``-feasible constraints
        - All other points ``x`` which share with ``xbest`` the same ``num_nef`` have a higher maximum violation

        We test the above for different random seeds and for all four cases ``rescale=True/False`` and
        ``RI.trueFuncForSurrogates=True/False``.

        We test that all four cases lead to the same ``num_nef`` pattern (across the data points ``x``).

        We can enforce ``x_best`` to be non-``eps2``-feasible by making the vector ``q`` (controlling the size of the
        components of ``deltaMat``) sufficiently small (e.g. factor 5e-4 when ``eps2=1e-2``). Because then all
        generated random vectors will be too small to land in the ``eps2``-feasible sector.
        """
        def fn(x):
            """  A very simple COP with sphere objective and two axis-parallel constraints """
            return np.array([3 * np.sum(x ** 2), 2 * (x[0] - 1), (x[1] - 3)])
        x_active = np.array([1, 3])
        # x_active is the point where both constraints of fn are active (intersection of constraint lines).
        # Be sure that x_active is within search space [lower,upper] and not directly at the border (!)

        seed = 42
        self.first_pass = True
        for rsc in [False, True]:    #
            for tfs in [False, True]:   #
                print(f"\n*** rescale={rsc}, trueFunc={tfs} ***")
                np.random.seed(seed)        # every call of inner_fbf with same seed, to make rsc=True/False comparable
                self.inner_fbf(fn, x_active, rsc, tfs, seed)
        print(np.int32(self.nef_pattern))
        print("[test_find_best_feas] All assertions passed.")

    def inner_fbf(self, fn, x_active, rsc, tfs, seed=42):
        x0 = np.array([2.5, 2.4])
        dim = x_active.size
        idp = 20   # (dim + 1) * (dim +2) // 2
        idesign = "OPTCOBYLA"     # "OPTCOBYLA" | "OPTBIASED"
        lower = np.array([-10, -10])    # be sure that all points x below fall into search space [lower, upper],
        upper = np.array([+10, 10])     # otherwise con_s values can differ across cases (!)
        is_equ = np.array([False, False])

        ID = IDoptions(initDesign=idesign, initDesPoints=idp, rescale=rsc)
        RI = RIoptions(eps2=2e-4, trueFuncForSurrogates=tfs)
        cobra = CobraInitializer(x0, fn, "f_name", lower, upper, is_equ,
                                 s_opts=SACoptions(verbose=verb, cobraSeed=seed,
                                                   ID=ID, RI=RI))
        cobra.sac_res['muVec'][-1] = 1e-4   # currentMu
        p2 = Phase2Vars(cobra)
        p2 = Surrogator.trainSurrogates(cobra, p2)

        ri2 = RI2(cobra)
        constrSurr = p2.constraintSurrogates
        n_delta = 100
        q = np.array([0.2, 0.5]) * 5e-3
        if ID.rescale:
            q = ri2.rw.forward(q)
        deltaMat = (2 * np.random.random_sample((n_delta,2)) - 1) * q
        # NOTE: delta is in rescaled space if ID.rescale (as needed by ri2.find_best_feasible)

        is_feas = np.zeros(n_delta, dtype=bool)
        num_nef = np.zeros(n_delta, dtype=float)
        act_con_s = np.zeros((n_delta, cobra.sac_res['nConstraints']), dtype=float)

        if ID.rescale:
            x_active = ri2.rw.forward(x_active)
        # NOTE: x_active is in rescaled space if ID.rescale (as needed by ri2.find_best_feasible)

        x_best = x_active + ri2.find_best_infeasible(x_active, deltaMat, RI.eps2, constrSurr)
        print(x_best - x_active)
        is_f_best = ri2.is_epsilon_feasible(x_best, RI.eps2, constrSurr)
        max_v_best = np.max(ri2.con_s)
        num_f_best = np.flatnonzero(ri2.con_s + RI.eps2 > 0).size   # number of not eps2-feasible constraints in x_best
        for i in range(n_delta):
            x = x_active + deltaMat[i,:]
            is_feas[i] = ri2.is_epsilon_feasible(x, RI.eps2, constrSurr)
            act_con_s[i, :] = ri2.con_s
            num_nef[i] = np.flatnonzero(act_con_s[i, :] + RI.eps2 > 0).size
            np.set_printoptions(precision=5)
        max_v = np.max(act_con_s, axis=1)
        if is_f_best:
            print("x_best is eps2-feasible")
            assert np.all(x_best - x_active < 0.0), "not all components of x_best-x_active are negative!"
            ri2.is_epsilon_feasible(x_best, RI.eps2, constrSurr)
            assert np.all(ri2.con_s <= RI.eps2), f"x_best is not eps2-feasible (eps2={RI.eps2})"
            assert np.all(max_v >= max_v_best), f"another x has a lower max violation than x_best"
        else:
            print("x_best is NOT eps2-feasible")
            assert np.all(num_nef >= num_f_best), f"another x has fewer not eps2-feasible constraints"
            ind = np.flatnonzero(num_nef == num_f_best)
            assert np.all(max_v[ind] >= max_v_best), f"another x with same num_nef has a lower max viol than x_best"
        if self.first_pass:
            self.nef_pattern = num_nef.copy()
            self.first_pass = False
        else:
            assert np.allclose(self.nef_pattern, num_nef, atol=1e-6)

    def test_repair(self):
        """
        Test ``ri2.repairInfeasRI2``: Given a COP with two linear inequality constraints, the intersection
        of the two constraint lines is the point ``x_active`` where both constraints are active. If we move from
        ``x_active`` to the right, we move into the infeasible region.

        We form the infeasible points ``x=x_active+[delta,0]`` for different values
        ``delta=[0.01,0.1,1.0]`` and call ``z=ri2.repairInfeasRI2(x,...)``. We test the following:

        - Is the numerical gradient calculated from the constraint surrogates close to the true gradient (we pass in
          the latter via parameter ``true_grad``)?
        - For all ``eps``-infeasible constraints ``k``: Is the single repair step ``k`` (row ``k`` of matrix
          ``del_mat``) such that it transports an ``eps``-infeasible solution to a new location that is very close
          to the ``eps``-feasible border in the single constraint ``k``?
        - Assert that ``z`` is ``eps2``-feasible in all constraints.

        The first two assertions are done in ``ri2.check_single_constr`` called from ``ri2.repairInfeasRI2`` if
        ``checkIt=True``. The last assertion is done in ``inner_repair``.
        """

        def fn(x):
            """  A simple COP with sphere objective and two linear constraints """
            return np.array([3 * np.sum(x ** 2), 10 * (np.sum(x) - 1), -(x[1] - x[0] + 10)])

        x_active = np.array([5.5, -4.5])
        # x_active is the point where both constraints of fn are active (intersection of constraint lines).
        # Be sure that x_active is within search space [lower,upper] and not directly at the border (!)
        orig_grad = np.array([[10., 10.],
                              [ 1., -1.]])
        # orig_grad[0] is the true gradient of g[0], orig_grad[1] of g[1] (not rescaled)

        for rsc in [False, True]:    #
            for tfs in [False]:   # , True
                print(f"\n*** rescale={rsc}, trueFunc={tfs} ***")
                self.inner_repair(fn, x_active, rsc, tfs, orig_grad)
        print("[test_repair] All assertions passed.")

    def inner_repair(self, fn, x_active, rsc, tfs, orig_grad):
        x0 = np.array([2.5, 2.4])
        dim = x_active.size
        idp = 20  # (dim + 1) * (dim +2) // 2
        idesign = "OPTCOBYLA"  # "OPTCOBYLA" | "OPTBIASED"
        lower = np.array([-10, -10])  # be sure that all points x below fall into search space [lower, upper],
        upper = np.array([+10, +10])  # otherwise con_s values can differ across cases (!)
        is_equ = np.repeat(False, dim)
        currentMu = 8e-2

        ID = IDoptions(initDesign=idesign, initDesPoints=idp, rescale=rsc)
        RI = RIoptions(eps2=2e-4, mmax=1000, trueFuncForSurrogates=tfs)
        cobra = CobraInitializer(x0, fn, "f_name", lower, upper, is_equ,
                                 s_opts=SACoptions(verbose=verb, cobraSeed=43,
                                                   ID=ID, RI=RI))
        p2 = Phase2Vars(cobra)
        p2 = Surrogator.trainSurrogates(cobra, p2)

        ri2 = RI2(cobra)
        constrSurr = p2.constraintSurrogates
        deltas = [1.0]      # 0.01, 0.1,

        for i, delta in enumerate(deltas):
            true_grad = orig_grad.copy()
            x = x_active + np.array([delta, 0.0])       # apply delta always to non-rescaled x_active ...
            if ID.rescale:                              #
                x_orig = x.copy()                       #
                x = ri2.rw.forward(x)                   # ... then do optional rescale
                true_grad = (x_orig / x) * true_grad    # If we rescale from [-10,10] to [-1,1] in every x-dim
                # then every x-distance shrinks by factor 10 --> gradient increases by factor 10.
                # NOTE: pointwise multiply takes into account that different dims may be rescaled differently.
            gReal = cobra.sac_res['fn'](x)[1:]
            z = ri2.repairInfeasRI2(x, gReal, constrSurr, cobra, currentMu, True, true_grad)
            assert ri2.is_epsilon_feasible(z, RI.eps2, constrSurr), f"z={z} is not eps2-feasible!"
            print(f"delta={delta:.0e}, x={x}, z={z}")

    def test_time_repair(self):
        """
        Measure time of repair (avg. from 100 runs) [R times from demo-repair.R]:

        =========== ============ ==========
         mmax        Python         R
        =========== ============ ==========
         1000        10 ms           60 ms
         5000        40 ms          629 ms
         10000       116 ms        1740 ms
        =========== ============ ==========
        """
        def fn(x):
            """  A simple COP with sphere objective and two linear constraints """
            return np.array([3 * np.sum(x ** 2), 10 * (np.sum(x) - 1), -(x[1] - x[0] + 10)])

        x_active = np.array([5.5, -4.5])
        # x_active is the point where both constraints of fn are active (intersection of constraint lines).
        # Be sure that x_active is within search space [lower,upper] and not directly at the border (!)

        x0 = np.array([2.5, 2.4])
        dim = x_active.size
        idp = 20  # (dim + 1) * (dim +2) // 2
        idesign = "OPTCOBYLA"  # "OPTCOBYLA" | "OPTBIASED"
        lower = np.array([-10, -10])  # be sure that all points x below fall into search space [lower, upper],
        upper = np.array([+10, 10])  # otherwise con_s values can differ across cases (!)
        is_equ = np.repeat(False, dim)
        currentMu = 8e-2

        ID = IDoptions(initDesign=idesign, initDesPoints=idp, rescale=True)
        RI = RIoptions(eps2=2e-4, mmax=1000, trueFuncForSurrogates=False)
        cobra = CobraInitializer(x0, fn, "f_name", lower, upper, is_equ,
                                 s_opts=SACoptions(verbose=verb, cobraSeed=43,
                                                   ID=ID, RI=RI))
        p2 = Phase2Vars(cobra)
        p2 = Surrogator.trainSurrogates(cobra, p2)

        ri2 = RI2(cobra)
        constrSurr = p2.constraintSurrogates
        delta = 1.0      # 0.01, 0.1,
        runs = 100

        start = time.perf_counter()
        for k in range(runs):    #
            x = x_active + np.array([delta, 0.0])       # apply delta always to non-rescaled x_active ...
            if ID.rescale:                              #
                x = ri2.rw.forward(x)                   # ... then do optional rescale
            gReal = cobra.sac_res['fn'](x)[1:]
            z = ri2.repairInfeasRI2(x, gReal, constrSurr, cobra, currentMu, False)
        time_ms = (time.perf_counter() - start) / runs * 1000
        print(f"[test_time repair] time per repair = {time_ms} ms")


if __name__ == '__main__':
    unittest.main()
