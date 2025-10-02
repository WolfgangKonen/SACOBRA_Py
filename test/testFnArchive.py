import unittest
import numpy as np

from cobraInit import CobraInitializer
from cobraPhaseII import CobraPhaseII
from fnArchiveFact import FnArchiveFactory
from opt.idOptions import IDoptions
from opt.sacOptions import SACoptions

verb = 1


class TestFnArchive(unittest.TestCase):
    """
        Several tests for FnArchive, OPTCOBYLA, ...
    """
    def test_fn_archive(self):
        """  Test FnArchive
        """
        def fn(x):
            return x ** 2 + 3

        x0 = np.zeros(2)
        fnArchiveF = FnArchiveFactory(fn, x0)

        z_0 = np.array([3, 2])
        z_1 = np.array([4, 3])
        fnArchiveF(z_0)
        fnArchiveF(z_1)

        A = fnArchiveF.getSoluArchive()
        R = fnArchiveF.getFuncArchive()
        print(A)
        print(R)
        assert A.shape[1] == z_0.size
        assert A.shape[0] == 2
        assert np.allclose(z_0, A[0,:])
        assert np.allclose(z_1, A[1,:])

    def test_optcobyla(self):
        """
        Test whether initial design OPTCOBYLA works as expected. To this end we perform SACOBRA with
        ``ID.initDesign="OPTCOBYLA"`` on a simple COP (objective = sphere, one constraint 1 - sum(x) <= 0, either
        as inequality or as equality constraint). The optimum is in both cases (0.5, 0.5) which corresponds to (0.1, 0.1)
        in rescaled space.

        If the initial design is done with ``ID.initDesPoints=20``, then we see often in the initial design ``A`` a best
        feasible point close to (0.1, 0.1). That is, OPTCOBYLA works often as desired. But we cannot
        place an assertion on this (even not ``np.allclose`` with large ``atol``) because some initial designs produce
        no feasible point at all or only a feasible point far away from (0.1, 0.1). Probably this is due to the still
        small number of 20 iterations.

        But what we can assert is that the **median** of all max_delta values is less than 0.05.
        """
        def fn(x):
            return np.array([ 3*np.sum(x ** 2), -(np.sum(x)-1) ])

        x0 = np.array([2.5, 2.4])
        dim = x0.size
        idp = 20 # (dim + 1) * (dim +2) // 2
        lower = np.array([-5, -5])
        upper = np.array([ 5, 5])
        runs = 10
        max_delta = np.zeros(2*runs)
        for r in range(runs):
            seed = 42 + r
            np.random.seed(seed)
            x0 = np.random.random_sample(dim)*(upper-lower)+lower
            is_equ = np.array([True])
            max_delta[2*r] = self.inner_optcobyla(x0, fn, lower, upper, is_equ, idp, seed)
            is_equ = np.array([False])
            max_delta[2*r+1] = self.inner_optcobyla(x0, fn, lower, upper, is_equ, idp, seed)
        print(np.sort(max_delta))
        print(np.median(max_delta))
        assert np.median(max_delta) < 0.05

    def inner_optcobyla(self, x0, fn, lower, upper, is_equ, idp, seed=42):
        """
        :return: ``np.max(np.abs(xi-xb))``, where ``xi`` is the best feasible solution after initial design and
                ``xb = cobra.get_xbest_cobra()`` after phase II
        """
        cobra = CobraInitializer(x0, fn, "f_name", lower, upper, is_equ,
                                 s_opts=SACoptions(verbose=verb, cobraSeed=seed,
                                                   ID=IDoptions(initDesign="OPTCOBYLA", initDesPoints=idp)))
        sac_res = cobra.get_sac_res()
        A = sac_res['A']
        # A_for = cobra.for_rbf['A']
        # print(f"A.shape = {A.shape}, {A_for.shape} = cobra.for_rbf['A'].shape")
        # print(f"A.shape = {A.shape}")
        assert A.shape[0] == idp
        xi = self.check_Fres_Gres(sac_res)
        # print(A)
        c2 = CobraPhaseII(cobra).start()
        cobra = c2.get_cobra()
        xb = cobra.get_xbest_cobra()
        assert cobra.phase == "phase2"
        max_d = np.max(np.abs(xi-xb))   # max delta between xi and xb
        print(xi)
        print(f"{xb}, {max_d:.2e}")
        # assert np.allclose(xi, xb, atol=1e-1)
        print(f"{cobra.get_fbest()}, is_equ = {is_equ}")
        return max_d

    def check_Fres_Gres(self, sac_res):
        """
        Check that ``Fres``, ``Gres`` match with what ``sac_res['fn']`` returns when fed with the appropriate rows
        of ``A``.

        :param sac_res: we extract from this dict the elements ``['A'], ['Fres'], ['Gres']``
        :return: the best solution within the points of ``A``: If feasible solutions exist, then the feasible solution
            with minimum objective ``Fres``. If no feasible solution exists, then select that point (row of ``A``) that
            has minimum maxViol.
        """
        A = sac_res['A']
        Fres = sac_res['Fres']
        Gres = sac_res['Gres']
        newfn = sac_res['fn']
        # these assertions should be valid for all values of fn, x0, lower, upper:
        for i in range(A.shape[0]):
            fnEval = newfn(A[i, :])
            self.assertEqual(Fres[i], fnEval[0])
            for j in range(Gres.shape[1]):
                self.assertEqual(Gres[i, j], fnEval[1+j])
        for j in range(Gres.shape[1]):
            self.assertEqual(Gres[-1, j], fnEval[1 + j])

        if Gres.shape[1] > 0:       # constrained problem
            maxViol = np.apply_along_axis(np.max, axis=1, arr=Gres)   # maximum constraint violation in each row
        else:                       # unconstrained problem
            maxViol = np.repeat(-1, Gres.shape[0])  # --> each point is feasible
        if min(maxViol) <= 0:       # we have feasible points
            # return index ibest of feasible point with minimum Fres:
            cond = (Fres == min(Fres[maxViol <= 0]))
            ibest = np.flatnonzero(cond)[0]
        else:                       # no feasible points --> return index of point with minimum maxViol:
            ibest = np.flatnonzero(maxViol == min(maxViol))[0]
        return A[ibest, :]

    def test_biased(self):
        """
        Test whether initial design BIASED works as expected. To this end we perform SACOBRA with
        ``ID.initDesign="OPTCOBYLA"`` on a simple COP (objective = sphere, one constraint 1 - sum(x) <= 0, either
        as inequality or as equality constraint). The optimum is in both cases (0.5, 0.5) which corresponds to (0.1, 0.1)
        in rescaled space.

        The test is just that BIASED runs through, no special assertions.
        """
        def fn(x):
            return np.array([ 3*np.sum(x ** 2), -(np.sum(x)-1) ])

        x0 = np.array([2.5, 2.4])
        dim = x0.size
        idp = 20 # (dim + 1) * (dim +2) // 2
        lower = np.array([-5, -5])
        upper = np.array([ 5, 5])
        runs = 5
        max_delta = np.zeros(2*runs)
        for r in range(runs):
            seed = 42 + r
            np.random.seed(seed)
            x0 = np.random.random_sample(dim)*(upper-lower)+lower
            is_equ = np.array([True])
            max_delta[2*r] = self.inner_biased(x0, fn, lower, upper, is_equ, idp, seed)
            is_equ = np.array([False])
            max_delta[2*r+1] = self.inner_biased(x0, fn, lower, upper, is_equ, idp, seed)
        print(np.sort(max_delta))
        print(np.median(max_delta))

    def inner_biased(self, x0, fn, lower, upper, is_equ, idp, seed=42):
        """
        :return: ``np.max(np.abs(xi-xb))``, where ``xi`` is the best feasible solution after initial design and
                ``xb = cobra.get_xbest_cobra()`` after phase II
        """
        cobra = CobraInitializer(x0, fn, "f_name", lower, upper, is_equ,
                                 s_opts=SACoptions(verbose=verb, cobraSeed=seed,
                                                   ID=IDoptions(initDesign="BIASED", initDesPoints=idp)))
        sac_res = cobra.get_sac_res()
        A = sac_res['A']
        # A_for = cobra.for_rbf['A']
        # print(f"A.shape = {A.shape}, {A_for.shape} = cobra.for_rbf['A'].shape")
        # print(f"A.shape = {A.shape}")
        assert A.shape[0] == idp
        xi = self.check_Fres_Gres(sac_res)
        # print(A)
        c2 = CobraPhaseII(cobra).start()
        cobra = c2.get_cobra()
        xb = cobra.get_xbest_cobra()
        assert cobra.phase == "phase2"
        max_d = np.max(np.abs(xi-xb))   # max delta between xi and xb
        print(xi)
        print(f"{xb}, {max_d:.2e}")
        # assert np.allclose(xi, xb, atol=1e-1)
        print(f"{cobra.get_fbest()}, is_equ = {is_equ}")
        return max_d

    def test_unconstrained(self):
        """
        Test whether all initial designs work as expected for an unconstrained problem.

        The test is just that the process runs through, no special assertions.
        """
        def fn(x):
            return np.array([ 3*np.sum(x ** 2)])

        x0 = np.array([2.5, 2.4])
        dim = x0.size
        idp = 20 # (dim + 1) * (dim +2) // 2
        lower = np.array([-5, -5])
        upper = np.array([ 5, 5])
        idesigns = ["RAND_REP", "LHS", "BIASED", "OPTCOBYLA"]
        runs = len(idesigns)
        max_delta = np.zeros(runs)
        for r, ides in enumerate(idesigns):
            print(ides)
            seed = 42
            np.random.seed(seed)
            x0 = np.random.random_sample(dim)*(upper-lower)+lower
            is_equ = np.array([])
            max_delta[r] = self.inner_unconstrained(x0, fn, lower, upper, ides, idp, seed)
        print(np.sort(max_delta))
        print(np.median(max_delta))

    def inner_unconstrained(self, x0, fn, lower, upper, ides, idp, seed=42):
        """
        :return: ``np.max(np.abs(xi-xb))``, where ``xi`` is the best feasible solution after initial design and
                ``xb = cobra.get_xbest_cobra()`` after phase II
        """
        cobra = CobraInitializer(x0, fn, "f_name", lower, upper, np.array([]),
                                 s_opts=SACoptions(verbose=verb, cobraSeed=seed,
                                                   ID=IDoptions(initDesign=ides, initDesPoints=idp)))
        sac_res = cobra.get_sac_res()
        A = sac_res['A']
        # A_for = cobra.for_rbf['A']
        # print(f"A.shape = {A.shape}, {A_for.shape} = cobra.for_rbf['A'].shape")
        # print(f"A.shape = {A.shape}")
        assert A.shape[0] == idp
        xi = self.check_Fres_Gres(sac_res)
        # print(A)
        c2 = CobraPhaseII(cobra).start()
        cobra = c2.get_cobra()
        xb = cobra.get_xbest_cobra()
        assert cobra.phase == "phase2"
        max_d = np.max(np.abs(xi-xb))   # max delta between xi and xb
        print(xi)
        print(f"{xb}, {max_d:.2e}")
        # assert np.allclose(xi, xb, atol=1e-1)
        print(f"{cobra.get_fbest()}")
        return max_d

    def test_phaseII(self):
        def fn(x):
            return np.array([3 * np.sum(x ** 2), np.sum(x) - 1,  -3000*(np.sum(x)-10)])
        x0 = np.array([2.5, 2.4])
        lower = np.array([-5, -5])
        upper = np.array([5, 5])
        is_equ = np.array([False, False])
        cobra = CobraInitializer(x0, fn, "f_name", lower, upper, is_equ,
                                 s_opts=SACoptions(verbose=verb, ID=IDoptions(initDesign="RAND_R")))
        print("\ntest_phaseII:")
        assert cobra.phase == "init"
        print(cobra.sac_opts.ISA.TGR)
        c2 = CobraPhaseII(cobra)
        cobra = c2.get_cobra()
        assert cobra.phase == "phase2"
        print(cobra.sac_opts.ISA.TGR)


if __name__ == '__main__':
    unittest.main()
