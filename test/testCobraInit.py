import unittest
import numpy as np
from cobraInit import CobraInitializer
from cobraPhaseII import CobraPhaseII
from opt.equOptions import EQUoptions
from opt.isaOptions import ISAoptions
from opt.seqOptions import SEQoptions
from rescaleWrapper import RescaleWrapper
from opt.idOptions import IDoptions
from opt.sacOptions import SACoptions
from opt.trOptions import TRoptions

verb = 1


class TestCobraInit(unittest.TestCase):
    """
        Several tests for component :class:`CobraInitializer`
    """
    def test_fn_rescale1(self):
        """  Test rescaling for a specific 1D-function
        """
        def fn(x):
            return np.array([3*np.sum(x ** 2)])
        x0 = np.array([2.5, 2.5])
        lower = np.array([-5, -5])
        upper = np.array([ 5, 5])
        is_equ = np.array([])
        cobra = CobraInitializer(x0, fn, "f_name", lower, upper, is_equ, s_opts=SACoptions(verbose=verb))
        sac_res = cobra.get_sac_res()
        # sac_opts = cobra.get_sac_opts()
        rescaler = RescaleWrapper(fn, lower, upper, sac_res['lower'], sac_res['upper'])
        # these assertions should be valid for all values of fn, x0, lower, upper:
        self.assertEqual(sac_res['lower'][0], -1)
        self.assertEqual(sac_res['upper'][0], 1)
        self.assertEqual(sac_res['fn'](sac_res['x0']), fn(x0))
        self.assertTrue((rescaler.inverse(rescaler.forward(x0)) == x0).all())
        # these assertions are valid only for the specific values above of fn, x0, lower, upper:
        self.assertEqual(sac_res['fn'](sac_res['x0']), 37.5)
        self.assertTrue((rescaler.forward(x0) == np.array([0.5, 0.5])).all())
        print("test_rescale1:\n", fn(x0))

    def test_fn_rescale2(self):
        """  Test rescaling for a specific 2D-function
        """
        def fn(x):
            return np.array([3*np.sum(x ** 2), np.sum(x)-1])
        x0 = np.array([2.5, 2.5])
        lower = np.array([-5, -5])
        upper = np.array([ 5, 5])
        is_equ = np.array([False])
        cobra = CobraInitializer(x0, fn, "f_name", lower, upper, is_equ, s_opts=SACoptions(verbose=verb))
        sac_res = cobra.get_sac_res()
        # sac_opts = cobra.get_sac_opts()
        rescaler = RescaleWrapper(fn, lower, upper, sac_res['lower'], sac_res['upper'])
        # these assertions should be valid for all values of fn, x0, lower, upper:
        self.assertEqual(sac_res['lower'][0], -1)
        self.assertEqual(sac_res['upper'][0], 1)
        self.assertTrue((sac_res['fn'](sac_res['x0']) == fn(x0)).all())
        self.assertTrue((rescaler.inverse(rescaler.forward(x0)) == x0).all())
        # these assertions are valid only for the specific values above of fn, x0, lower, upper:
        self.assertTrue((sac_res['fn'](sac_res['x0']) == np.array([37.5, 4])).all())
        self.assertTrue((rescaler.forward(x0) == np.array([0.5, 0.5])).all())
        print("test_rescale2:\n", fn(x0))
        # arr = np.vstack((x0,x0-2, x0-5))
        # z = np.apply_along_axis(fn,axis=1,arr=arr)
        # print(z.shape)
        # print(z)

    def test_init_design(self):
        """ Test whether ``InitDesigner`` produces arrays ``Fres``, ``Gres`` that are numerically equivalent
            to what we compute from (rescaled) ``A`` and ``fn``.
        """
        def fn(x):
            return np.array([3*np.sum(x ** 2), np.sum(x)-1])
        x0 = np.array([2.5, 2.4])
        lower = np.array([-5, -5])
        upper = np.array([ 5, 5])
        is_equ = np.array([False])
        cobra = CobraInitializer(x0, fn, "f_name", lower, upper, is_equ,
                                 s_opts=SACoptions(verbose=verb, TR=TRoptions(radiInit=0.42)))
        sac_res = cobra.get_sac_res()
        A = sac_res['A']
        newXStart = sac_res['x0']
        Fres = sac_res['Fres']
        Gres = sac_res['Gres']
        newfn = sac_res['fn']
        # these assertions should be valid for all values of fn, x0, lower, upper:
        for i in range(A.shape[0]):
            fnEval = newfn(A[i, :])
            self.assertEqual(Fres[i], fnEval[0])
            for j in range(Gres.shape[1]):
                self.assertEqual(Gres[i, j], fnEval[1+j])
        fnEval = newfn(newXStart)
        self.assertEqual(Fres[-1], fnEval[0])
        for j in range(Gres.shape[1]):
            self.assertEqual(Gres[-1, j], fnEval[1 + j])
        # self.assertEqual(sac_res['upper'][0], 1)
        # self.assertTrue((sac_res['fn'](sac_res['x0'])==fn(x0)).all())
        print("test_init_design:\n", cobra.get_sac_opts().TR.radiInit)
        print(A)
        print(newXStart)

    def test_init_design_R(self):
        """
            Test whether ``InitDesigner`` produces arrays ``Fres``, ``Gres`` that are numerically equivalent to results
            from R (see ``demo-id.R``). Uses reproducible random numbers from RNG ``self.my_rng2`` that avoids cycles.
        """
        def fn(x):
            return np.array([3 * np.sum(x ** 2), np.sum(x) - 1])

        x0 = np.array([2.5, 2.4])
        lower = np.array([-5, -5])
        upper = np.array([5, 5])
        idp = 2*x0.size + 1
        is_equ = np.array([False])
        cobra = CobraInitializer(x0, fn, "f_name", lower, upper, is_equ,
                                 s_opts=SACoptions(verbose=verb,
                                                   ID=IDoptions(initDesign="RAND_R", initDesPoints=idp)))
        sac_res = cobra.get_sac_res()
        A = sac_res['A']
        # newXStart = sac_res['x0']
        Fres = sac_res['Fres']
        Gres = sac_res['Gres']
        # self.assertEqual(sac_res['upper'][0], 1)
        # self.assertTrue((sac_res['fn'](sac_res['x0'])==fn(x0)).all())

        # these are the results computed on the R side (file demo-id.R):
        A_from_R = np.array( [[-0.96472247, -0.7704361],
                              [ 0.16435849,  0.2802904],
                              [-0.09018369,  0.1833372],
                              [ 0.24347296, -0.3384863],
                              [ 0.50000000,  0.4800000]])
        F_from_R = np.array([114.31958851,   7.91823088,   3.13092105,  13.03890450,  36.03000000])
        G_from_R = np.array([[-9.67579269],
                             [ 1.22324437],
                             [-0.53423260],
                             [-1.47506675],
                             [ 3.90000000]])
        self.assertTrue(np.allclose(A, A_from_R), "A and A_from_R are not close")
        self.assertTrue(np.allclose(Fres, F_from_R), "Fres and F_from_R are not close")
        self.assertTrue(np.allclose(Gres, G_from_R), "Gres and G_from_R are not close")
        print("test_init_design_R:\n", A)
        print(Fres)
        # print(Fres - F_from_R)
        # print(Gres - G_from_R)

        # Result: The numbers do all fulfil the np.allclose assertions.
        # The errors are in most cases < 5*10e-7. In one case (Fres[2]), the error is 1e-6.

    def test_adCon_R(self):
        """
            Given a problem ``fn`` with two constraints that trigger ``adCon`` normalization. Test whether adjustment
            of constraints works as expected, i.e. whether results are numerically equivalent to R
            (see ``demo-adCon.R``).

            Results:

            - The numbers in ``Gres`` (after constraint normalization) do all fulfill the ``np.allclose`` assertion.
            - The *relative* errors are in all cases < 5e-7.
            - The two columns of the normalized ``Gres`` have exactly the same min-max-range 1.0.
        """
        def fn(x):
            return np.array([3 * np.sum(x ** 2), np.sum(x) - 1,  -3000*(np.sum(x)-10)])

        x0 = np.array([2.5, 2.4])
        lower = np.array([-5, -5])
        upper = np.array([5, 5])
        idp = 2*x0.size + 1
        is_equ = np.array([False, False])
        cobra = CobraInitializer(x0, fn, "f_name", lower, upper, is_equ,
                                 s_opts=SACoptions(verbose=verb,
                                                   ID=IDoptions(initDesign="RAND_R", initDesPoints=idp)))
        sac_res = cobra.get_sac_res()
        A = sac_res['A']
        # newXStart = sac_res['x0']
        Fres = sac_res['Fres']
        Gres = sac_res['Gres']
        self.assertEqual(sac_res['upper'][0], 1)

        # these are the results computed on the R side (file demo-adCon.R):
        A_from_R = np.array([[ -0.96472247, -0.7704361],
                             [  0.16435849,  0.2802904],
                             [ -0.09018369,  0.1833372],
                             [  0.24347296, -0.3384863],
                             [  0.50000000,  0.4800000]])
        F_from_R = np.array([114.31958851,   7.91823088,   3.13092105,  13.03890450,  36.03000000])
        # this is the new Gres after constraint normalization (both columns have the same max-min-range):
        G_from_R = np.array([[-0.71272396, 1.3756687],
                             [0.09010482, 0.5728399],
                             [-0.03935185, 0.7022966],
                             [-0.10865419, 0.7715989],
                             [0.28727604, 0.3756687]])

        self.assertTrue(np.allclose(A, A_from_R), "A and A_from_R are not close")
        self.assertTrue(np.allclose(Fres, F_from_R), "Fres and F_from_R are not close")
        self.assertTrue(np.allclose(Gres, G_from_R), "Gres and G_from_R are not close")
        # test that all two columns of new Gres have the same max-min-range:
        GRL = np.apply_along_axis(self.minMaxLen, axis=0, arr=Gres)
        self.assertTrue(np.allclose(GRL[0], GRL[1]), "GRL is not the same for the (normalized) constraints")
        print("GRL: ", GRL)
        for i in range(A.shape[0]):
            # test that sac_res['fn'] is appropriately scaled such that each row of A produces the corresponding
            # row of the (new, normalized) Gres:
            x = A[i, :]
            y = np.hstack((Fres[i], Gres[i, :]))
            # print((y - sac_res['fn'](x))/y)
            self.assertTrue(np.allclose(sac_res['fn'](x), y))
        # print(Fres)
        # print(Fres - F_from_R)
        print("rel.err(Gres) = ", np.max((Gres - G_from_R)/Gres))
        print("[test_adCon_R passed]")

    def test_adCon2(self):
        """
            Given a problem ``fn`` with one equality and one inequality constraint that trigger ``adCon`` normalization.
            Test whether adjustment of constraints works as expected when margins are involved, i.e. for equality
            constraints with margin mu (muFinal) or for general margin conTol (tau).

            To this end, we form one cobra structure ``cob_1`` where ``adCon``'s constraint normalization is done and
            another cobra structure ``cob_2`` where it is not (by setting ``ISA.TGR=np.inf``). We grab the matrices
            ``A, Fres, Gres`` from each cobra structure after ``cobraInit`` and test the following:

            Results:

            - Matrices ``A`` and ``Fres`` are the same (of course)
            - Matrix ``GRfact * Gres`` of ``cob1`` is the same as matrix ``Gres`` of `` cob_2``
            - The important test is the following: If we consider infill points ``x = x_solu + delta`` where ``x_solu``
              is the fully feasible solution vector and ``delta`` are different perturbations such that ``x`` is
              sometimes feasible (within the equality constraint band) and sometimes not: Is the condition 'feasible'
              and the max violation always the same in both cobra structures for all ``x``? -- Yes it is, after
              ensuring that all points are inside the search volume [lower, upper] and therefore not clipped.
            -
        """
        def fn(x):
            return np.array([3 * np.sum(x ** 2), 10000*(np.sum(x) - 1),  x[1]-x[0]+10])

        x_solu = np.array([5.0, -5.0])   # with objective 150 and G-values [-1.0, 0.0]

        silent = True
        is_equ = np.array([False, True])
        self.inner_adCon2(fn, is_equ, x_solu, muFinal=1e-7, conTol=0, silent=silent)
        self.inner_adCon2(fn, is_equ, x_solu, muFinal=1e-7, conTol=2.5e-8, silent=silent)
        is_equ = np.array([False, False])
        self.inner_adCon2(fn, is_equ, x_solu, muFinal=1e-7, conTol=0, silent=silent)
        self.inner_adCon2(fn, is_equ, x_solu, muFinal=1e-7, conTol=2.5e-8, silent=silent)
        print("[test_adCon2 passed]")

    def inner_adCon2(self, fn, is_equ, x_solu, muFinal=1e-7, conTol=0.0, silent=False):
        x0 = np.array([2.5, 2.4])
        u = 10                          # upper bound
        lower = np.array([-u, -u])
        upper = np.array([u, u])
        idp = 2*x0.size + 1
        cob_1 = CobraInitializer(x0, fn, "f_name", lower, upper, is_equ,
                                 s_opts=SACoptions(verbose=verb, verboseIter=10, feval=idp+5, cobraSeed=42,
                                                   ID=IDoptions(initDesign="LHS", initDesPoints=idp),
                                                   SEQ=SEQoptions(conTol=conTol),
                                                   EQU=EQUoptions(refine=False, muFinal=muFinal)))
        s_res = cob_1.get_sac_res()
        fn1 = s_res['fn']
        A1 = s_res['A']
        Fres1 = s_res['Fres']
        Gres1 = s_res['Gres']
        GRfact = s_res['GRfact']
        # test that all two columns of new Gres1 have the same max-min-range:
        GRL = np.apply_along_axis(self.minMaxLen, axis=0, arr=Gres1)
        self.assertTrue(np.allclose(GRL[0], GRL[1]), "GRL is not the same for the (normalized) constraints")
        print("GRL: ", GRL)
        self.assertEqual(s_res['upper'][0], 1)

        c1 = CobraPhaseII(cob_1).start()
        p1 = c1.p2

        cob_2 = CobraInitializer(x0, fn, "f_name", lower, upper, is_equ,
                                 s_opts=SACoptions(verbose=verb, verboseIter=10, feval=idp+5, cobraSeed=42,
                                                   ID=IDoptions(initDesign="LHS", initDesPoints=idp),
                                                   ISA=ISAoptions(TGR=np.inf),
                                                   SEQ=SEQoptions(conTol=conTol),
                                                   EQU=EQUoptions(refine=False, muFinal=muFinal)))
        s_res = cob_2.get_sac_res()
        fn2 = s_res['fn']
        A2 = s_res['A']
        Fres2 = s_res['Fres']
        Gres2 = s_res['Gres']

        c2 = CobraPhaseII(cob_2).start()
        p2 = c2.p2

        self.assertTrue(np.allclose(A1, A2), "A1 and A2 are not close")
        self.assertTrue(np.allclose(Fres1, Fres2), "Fres1 and Fres2 are not close")
        self.assertTrue(np.allclose(GRfact * Gres1, Gres2), "Gres1 * GRfact and Gres2 are not close")
        # array broadcasting makes "GRfact * Gres1" work: if GRfact.shape = (3,) and Gres1.shape = (5,3), then
        # array broadcasting will make (3,) --> (1, 3) and then this one row of GRfact is replicated 5 times. This is
        # exactly what we want: Each element Gres1[i,j] is multiplicated with 'its' column-j GRfact.

        x_perp = np.array([-1.0,1.0])     # a vector perpendicular to the equality constraint line (5,-5) + r * (1,1)
        for fac in 1e-6*np.arange(-0.1, 0.1, 0.005):
            xNew = (x_solu + fac*x_perp) / u      # ' / u' : make the rescale trafo 'by hand'
            f1x = fn1(xNew)
            f2x = fn2(xNew)
            self.assertTrue(np.allclose(f1x * np.append(1,GRfact), f2x), "f1x * [1,GRfact] and f2x are not close")
            p1.ev1.update(xNew, cob_1, p1, p1.currentMu)
            p2.ev1.update(xNew, cob_2, p2, p2.currentMu)
            # self.assertEqual(p1.currentMu, p2.currentMu)
            if not silent:
                print(f"{fac:.3e}: {xNew} {p1.ev1.trueMaxViol:.9e}, {p2.ev1.trueMaxViol:.9e}, {p2.ev1.trueNumViol==0}, f2x[2] = {f2x[2]:.9e}")
            self.assertTrue(np.allclose(p1.ev1.trueMaxViol, p2.ev1.trueMaxViol))
            self.assertEqual(p1.ev1.trueNumViol, p2.ev1.trueNumViol)
            # the following assertions assume that the second constraint f2x[2] is responsible for feasibility or not,
            # as it is the case in our toy problem fn.
            if is_equ[1] == True:
                self.assertEqual(np.abs(f2x[2]) <= muFinal+conTol, p1.ev1.trueNumViol == 0)
            else:
                self.assertEqual(f2x[2] <= conTol, p1.ev1.trueNumViol == 0)

        print(f"[inner_adCon2 with is_equ = {is_equ}, muFinal = {muFinal:.1e}, conTol = {conTol:.1e} passed]")

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

    def minMaxLen(self, x):
        maxL = max(x)
        minL = min(x)
        return maxL - minL


if __name__ == '__main__':
    unittest.main()
