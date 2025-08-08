import time
import unittest
import numpy as np
from cobraInit import CobraInitializer
from gCOP import GCOP
from cobraPhaseII import CobraPhaseII
from opt.isaOptions import ISAoptions, O_LOGIC
from opt.sacOptions import SACoptions
from opt.idOptions import IDoptions
from opt.rbfOptions import RBFoptions
from opt.seqOptions import SEQoptions

verb = 1


class TestCOP(unittest.TestCase):
    """
        Several tests for the more ambitious COP G06 from the G function benchmark.

        - G06 is a COP with two circular inequality constraints that form a very small and narrow feasible region.
    """
    def test_G06_R(self):
        """ Test whether COP G06 has numerical equivalent results to the R side with ``squares=False``, if we choose on
            the Python side {``RBF.model="cubic"``, ``RBF.degree=1``, ``cobraSeed=42`` and ``ID.initDesignPoints=5``}
            and use ``solve_G06_R`` in ``ex_COP.R`` on the R side.

            Both sides have larger errors (1e-3...1e-2) in this case ``squares=False``, but the RBF models are exactly
            the same.

            What we test:

            - that ``A, Fres, Gres`` after initial design are exactly the same in R and Python
            - that ``fp1`` and ``gp1``, certain surrogate values when starting ``cobraPhaseII``, are exactly the
              same in R and Python
            - that ``df.XI`` contains (after the initial ``np.nan``'s) tiles that are exact copies of
              ``cobra.sac_opts.XI``
            - that two ways to calculate ``pEffect`` on the Python side produce the same results
            - that the iterations with restart and their ``x0`` are exactly equal in R and Python
              (``cobra.sac_res['rs']``)
            - that the final errors are similar (abs.tol < 1e-2) in R and Python although the iteration paths differ

            What we NOT test (or find NOT confirmed):

            - that the iteration path is the same in R and Python
            - that ``A, Fres, Gres`` are the same in SACOBRA phase II (they differ because iteration path differs)
            - (as a consequence) that the final error is exactly the same

            We test also the branch **onlinePLOG=O_LOGIC.XNEW**: It seems that the switch ``pEff_DBG=True`` gives
            better results. This switch uses *NOT* ``xNew`` in ``calcPEffect`` but the midpoint between ``A[0,:]`` and
            ``A[1,:]`` (more stable behavior in pEffect). What we test and find:

            - that the first ten values of ``pEffect`` are the same (or at least similar) as in R.
              (In case ``pEff_DBG=False`` this holds only for the first three or seven values.)
            - that the final error is similar in Python and in R (not the same, but similar with ``atol=1e-2``). The
              combination (``onlinePLOG=O_LOGIC.XNEW``, ``pEff_DBG=True``) has the smallest final error 4e-3. The
              combination (``onlinePLOG=O_LOGIC.XNEW``, ``pEff_DBG=False``) has the largest final error 2e-1.
        """
        for onlinePLOG in list(O_LOGIC):   #
            for pEff_DBG in [True]:  #False,
                TestCOP.inner_test_G06_R(onlinePLOG, pEff_DBG)

    def test_G06_R_no_onlinePLOG(self):
        TestCOP.inner_test_G06_R(O_LOGIC.NONE, False)

    def test_G06_R_new_pEffect_midp(self):
        TestCOP.inner_test_G06_R(O_LOGIC.MIDPTS, False)

    @staticmethod
    def inner_test_G06_R(opl, pdb):
        G06 = GCOP("G06")

        cobra = CobraInitializer(G06.x0, G06.fn, G06.name, G06.lower, G06.upper, G06.is_equ,
                                 s_opts=SACoptions(verbose=verb, feval=40, cobraSeed=42,
                                                   ID=IDoptions(initDesign="RAND_REP", initDesPoints=5),
                                                   ISA=ISAoptions(onlinePLOG=opl, onlineFreqPLOG=1,
                                                                  pEff_DBG=pdb),  # isa_ver=2, , pEffectLogic=pel
                                                   RBF=RBFoptions(degree=1),
                                                   SEQ=SEQoptions(finalEpsXiZero=True, conTol=0)))  # 1e-7
        # cobra.sac_opts.ISA.RS = False  # temp
        cobra.sac_opts.ISA.RS_rep = True

        # all ..._from_R values are obtained with solve_G06_R in ex_COP.R
        A_from_R = np.array([[-0.77165545,  0.50386505],
                             [-0.96567336,  0.06279872],
                             [-0.07241825,  0.10488204],
                             [ 0.70542654, -0.28823662],
                             [-0.83678161, -0.88320000]])
        F_from_R = np.array([170298.137,  36486.825, 125241.549, 463638.138,  -1808.858])
        G_from_R = np.array([[ -5148.6848, 5131.0088],
                             [ -2307.5744, 2306.7780],
                             [ -4762.1736, 4683.6639],
                             [ -7590.1836, 7444.0015],
                             [  -128.7156,  116.7056]])
        assert np.allclose(A_from_R, cobra.sac_res['A'])
        assert np.allclose(F_from_R, cobra.sac_res['Fres'])
        assert np.allclose(G_from_R, cobra.sac_res['Gres'])

        c2 = CobraPhaseII(cobra)
        c2.start()

        # fp1: fitness surrogate at 'xbest + 1' in cobraPhaseII.py, first while loop pass
        # gp1: constraint surrogates at 'xbest + 1'
        fp1_from_R = 203058.2 if cobra.sac_opts.ISA.onlinePLOG is not O_LOGIC.NONE else 13.10047615
        gp1_from_R = np.array([-5797.53931915, 5698.52931915])
        assert np.allclose(c2.p2.fp1, fp1_from_R)
        assert np.allclose(c2.p2.gp1, gp1_from_R)
        idp = cobra.sac_opts.ID.initDesPoints
        tsz = cobra.sac_opts.XI.size  # tile size
        for k in range(100):  # after the first idp np.nan, cobra.df.XI should contain tiles of the global DRC
            up = idp+tsz*(k+1)
            if up > cobra.df.XI.size: break     # break if tile would exceed the rim of df.XI
            assert np.allclose(cobra.sac_opts.XI, cobra.df.XI[up-tsz:up])
        pEffect1 = np.array(cobra.df2['pEffect'])[0:10]
        pEffect2 = np.array([float(np.log10(np.nanmedian(c2.p2.errRatio[0:i]))) for i in range(1, 10+1)])
        assert np.allclose(pEffect1, pEffect2)
        if cobra.sac_opts.ISA.onlinePLOG is not O_LOGIC.MIDPTS:
            if cobra.sac_opts.ISA.onlinePLOG == O_LOGIC.XNEW:
                if cobra.sac_opts.ISA.pEff_DBG:
                    pEffect_from_R = np.array([-0.4797520, -0.6504113, -0.9350294, -0.8891745, -0.8487078,
                                               -0.8482750, -0.8478426, -0.8477724, -0.8477022, -0.8475222])
                    assert np.allclose(pEffect1, pEffect_from_R)
                else:
                    pEffect_from_R = np.array([2.4071584, 2.1072531, 0.2994246, 0.2584721, 0.2132520,
                                               0.2110046, 0.2132520, 0.2110046, 0.2087455, 0.2110046])
                    # in this case no assertion holds, because the points xNew for which pEffect is calculated differ a lot
                    # on the iteration path, so that pEffect1 and pEffect_from_R differ up to 4e-1. But the first three
                    # values are very similar (differ by less than 1e-6):
                    assert np.allclose(pEffect1[0:3], pEffect_from_R[0:3], atol=1e-6)
            else:   # i.e. O_LOGIC.NONE
                if cobra.sac_opts.ISA.pEff_DBG:
                    pEffect_from_R = np.array([-0.4797520, -0.5796741, -0.7096800, -0.7607614, -0.8186618,
                                               -0.8208656, -0.8230806, -0.8555640, -0.8906749, -0.8926831])
                    assert np.allclose(pEffect1, pEffect_from_R, atol=4e-3)
                else:
                    pEffect_from_R = np.array([-0.12739167, 1.00969922, 0.40261829, 0.21389634, 0.24275199,
                                                0.09597503,-0.12739167,-0.14793669,-0.12739167,-0.14793669])
                    # in this case no assertion holds, because the points xNew for which pEffect is calculated differ a lot
                    # on the iteration path, so that pEffect1 and pEffect_from_R differ up to 4e-2. But the first seven
                    # values are very similar (differ by less than 1e-6):
                    assert np.allclose(pEffect1[0:7], pEffect_from_R[0:7], atol=1e-6)
            p_err = np.abs(pEffect1 - pEffect_from_R)

        rs_from_R = np.array([[ 5, 0.50387, -0.96567],
                              [16, 0.48229, -0.13682],
                              [17, 0.44305, -0.17194],
                              [18,-0.08535,  0.94597],
                              [19,-0.23118,  0.36581],
                              [24,-0.87773,  0.53504],
                              [32,-0.64901,  0.91134],])
        rs = cobra.sac_res['rs'].reshape(-1,3)
        assert np.allclose(rs,rs_from_R,rtol=1e-4)
        # it happens that rs is exactly the same for ISA.RStype=RSTYPE.SIGMOID and RStype=RSTYPE.CONSTANT. But this is just a
        # 'lucky' coincidence of this cobraSeed = 42 that produces very small anewrand in iter 5, 16, 17, ... which
        # trigger a restart for both ISA.RStype values (although the p_restart values in randomStarter.py differ).
        # For other values of cobraSeed, results in rs differ for different ISA.RStype, as they should.

        # print(cobra.df)
        # print(cobra.df2)
        final_err = np.array(cobra.df.fBest - G06.fn(G06.solu)[0])[-1]
        print(f"final err: {final_err}")
        if cobra.sac_opts.ISA.onlinePLOG == O_LOGIC.XNEW:
            if cobra.sac_opts.ISA.pEff_DBG:
                final_err_from_R = 3.617988e-03
            else:
                final_err_from_R = 2.320659e-01
        elif cobra.sac_opts.ISA.onlinePLOG == O_LOGIC.MIDPTS:
            final_err_from_R = 3.617988e-03
        else:    # i.e. O_LOGIC.NONE
            final_err_from_R = 1.391763e-02
        print(f"final_err_from_R={final_err_from_R} (onlinePLOG={opl.name}, pEff_DBG={pdb}, degree=1, seed=42)")
        # the iteration path is different in R and Python --> final_err is only approximately the same:
        assert np.allclose(final_err, final_err_from_R, atol=1e-2)
        print(f"[test_G06_R] all assertions passed (onlinePLOG={opl.name}, pEff_DBG={pdb}, degree=1, seed=42)")
        dummy = 0

    def test_G06(self):
        """ Test whether COP G06 has statistical equivalent results to the R side with ``squares=True``, if we set
            on the Python side ``RBF.degree=2`` (which is similar, but not the same).

            We test that the median of 15 final errors is < 5e-6, which is statistically equivalent to the R side
            (see ``ex_COP.R``, function ``multi_G06``)

            ``ID.initDesignPoints=6`` is required for ``RBF.degree=2``.

            First, we had a bug: Depending on ``cobraSeed``, ``RBFInterpolator`` would often produce
            ``numpy.linalg.LinAlgError`` "Singular Matrix". We traced this back to same rows appearing in matrix ``A``.
            Bug fix: The new dict ``cobra.for_rbf`` with elements ``['A', 'Fres', 'Gres']``, which contains only
            non-identical rows.
        """
        G06 = GCOP("G06")

        start = time.perf_counter()
        fin_err_list = np.array([])
        runs = 15
        for run in range(runs):
            cobra = CobraInitializer(G06.x0, G06.fn, G06.name, G06.lower, G06.upper, G06.is_equ,
                                     s_opts=SACoptions(verbose=verb, feval=40, cobraSeed=39+run,
                                                       ID=IDoptions(initDesign="RAND_REP", initDesPoints=6),
                                                       RBF=RBFoptions(degree=2),
                                                       SEQ=SEQoptions(finalEpsXiZero=True, conTol=1e-7)))

            c2 = CobraPhaseII(cobra)
            c2.start()

            # print(cobra.df)
            # print(cobra.df2)
            fin_err = np.array(cobra.df.fBest - G06.fn(G06.solu)[0])[-1]
            fin_err_list = np.concatenate((fin_err_list, fin_err), axis=None)
            print(f"final err: {fin_err}")
            dummy = 0

        print(np.array(sorted(fin_err_list), dtype=float))   # to get rid of 'np.float64(...)'
        med_fin_err = np.median(fin_err_list)
        print(f"min: {np.min(fin_err_list)},  max: {np.max(fin_err_list)}")
        # thresh = 5e-6       # this is for s_opts.finalEpsXiZero=False
        thresh = 5e-8       # this is for s_opts.finalEpsXiZero=True and s_opts.SEQ.conTol=1e-7
        assert med_fin_err <= thresh, f"median(fin_err_list) = {med_fin_err} > {thresh}, which is too large"
        print(f"[test_G06] median(final error) = {med_fin_err} < {thresh}: OK")
        print(f"[test_G06] ... finished ({(time.perf_counter() - start)/runs*1000:.4f} msec per run, {runs} runs)")
        print( "[test_G06] all assertions passed (degree=2, seed=39+run)")


if __name__ == '__main__':
    unittest.main()
