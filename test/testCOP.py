import time
import unittest
import numpy as np
import nlopt
from cobraInit import CobraInitializer
from gCOP import GCOP
from cobraPhaseII import CobraPhaseII
from innerFuncs import verboseprint, plog, plogReverse
from rescaleWrapper import RescaleWrapper
from trainSurrogates import trainSurrogates, calcPEffect
from opt.sacOptions import SACoptions
from opt.idOptions import IDoptions
from opt.rbfOptions import RBFoptions
from opt.seqOptions import SEQoptions

verb = 1


class TestCOP(unittest.TestCase):
    """
        Several tests for component :class:`CobraPhaseII`
    """
    def test_G06_R(self):
        """ Test whether COP G06 has numerical equivalent results to the R side with squares=F, if we choose on the
            Python side {RBF.model="cubic", RBF.degree=1, cobraSeed=42 and ID.initDesignPoints=5}.

            Both sides have larger errors in this (squares=F)-case, but the RBF models are exactly the same.

            What we test:

            - that A, Fres, Gres after initial design are exactly equal in R and Python
            - that df.XI contains after the initial np.nan's tiles that are exact copies of cobra.sac_opts.XI
            - that two ways to calculate pEffect produce the same result
            - that the iterations with restart and their xStart are exactly equal in R and Python ('rs')
        """
        G06 = GCOP("G06")

        cobra = CobraInitializer(G06.xStart, G06.fn, G06.name, G06.lower, G06.upper,
                                 s_opts=SACoptions(verbose=verb, feval=40, cobraSeed=42,
                                                   ID=IDoptions(initDesign="RAND_REP", initDesPoints=5),
                                                   RBF=RBFoptions(degree=1),
                                                   SEQ=SEQoptions()))       # trueFuncForSurrogates=True
        cobra.sac_opts.ISA.RS_rep = True
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

        fp1_from_R = 13.10047615
        gp1_from_R = np.array([-5797.53931915, 5698.52931915])
        assert np.allclose(c2.p2.fp1, fp1_from_R)
        assert np.allclose(c2.p2.gp1, gp1_from_R)
        idp = cobra.sac_opts.ID.initDesPoints
        for k in range(4):  # after the first idp np.nan, cobra.df.XI should contain tiles of the global DRC
            assert np.allclose(cobra.sac_opts.XI, cobra.df.XI[idp+2*k:idp+2*(k+1)])
        pEffect1 = np.array(cobra.df2['pEffect'])[0:10]
        pEffect2 = np.array([float(np.log10(np.nanmedian(c2.p2.errRatio[0:i]))) for i in range(1,10+1)])
        assert np.allclose(pEffect1, pEffect2)
        rs_from_R = np.array([[ 5, 0.50387, -0.96567],
                              [16, 0.48229, -0.13682],
                              [17, 0.44305, -0.17194],
                              [18,-0.08535,  0.94597],
                              [19,-0.23118,  0.36581],
                              [24,-0.87773,  0.53504],
                              [32,-0.64901,  0.91134],])
        rs = cobra.sac_res['rs'].reshape(-1,3)
        assert np.allclose(rs,rs_from_R,rtol=1e-4)
        # it happens that rs is exactly the same for RStype="SIGMOID" and RStype="CONSTANT". But this is just a
        # 'lucky' coincidence of this cobraSeed = 42 that produces very small anewrand in iter 5, 16, 17, ... which
        # trigger a restart for both p_restart values (although they differ). For other cobraSeed, results differ, as
        # they should.
        print("[test_G06_R] all assertions passed (degree=1, seed=42)")

        #print(cobra.df)
        #print(cobra.df2)
        print(f"final err: {np.array(cobra.df.Best - G06.fn(G06.solu)[0])[-1]}")
        dummy = 0

    def test_G06(self):
        """ Test whether COP G06 has statistical equivalent results to the R side with squares=T, if we set on the
            Python side RBF.degree=2 (which is similar, but not the same).

            We test that the median of 15 final errors is < 1e-6, which is statistically equivalent to the R side
            (see ex_COP.R, function multi_G06)

            ID.initDesignPoints=6 is required for RBF.degree=2.

            First, we had a bug: Depending on cobraSeed, RBFInterpolator would ofter produce
            numpy.linalg.LinAlgError "Singular Matrix". We traced this back to same rows appearing in matrix A.
            Bug fix: The new dict cobra.for_rbf with elements ['A', 'Fres', 'Gres'], which contains only non-identical
            rows.
        """
        G06 = GCOP("G06")

        start = time.perf_counter()
        fin_err_list = np.array([])
        runs = 15
        for run in range(runs):
            cobra = CobraInitializer(G06.xStart, G06.fn, G06.name, G06.lower, G06.upper,
                                     s_opts=SACoptions(verbose=verb, feval=40, cobraSeed=39+run,
                                                       ID=IDoptions(initDesign="RAND_REP", initDesPoints=6),
                                                       RBF=RBFoptions(degree=2),
                                                       SEQ=SEQoptions()))       # trueFuncForSurrogates=True

            c2 = CobraPhaseII(cobra)
            c2.start()

            #print(cobra.df)
            #print(cobra.df2)
            fin_err = np.array(cobra.df.Best - G06.fn(G06.solu)[0])[-1]
            fin_err_list = np.concatenate((fin_err_list, fin_err), axis=None)
            print(f"final err: {fin_err}")
            dummy = 0

        print(fin_err_list)
        med_fin_err = np.median(fin_err_list)
        print(f"min: {np.min(fin_err_list)},  max: {np.max(fin_err_list)}")
        print(f"median: {med_fin_err}")
        thresh = 1e-6
        assert med_fin_err <= thresh, f"median(fin_err_list) = {med_fin_err} > {thresh}, which is too large"
        print(f"[test_G06] ... finished ({(time.perf_counter() - start)/runs*1000:.4f} msec per run, {runs} runs)")


if __name__ == '__main__':
    unittest.main()
