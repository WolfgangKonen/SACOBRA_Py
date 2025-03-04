import unittest
import numpy as np
import nlopt
from cobraInit import CobraInitializer
from phase2Vars import Phase2Vars
from seqOptimizer import SeqFuncFactory
from cobraPhaseII import CobraPhaseII
from innerFuncs import verboseprint, plog, plogReverse
from rescaleWrapper import RescaleWrapper
from trainSurrogates import trainSurrogates, calcPEffect
from opt.sacOptions import SACoptions
from opt.idOptions import IDoptions
from opt.isaOptions import ISAoptions
from opt.seqOptions import SEQoptions

verb = 1

def fn(x):
    return np.array([3 * np.sum(x ** 2), np.sum(x) - 1])


class TestPhaseII(unittest.TestCase):
    """
        Several tests for component :class:`CobraPhaseII`
    """
    def test_plog(self):
        xi = np.array([2, 100])

        fxi = fn(xi)
        for pshift in [1,10,50]:
            pxi = plog(fxi,pShift=pshift)
            fxi2 = plogReverse(pxi, pShift=pshift)
            assert np.allclose(fxi, fxi2)
        print("[test_plog passed]")

    def test_train_surr(self):
        """ Test whether trainSurrogates works as expected:

            - whether results from adFit are numerically equivalent to R (see demo-train-surr.R) for three different
              fitness function ranges (fnfac = [1, 10, 100])
            - whether p2.fitnessSurrogate and p2.constraintSurrogate (only kernel="cubic", degree=None --> 1)
              interpolate the same values as in R (see yfit and ycon in demo-train-surr.R)
            - whether calcPEffect gives the same results (same errRatio and pEffect as in demo-train-surr.R)
        """
        nobs = 10
        xStart = np.array([2.5, 2.4])
        lower = np.array([-5, -5])
        upper = np.array([5, 5])
        xflat = np.array([[ 1.3,  4.1],
                          [-4.5, -2.3],
                          [ 2.4, -2.1],
                          [-3.9, -1.6]])
        for fnfac in [1,10,100]:
            def fn(x):
                return np.array([3 * fnfac * np.sum(x ** 2), np.sum(x) - 1])
            cobra = CobraInitializer(xStart, fn, "fName", lower, upper,
                                     s_opts=SACoptions(verbose=verb, feval=2*nobs,
                                                       ID=IDoptions(initDesign="RAND_R"),
                                                       ISA=ISAoptions(TFRange=500)))
            c2 = CobraPhaseII(cobra)
            p2 = c2.get_p2()

            trainSurrogates(cobra, p2)

            fr_bef = p2.adFit.get_FRange_before()
            fr_aft = p2.adFit.get_FRange_after()
            if verb == 2:
                print(f"[adFit] FRange before = {fr_bef}, PLOG={p2.PLOG[-1]}, FRange after = {fr_aft}")
            if p2.PLOG[-1]:
                assert fr_bef != fr_aft, "[adFit] PLOG=True, but FRange_after == FRange_before"
            else:
                assert fr_bef == fr_aft, "[adFit] PLOG=True, but FRange_after != FRange_before"
            # the following ...from_R are copied from the output of demo-train-surr.R
            if fnfac == 1:
                Fres_from_R = np.array([114.319589, 7.918231, 3.130921, 13.038905, 36.030000])
                yfit_from_R = np.array([270.3582, 583.9044, 143.1132, 466.6880])
                errRatio_from_R = np.array([9.772734e-03, 3.304260e-06, 1.154493e-03, 1.950927e-04])
                pEffect_from_R = -3.170829
            elif fnfac == 10:
                Fres_from_R = np.array([7.042457, 4.384303, 3.475352, 4.878163, 5.889709])
                yfit_from_R = np.array([11.80334, 22.26042, 14.55043, 17.54240])
                errRatio_from_R = np.array([1.000507, 1.117525, 1.062085, 1.052738])
                pEffect_from_R = 0.02424406
            elif fnfac == 100:
                Fres_from_R = np.array([9.344256, 6.675600, 5.749686, 7.173874, 8.189800])
                yfit_from_R = np.array([14.03572, 24.69325, 16.94072, 19.92351])
                errRatio_from_R = np.array([1.000095, 1.005193, 1.003286, 1.002844])
                pEffect_from_R = 0.001329133
            else:
                raise NotImplementedError(f"No data from the R side for fnfac={fnfac}")
            assert np.allclose(p2.adFit(), Fres_from_R, rtol=1e-6), "Fres_from_R and Fres after adFit are not close"
            # print(p2.adFit() - Fres_from_R)

            yfit = p2.fitnessSurrogate(xflat)
            ycon = p2.constraintSurrogates(xflat).reshape(4,)
            ycon_from_R = np.array([26.0, -35.0,   0.5, -28.5])
            assert np.allclose(yfit, yfit_from_R, rtol=1e-6), "yfit_from_R and yfit are not close"
            assert np.allclose(ycon, ycon_from_R, rtol=1e-6), "ycon_from_R and ycon are not close"

            for i in range(xflat.shape[0]):
                xNew = xflat[i,:]
                xNewEval = fn(xNew)
                calcPEffect(cobra, p2, xNew, xNewEval)
                dummy = 0
            # print(p2.errRatio)
            assert np.allclose(p2.errRatio, errRatio_from_R, rtol=3e-6), "errRatio_from_R and p2.errRatio are not close"
            assert np.allclose(p2.pEffect, pEffect_from_R, rtol=3e-6), "pEffect_from_R and p2.pEffect are not close"
            print(f"fnfac={fnfac:3d}: p2.PLOG = {p2.PLOG}, p2.pEffect = {p2.pEffect}")
        print("[test_train_surr passed]")

    def test_nlopt(self):
        """ Test whether package ``nlopt`` works as expected on a simple constrained problem:

            - When the objective is the sphere function and the constraint is :math:`\sum{x} \geq 1`, then the
              constraint is active and the result should be x = [0.5, 0.5] (the vector with smallest norm that fulfills
              the constraint).
            - When the constraint is :math:`\sum{x} \leq 1`, then the constraint is inactive and the result is the global
              optimum x = [0.0, 0.0].
        """
        def fn_nl(x, grad): return 3 * np.sum(x ** 2)

        def gn_active(x, grad): return -(np.sum(x) - 1)

        def gn_inactive(x, grad): return np.sum(x) - 1

        xStart = np.array([2.5,2.5])
        lower = np.array([-5,-5])
        opt = nlopt.opt(nlopt.LN_COBYLA, xStart.size)
        opt.set_lower_bounds(+lower)
        opt.set_upper_bounds(-lower)
        opt.set_min_objective(fn_nl)
        opt.add_inequality_constraint(gn_active, 0)
        opt.set_xtol_rel(1e-6)
        opt.set_maxeval(1000)
        x = opt.optimize(xStart)
        min_f = opt.last_optimum_value()
        assert np.allclose(x, np.array([0.5,0.5]))
        assert np.allclose(min_f, 1.5)       # 1.5 = 3 * (0.5**2 + 0.5**2)
        opt.remove_inequality_constraints()
        opt.add_inequality_constraint(gn_inactive, 0)
        x = opt.optimize(xStart)
        min_f = opt.last_optimum_value()
        assert np.linalg.norm(x - np.array([0.0,0.0])) < 1e-5
        assert np.allclose(min_f, 0.0)
        print(f"x = {x}, min_f = {min_f}, result code = {opt.last_optimize_result()}")
        print("[test_nlopt passed]")

    def test_seq_funcs(self):
        """ Test whether the :class:`SeqFuncFactory` functions ``subProb2`` and ``gCOBRA`` work as expected
            (for different distance requirement parameters ``p2.ro``)
        """
        nobs = 10
        xStart = np.array([2.5, 2.4])
        lower = np.array([-5, -5])
        upper = np.array([5, 5])
        def fn(x):
            return np.array([3 * np.sum(x ** 2), np.sum(x) - 1])

        cobra = CobraInitializer(xStart, fn, "fName", lower, upper,
                                 s_opts=SACoptions(verbose=verb, feval=2*nobs,
                                                   ID=IDoptions(initDesign="RAND_R"),
                                                   ISA=ISAoptions(TFRange=500)))
        rw = RescaleWrapper(fn, lower, upper, cobra.sac_res['lower'], cobra.sac_res['upper'])
        xflat = np.array([[ 1.3,  4.1],
                          [-4.5, -2.3],
                          [ 2.4, -2.1],
                          [-3.9, -1.6]])
        for i in range(xflat.shape[0]):
            xflat[i,:] = rw.forward(xflat[i,:])
        c2 = CobraPhaseII(cobra)
        p2 = c2.get_p2()


        trainSurrogates(cobra, p2)

        for ro in [1, 0.5]:
            p2.ro = ro
            sf_factory = SeqFuncFactory(cobra, p2)
            def gCOBRA_c(x):
                # inequality constraints for nloptr::cobyla
                return -sf_factory.gCOBRA(x, None)  # new convention h_i <= 0.

            yfit2 = np.apply_along_axis(lambda x: sf_factory.subProb2(x, None), axis=1, arr=xflat)
            ycon2 = np.apply_along_axis(gCOBRA_c, axis=1, arr=xflat)
            # print(yfit2)
            # print(ycon2)
            # the following variables ...from_R are copied from the output of demo-innerFuncs.R:
            yfit2_from_R = np.array([36.63426, 82.37762, 20.04144, 60.94728])
            if ro == 1:
                ycon2_from_R = np.array([[1.3090944,  4.4001],
                                         [ 0.6828887, -7.7999],
                                         [ 1.2513251, -0.6999],
                                         [ 0.6592290, -6.49999]])
            elif ro == 0.5:
                ycon2_from_R = np.array([[0.08382696,  4.4001],
                                         [0.18288874, -7.7999],
                                         [0.24982102, -0.6999],
                                         [0.01315809, -6.49999]])
            else:
                raise NotImplementedError(f"No data from the R side for ro={ro}")
            assert np.allclose(yfit2, yfit2_from_R)
            assert np.allclose(ycon2, ycon2_from_R, rtol=2e-5)
        print("[test_inner_funcs passed]")


    def test_phaseII(self):
        """ Test whether the inner funcs ``subProb2`` and ``gCOBRA`` work as expected (for different distance
            requirement parameters ``p2.ro``)
        """
        nobs = 10
        xStart = np.array([2.5, 2.4])
        lower = np.array([-5, -5])
        upper = np.array([5, 5])

        def fn(x):
            return np.array([3 * np.sum(x ** 2), -(np.sum(x) - 1)])

        cobra = CobraInitializer(xStart, fn, "fName", lower, upper,
                                 s_opts=SACoptions(verbose=verb, feval=2 * nobs,
                                                   ID=IDoptions(initDesign="RAND_R",rescale=False),
                                                   ISA=ISAoptions(TFRange=500),
                                                   SEQ=SEQoptions(trueFuncForSurrogates=True)))
        c2 = CobraPhaseII(cobra)
        p2 = c2.get_p2()
        c2.start()


if __name__ == '__main__':
    unittest.main()
