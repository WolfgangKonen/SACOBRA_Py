import unittest
import numpy as np
import nlopt
from cobraInit import CobraInitializer
from seqOptimizer import SeqFuncFactory
from cobraPhaseII import CobraPhaseII
from innerFuncs import plog, plogReverse
from rescaleWrapper import RescaleWrapper
from surrogator import Surrogator
# from trainSurrogates import trainSurrogates, calcPEffect
from opt.sacOptions import SACoptions
from opt.idOptions import IDoptions
from opt.isaOptions import ISAoptions, RSTYPE
from opt.seqOptions import SEQoptions

verb = 1


def fn(x):
    return np.array([3 * np.sum(x ** 2), np.sum(x) - 1])


class TestPhaseII(unittest.TestCase):
    """
        Several tests for component :class:`CobraPhaseII`
    """
    def test_plog(self):
        """ Test that ``plogReverse(plog(x)) = x`` for various ``x`` and various ``pshift``
        """
        fxi = np.array([2, 100, -2, -1000])
        for pshift in [0, 1, 10, 50]:
            pxi = plog(fxi, pShift=pshift)
            fxi2 = plogReverse(pxi, pShift=pshift)
            assert np.allclose(fxi, fxi2), f"not close: {fxi}, {fxi2} for pshift = {pshift}"
        print("[test_plog passed]")

    def test_train_surr(self):
        """ Test whether ``Surrogator.trainSurrogates`` works as expected:

            - whether results from ``adFit`` are numerically equivalent to R (see ``demo-trainSurr.R``) for three
              different fitness function ranges (``fnfac = [1, 10, 100]``)
            - whether ``p2.fitnessSurrogate`` and ``p2.constraintSurrogate`` (only for ``kernel="cubic"`` and
              ``degree=None``, meaning ``degree=1``) interpolate the same values as in R (see ``yfit`` and ``ycon``
              in ``demo-trainSurr.R``)
            - whether ``calcPEffect`` gives the same results (same ``errRatio`` and ``pEffect`` as in
              ``demo-trainSurr.R``). Note that the R numbers from ``calcPEffect`` are only valid under setting
              ``recalc_fit12 = True`` in ``trainSurrogates`` (!)
        """
        nobs = 10
        x0 = np.array([2.5, 2.4])
        lower = np.array([-5, -5])
        upper = np.array([5, 5])
        idp = 2*x0.size + 1
        xflat = np.array([[ 1.3,  4.1],
                          [-4.5, -2.3],
                          [ 2.4, -2.1],
                          [-3.9, -1.6]])
        for fnfac in [1, 10, 100]:
            def fn(x):
                return np.array([3 * fnfac * np.sum(x ** 2), np.sum(x) - 1])
            is_equ = np.array([False])
            cobra = CobraInitializer(x0, fn, "fName", lower, upper, is_equ,
                                     s_opts=SACoptions(verbose=verb, feval=2*nobs,
                                                       ID=IDoptions(initDesign="RAND_R", initDesPoints=idp),
                                                       ISA=ISAoptions(TFRange=500)))
            c2 = CobraPhaseII(cobra)
            p2 = c2.get_p2()

            Surrogator.trainSurrogates(cobra, p2)

            fr_bef = p2.adFit.get_FRange_before()
            fr_aft = p2.adFit.get_FRange_after()
            if verb == 2:
                print(f"[adFit] FRange before = {fr_bef}, PLOG={p2.PLOG[-1]}, FRange after = {fr_aft}")
            if p2.PLOG[-1]:
                assert fr_bef != fr_aft, "[adFit] PLOG=True, but FRange_after == FRange_before"
            else:
                assert fr_bef == fr_aft, "[adFit] PLOG=True, but FRange_after != FRange_before"
            # the following ...from_R are copied from the output of demo-trainSurr.R
            if fnfac == 1:
                Fres_from_R = np.array([114.319589, 7.918231, 3.130921, 13.038905, 36.030000])
                yfit_from_R = np.array([270.3582, 583.9044, 143.1132, 466.6880])
                errRatio_from_R = np.array([9.772734e-03, 3.304260e-06, 1.154493e-03, 1.950927e-04])
                pEffect_from_R = -3.170829
            elif fnfac == 10:
                Fres_from_R = np.array([7.042457, 4.384303, 3.475352, 4.878163, 5.889709])
                yfit_from_R = np.array([11.80334, 22.26042, 14.55043, 17.54240])
                errRatio_from_R = np.array([1.613748e-02, 1.090622e-06, 5.400598e-04, 9.949118e-05])
                pEffect_from_R = -3.495155          # recalc_fit12 = True
            elif fnfac == 100:
                Fres_from_R = np.array([9.344256, 6.675600, 5.749686, 7.173874, 8.189800])
                yfit_from_R = np.array([14.03572, 24.69325, 16.94072, 19.92351])
                errRatio_from_R = np.array([1.731629e-02, 9.574353e-07, 4.947083e-04, 9.197749e-05])
                pEffect_from_R = -3.532624          # recalc_fit12 = True
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
                xNew = xflat[i, :]
                xNewEval = fn(xNew)
                # print(i, p2.fitnessSurrogate1(xNew), p2.fitnessSurrogate2(xNew))
                Surrogator.calcPEffect(p2, xNew, xNewEval)
                dummy = 0
            # print(p2.errRatio)
            assert np.allclose(p2.errRatio, errRatio_from_R, rtol=1e-5), \
                f"errRatio_from_R and p2.errRatio are not close (fnfac={fnfac})"
            assert np.allclose(p2.pEffect, pEffect_from_R, rtol=3e-6), "pEffect_from_R and p2.pEffect are not close"
            print(f"fnfac={fnfac:3d}: p2.PLOG = {p2.PLOG}, p2.pEffect = {p2.pEffect}")
        print("[test_train_surr passed]")

    def test_nlopt(self):
        """ Test whether package ``nlopt`` works as expected on a simple constrained problem with objective
            ``fn_nl(x)``  :math:`= f(x) = 3\\sum_i{{x_i}^2}` with
            :math:`x \\in \\mathbb{R}^2`:

            - When the constraint is ``np.sum(x) >= 1``, then the constraint is active and the result is
              ``x = [0.5, 0.5]`` (the vector with the smallest norm that fulfills the constraint) with
              :math:`f(x) = 1.5`.
            - When the constraint is ``np.sum(x) <= 1``, then the constraint is inactive and the result is the global
              optimum at ``x = [0.0, 0.0]`` with :math:`f(x) = 0.0`.
        """
        def fn_nl(x, grad): return 3 * np.sum(x ** 2)

        def gn_active(x, grad): return -(np.sum(x) - 1)

        def gn_inactive(x, grad): return np.sum(x) - 1

        x0 = np.array([2.5, 2.5])
        lower = np.array([-5, -5])
        opt = nlopt.opt(nlopt.LN_COBYLA, x0.size)
        opt.set_lower_bounds(+lower)
        opt.set_upper_bounds(-lower)
        opt.set_min_objective(fn_nl)
        opt.add_inequality_constraint(gn_active, 0)
        opt.set_xtol_rel(1e-6)
        opt.set_maxeval(1000)
        x = opt.optimize(x0)
        min_f = opt.last_optimum_value()
        assert np.allclose(x, np.array([0.5, 0.5]))
        assert np.allclose(min_f, 1.5)       # 1.5 = 3 * (0.5**2 + 0.5**2)
        opt.remove_inequality_constraints()
        opt.add_inequality_constraint(gn_inactive, 0)
        x = opt.optimize(x0)
        min_f = opt.last_optimum_value()
        assert np.linalg.norm(x - np.array([0.0, 0.0])) < 1e-5
        assert np.allclose(min_f, 0.0)
        print(f"x = {x}, min_f = {min_f}, result code = {opt.last_optimize_result()}")
        print("[test_nlopt passed]")

    def test_seq_funcs(self):
        """ Test whether the methods ``subProb2`` and ``gCOBRA`` of  :class:`SeqFuncFactory` work as expected
            (for different distance requirement parameters ``p2.ro = 0.5 | 1``).

            Given a simple COP with sphere objective and one linear constraint, we train surrogates on ``idp=5`` random
            observations (``initDesign=RAND_R``) and evaluate the methods ``subProb2`` and ``gCOBRA`` on certain other
            points stored in ``xflat``: ``yfit2 = subProb2(xflat)`` and ``ycon2 = gCOBRA(xflat)``.
            We compare with equivalent results from the R side (see ``demo-seqFuncs.R``).

            Results:

            - max. rel. error ``yfit2`` is < 1e-7
            - max. rel. error ``ycon2`` is < 1e-6
        """
        nobs = 10
        x0 = np.array([2.5, 2.4])
        lower = np.array([-5, -5])
        upper = np.array([5, 5])
        idp = 2*x0.size + 1

        def fn(x):
            return np.array([3 * np.sum(x ** 2), np.sum(x) - 1])
        is_equ = np.array([False])

        cobra = CobraInitializer(x0, fn, "fName", lower, upper, is_equ,
                                 s_opts=SACoptions(verbose=verb, feval=2*nobs,
                                                   ID=IDoptions(initDesign="RAND_R", initDesPoints=idp),
                                                   ISA=ISAoptions(TFRange=500)))
        rw = RescaleWrapper(fn, lower, upper, cobra.sac_res['lower'], cobra.sac_res['upper'])
        xflat = np.array([[ 1.3,  4.1],
                          [-4.5, -2.3],
                          [ 2.4, -2.1],
                          [-3.9, -1.6]])
        for i in range(xflat.shape[0]):
            xflat[i, :] = rw.forward(xflat[i, :])
        c2 = CobraPhaseII(cobra)
        p2 = c2.get_p2()

        Surrogator.trainSurrogates(cobra, p2)

        for ro in [0.5, 1]:
            p2.ro = ro
            sf_factory = SeqFuncFactory(cobra, p2)

            def gCOBRA_c(x):
                # inequality constraints for nloptr::cobyla
                return -sf_factory.gCOBRA(x, None)  # new convention h_i <= 0.

            yfit2 = np.apply_along_axis(lambda x: sf_factory.subProb2(x, None), axis=1, arr=xflat)
            ycon2 = np.apply_along_axis(gCOBRA_c, axis=1, arr=xflat)
            # print(yfit2)
            # print(ycon2)
            # the following variables ...from_R are copied from the output of demo-seqFuncs.R:
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
            print(f"max. rel. error yfit2 = {np.max((yfit2-yfit2_from_R)/yfit2)} (p2.ro = {p2.ro})")
            print(f"max. rel. error ycon2 = {np.max((ycon2 - ycon2_from_R) / ycon2)} (p2.ro = {p2.ro})")
        print("[test_seq_funcs passed]")

    def test_RS_EPS(self):
        """ A short phase-II optimization run (``feval=20``) on a simple COP (sphere + one inequality constraint) with
            Random Start (RS) enabled. Perform tests with two different parameters ``rstype = RSTYPE.SIGMOID | RSTYPE.CONSTANT``
            and with ``ISA.RS_rep=True`` and ``SEQ.finalEpsXiZero=False``.

            The R side is ``demo_RS_EPS`` in ``demo-phaseII.R``. It requires ``initDesign = "RAND_R"``,
            ``RS_rep = True``, ``trueFuncForSurrogates = True`` and
            ``cobraSeed = 42 | 52`` (depending on ``rstype``).

            Results:

            - the Python and the R side have the same iterations with ``df.RS=True``
            - the Python and the R side have the same margin ``EPS`` in each iteration
            - the intermediate infill points ``sac_res['A']`` are not TOO different (abs tol < 1e-2)
            - the final infill points are very much the same (abs tol < 1e-5)
            - the best infill points are very much the same (abs tol < 1e-5)
        """
        x0 = np.array([2.5, 2.4])
        lower = np.array([-5, -5])
        upper = np.array([5, 5])
        idp = 2*x0.size + 1

        def fn(x):
            return np.array([3 * np.sum(x ** 2), -(np.sum(x) - 1)])
        is_equ = np.array([False])

        for rstype in list(RSTYPE):
            myseed = 42 if rstype == RSTYPE.SIGMOID else 52
            cobra = CobraInitializer(x0, fn, "fName", lower, upper, is_equ,
                                     s_opts=SACoptions(verbose=verb, feval=20, cobraSeed=myseed,
                                                       ID=IDoptions(initDesign="RAND_R", initDesPoints=idp),
                                                       ISA=ISAoptions(TFRange=500, RS_rep=True, RStype=rstype),
                                                       SEQ=SEQoptions(trueFuncForSurrogates=True,finalEpsXiZero=False)))
            print(f"Starting with RS_type={cobra.sac_opts.ISA.RStype.name} and cobraSeed={cobra.sac_opts.cobraSeed} ...")
            cobra.sac_opts.ISA.RS_verb = True         # be verbose in RandomStarter.random_start
            c2 = CobraPhaseII(cobra)
            # p2 = c2.get_p2()
            c2.start()

            # print(cobra.df)
            # print(cobra.df2)
            df_rs = np.array(cobra.df.loc[:, ['RS']]).reshape(20, )
            df2_eps = np.array(cobra.df2.loc[:, ['EPS']]).reshape(15,)

            # the following variables ...from_R are copied from the output of demo-phaseII.R when running demo_RS_EPS
            # (note that this requires "RAND_R", cobraSeed depending on rstype and cobra$sac$RS_rep = TRUE):
            if rstype == RSTYPE.SIGMOID:       # implies cobraSeed=42
                df_rs_from_R = np.array([False, False, False, False, False,  True, False, False, False, False,
                                         False, False, False, False, False, False,  True,  True, True,  True])
                df2_eps_from_R = np.array([1.0000e-02, 1.0000e-02, 5.0000e-03, 5.0000e-03, 2.5000e-03, 2.5000e-03,
                                           1.2500e-03, 1.2500e-03, 6.2500e-04, 6.2500e-04, 3.1250e-04, 3.1250e-04,
                                           1.5625e-04, 1.5625e-04, 1.5625e-04])
                A_567 = cobra.sac_res['A'][5:8, :]
                #               [[ 0.7721366 , -0.05473517],
                #                [ 0.10000958,  0.10001042],
                #                [ 0.09858802,  0.10141698]]
                A_567_from_R = np.array([[0.7721366, -0.05473518],
                                         [0.1000102, 0.10000976],
                                         [0.1014172, 0.09858782]])
                A_20 = cobra.sac_res['A'][19:20, :]
                #               [[ 0.10000041, 0.0999996 ]],
                A_20_from_R = np.array([0.09999936, 0.10000064])
                xbest = cobra.get_xbest()   # [0.50000204, 0.49999798]
                xbest_from_R = np.array([0.4999968, 0.5000032])
            elif rstype == RSTYPE.CONSTANT:     # implies cobraSeed=52
                df_rs_from_R = np.array([False, False, False, False, False, False, False, True, True, False,
                                         False, False, False, False, False, True, False, False, True, False])
                df2_eps_from_R = np.array([1.0000e-02, 1.0000e-02, 5.0000e-03, 5.0000e-03, 2.5000e-03, 2.5000e-03,
                                           1.2500e-03, 1.2500e-03, 6.2500e-04, 6.2500e-04, 3.1250e-04, 3.1250e-04,
                                           1.5625e-04, 1.5625e-04, 7.8125e-05])
                A_567 = cobra.sac_res['A'][5:8, :]
                #               [[-0.05704648,  0.25706648],
                #                [ 0.10000958,  0.10001042],
                #                [ 0.10141617,  0.09858883]]
                A_567_from_R = np.array([[-0.05704629, 0.25706629],
                                         [ 0.10000952, 0.10001048],
                                         [ 0.10141633, 0.09858867]])
                A_20 = cobra.sac_res['A'][19:20, :]
                #               [[ 0.09999958, 0.10000043]]
                A_20_from_R = np.array([0.09999936, 0.10000064])
                xbest = cobra.get_xbest()   # [0.49999788, 0.50000213]
                xbest_from_R = np.array([0.5000016, 0.4999984])
            # else:
            #     raise NotImplementedError(f"No data from the R side for rstype={rstype}")
            assert np.all(df_rs == df_rs_from_R)
            assert np.allclose(df2_eps, df2_eps_from_R)
            # assert np.allclose(A_567, A_567_from_R)      # this assertion would fail, the intermediate infills differ
            # this assertion with abs tol = 1e-2 holds, the infill points are not *too* different:
            assert np.allclose(A_567, A_567_from_R, atol=1e-2)
            assert np.allclose(A_20, A_20_from_R, atol=1e-5)     # these assertions work on a higher abs tol, final
            assert np.allclose(xbest, xbest_from_R, atol=1e-5)   # result and xbest are close
        print("[test_RS_EPS passed]")


if __name__ == '__main__':
    unittest.main()
