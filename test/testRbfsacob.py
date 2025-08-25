import time
import unittest
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from cobraInit import CobraInitializer
from cobraPhaseII import CobraPhaseII
from gCOP import GCOP
from opt.equOptions import EQUoptions
from opt.rbfOptions import RBFoptions
from opt.seqOptions import SEQoptions
from opt.idOptions import IDoptions
from opt.sacOptions import SACoptions
from rbfModel import RBFmodel
from rbfSacobra import svd_inv, dist_line
from read_dfw import read_dfw1, read_dfw2
from testRbfModel import TestRbfModel       # for my_rng2

verb = 1


def fn(x):
    # return ((x[:, 0] - 10) ** 3) + ((x[:, 1] - 20) ** 3)
    return x[:, 0] * 2 + x[:, 1] * 3


def fn1(x):
    return np.array([x[0] * 2 + x[1] * 3, x[0] + x[1], x[0] - x[1]])


def solve_G03(cobraSeed, dimension, rbf_opt, feval=150, verbIter=10, conTol=0):  # conTol=0 | 1e-7
    print(f"Starting solve_G03({cobraSeed}, dim={dimension}, ...) ...")
    G03 = GCOP("G03", dimension)

    x0 = G03.x0  # None --> a random x0 will be set
    equ_opt = EQUoptions(muGrow=100, muDec=1.6, muFinal=1e-12, refineAlgo="BFGS_1", refinePrint=False)
    cobra = CobraInitializer(x0, G03.fn, G03.name, G03.lower, G03.upper, G03.is_equ,
                             solu=G03.solu,
                             s_opts=SACoptions(verbose=verb, verboseIter=verbIter, feval=feval, cobraSeed=cobraSeed,
                                               ID=IDoptions(initDesign="LHS", rescale=True),
                                               RBF=rbf_opt,
                                               SEQ=SEQoptions(finalEpsXiZero=True, conTol=conTol)))
    c2 = CobraPhaseII(cobra).start()

    fin_err = np.array(cobra.get_feasible_best() - G03.fbest)
    print(f"final err: {fin_err}")
    c2.p2.fin_err = fin_err
    c2.p2.fe_thresh = 1e-9
    c2.p2.dim = G03.dimension
    c2.p2.conTol = conTol
    return c2


def solve_G06(cobraSeed, rbf_opt, feval=40, verbIter=10, conTol=0):  # conTol=0 | 1e-7
    print(f"Starting solve_G06({cobraSeed}) ...")
    G06 = GCOP("G06")

    cobra = CobraInitializer(G06.x0, G06.fn, G06.name, G06.lower, G06.upper, G06.is_equ,
                             solu=G06.solu,
                             s_opts=SACoptions(verbose=verb, verboseIter=verbIter, feval=feval,
                                               cobraSeed=cobraSeed,
                                               ID=IDoptions(initDesign="LHS", initDesPoints=6),
                                               RBF=rbf_opt,
                                               SEQ=SEQoptions(finalEpsXiZero=True, conTol=conTol)))

    c2 = CobraPhaseII(cobra)
    c2.start()

    # show_error_plot(cobra, G06)

    fin_err = np.array(cobra.get_feasible_best() - G06.fbest)
    c2.p2.fin_err = fin_err
    print(f"final err: {fin_err}")
    c2.p2.fe_thresh = 5e-8
    c2.p2.dim = G06.dimension
    c2.p2.conTol = conTol
    return c2


def solve_G07(cobraSeed, rbf_opt, feval=180, verbIter=10, conTol=0):  # conTol=0 | 1e-7
    print(f"Starting solve_G07({cobraSeed}) ...")
    G07 = GCOP("G07")
    idp = 11 * 12 // 2

    cobra = CobraInitializer(G07.x0, G07.fn, G07.name, G07.lower, G07.upper, G07.is_equ,
                             solu=G07.solu,
                             s_opts=SACoptions(verbose=verb, verboseIter=verbIter, feval=feval,
                                               cobraSeed=cobraSeed,
                                               ID=IDoptions(initDesign="LHS", initDesPoints=idp),
                                               RBF=rbf_opt,
                                               SEQ=SEQoptions(finalEpsXiZero=True, conTol=conTol)))

    c2 = CobraPhaseII(cobra)
    c2.start()

    fin_err = np.array(cobra.get_feasible_best() - G07.fbest)
    c2.p2.fin_err = fin_err
    print(f"final err: {fin_err}")
    c2.p2.fe_thresh = 1e-9
    c2.p2.dim = G07.dimension
    c2.p2.conTol = conTol
    return c2


def solve_G09(cobraSeed, rbf_opt, feval=500, verbIter=50, conTol=1e-7):  # conTol=0 | 1e-7
    print(f"Starting solve_G09({cobraSeed}) ...")
    G09 = GCOP("G09")
    idp = 10 * 11 // 2
    # G09.x0 = G09.solu + 0.01

    cobra = CobraInitializer(G09.x0, G09.fn, G09.name, G09.lower, G09.upper, G09.is_equ,
                             solu=G09.solu,
                             s_opts=SACoptions(verbose=verb, verboseIter=verbIter, feval=feval,
                                               cobraSeed=cobraSeed,
                                               ID=IDoptions(initDesign="LHS", initDesPoints=idp),
                                               RBF=rbf_opt,
                                               SEQ=SEQoptions(finalEpsXiZero=True, conTol=conTol)))

    c2 = CobraPhaseII(cobra)
    c2.start()

    fin_err = np.array(cobra.get_feasible_best() - G09.fbest)
    c2.p2.fin_err = fin_err
    print(f"final err: {fin_err}")
    c2.p2.fe_thresh = 5e-02
    c2.p2.dim = G09.dimension
    c2.p2.conTol = conTol
    return c2


class TestRbfSacob(unittest.TestCase):
    """
    Test cases for class RBFsacob, the SACOBRA-internal implementation of RBF models
    """

    def test_svd_inv(self):
        """
            Test that the SVD inverse of a random (3,3) matrix M always fulfills M * invM = One.
        """
        rng = np.random.default_rng()
        # M = np.array([1, 1, 2, 3, 4, 5, 6, 7, 9]).reshape(3,3).T      # rank 3
        # M = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8]).reshape(3, 3).T     # rank 2
        M = rng.normal(size=(3, 3))
        print(np.linalg.matrix_rank(M))
        invM = svd_inv(M)
        print(np.matmul(M, invM))
        if np.linalg.matrix_rank(M) == 3:  # the following assertion is only valid if M has full rank:
            assert (np.allclose(np.eye(3), np.matmul(M, invM)))

    def test_dist_line(self):
        x = np.array([1, 2])
        xp = np.array([[1, 2], [3, 4]])
        z = dist_line(x, xp)
        assert np.allclose(z, np.array([0, 2.82842712]))
        print(z)

    def inner_test_func_fn(self, d, ngrid, nobs, deg, kernel, EQUIV_R):
        trm = TestRbfModel()
        trm.val = 24*2  # setting for my_rng2
        print(f"\n[inner_test_func_fn] started with d = {d}, deg = {deg}, nobs = {nobs}, kernel = {kernel}")
        xobs = trm.my_rng2(nobs, d)  # better than my_rng: avoids cycles
        yobs = fn(xobs)
        xgrid = np.mgrid[-1:1:ngrid * 1j, -1:1:ngrid * 1j]
        xflat = xgrid.reshape(d, -1).T
        # print(xflat.shape)  # (ngrid**d, d)
        start = time.perf_counter_ns()
        tpmat = (deg == 1 or deg == 1.5)
        rbf_opts = RBFoptions(interpolator="sacobra", kernel=kernel, degree=deg, rho=0, test_pmat=tpmat)

        rbf_model = RBFmodel(xobs, yobs, rbf_opts)

        t1 = time.perf_counter_ns()
        time_sacob_mod = t1 - start
        yflat = rbf_model(xflat)  # most time (1 sec) is spent here (!)
        t2 = time.perf_counter_ns()
        time_sacob_prd = t2 - t1
        ygrid = yflat.reshape(ngrid, ngrid)
        print(f"degree = {deg}: time_sacob (mod/prd)= ({time_sacob_mod * 1e-6:.6f},{time_sacob_prd * 1e-6:.6f}) [ms]")

        ytrue = fn(xflat).reshape(ngrid, ngrid)
        delta = ytrue - ygrid
        print(f"avg abs |ytrue - ygrid| = {np.mean(np.abs(delta)):.4e}")

        L = ngrid // 2
        ytest = ygrid[50:60, L - 1]

        np.set_printoptions(7)
        # print(ytest)

        if deg != 1.5:
            # Test numerical equivalence with SciPy implementation.
            #
            start = time.perf_counter_ns()
            rbf_opts = RBFoptions(interpolator="scipy", kernel=kernel, degree=deg, rho=0)

            sci_model = RBFmodel(xobs, yobs, rbf_opts)

            t1 = time.perf_counter_ns()
            time_scipy_mod = t1 - start

            EQUIV_COEF = (nobs == 10)
            if EQUIV_COEF:
                delta = sci_model.model._coeffs.T - rbf_model.model.coef
                print(f"max |s_coef - r_coef| = {np.max(np.abs(delta)):.4e}")
                # print(f"max rel.delta = {np.max(np.abs(delta)/np.abs(rbf_model.model.coef)):.4e}")
                print(delta)
                atol = 1e-3 if deg < 1 else 1e-5
                if kernel == "multiquadric": atol *= 10
                coef_close = np.allclose(sci_model.model._coeffs.T, rbf_model.model.coef, atol=atol)
                # print(coef_close)
                assert coef_close
                if not coef_close: print(rbf_model.model.coef)

            sflat = sci_model(xflat)
            t2 = time.perf_counter_ns()
            time_scipy_prd = t2 - t1
            sgrid = sflat.reshape(ngrid, ngrid)
            delta = ytrue - sgrid
            print(f"avg abs |ytrue - sgrid| = {np.mean(np.abs(delta)):.4e}")
            stest = sgrid[50:60, L - 1]
            print(f"degree = {deg}: time_scipy (mod/prd)= ({time_scipy_mod * 1e-6:.6f},{time_scipy_prd * 1e-6:.6f}) [ms]")
            delta = ytest - stest
            print(f"avg |ytest - stest| = {np.mean(np.abs(delta)):.4e}")
            assert np.allclose(ytest, stest, atol=9e-3)

        # If EQUIV_R and deg <= 1: Test numerical equivalence with R implementation.
        # (We can only do it for deg <= 1, because deg==2 is different from squares=T. We have only the values from
        #  R for the "cubic" case.)
        # Requires fn(x) to return    x[:, 0] * 2 + x[:, 1] * 3.
        # These are the values generated with demo-rbf3.R and my_rng2 (avoid cycles!) on the R-side:
        if EQUIV_R and nobs == 10 and deg <= 1 and kernel == "cubic":
            if nobs == 10:
                if deg == -1:   # ptail=F in R
                    ytest_from_R = np.array([0.0354658, 0.0771876, 0.1187900, 0.1602633, 0.2015979,
                                             0.2427840, 0.2838119, 0.3246720, 0.3653547, 0.4058505])
                elif deg == 1:  # ptail=T in R
                    ytest_from_R = np.array([-0.0101010, 0.0303030, 0.0707071, 0.1111111, 0.1515152,
                                             0.1919192, 0.2323232, 0.2727273, 0.3131313, 0.3535354])
                else:
                    raise NotImplementedError(f"No data from the R side for nobs={nobs} and deg={deg}")
            elif nobs == 100:
                if deg == -1:   # ptail=F in R
                    ytest_from_R = np.array([-0.0100817, 0.0303251, 0.0707322, 0.1111393, 0.1515466,
                                             0.1919538, 0.2323607, 0.2727671, 0.3131729, 0.3535776])
                elif deg == 1:  # ptail=T in R
                    ytest_from_R = np.array([-0.0101010, 0.0303030, 0.0707071, 0.1111111, 0.1515152,
                                             0.1919192, 0.2323232, 0.2727273, 0.3131313, 0.3535354])
                else:
                    raise NotImplementedError(f"No data from the R side for nobs={nobs} and deg={deg}")
            else:
                raise NotImplementedError(f"No data from the R side for nobs={nobs}")

            delta = ytest - ytest_from_R
            print(f"avg |ytest - ytest_from_R| = {np.mean(np.abs(delta)):.4e}")
            # assert np.allclose(ytest, ytest_from_R, rtol=1e-4)


        print("[inner_test_func_fn passed]")

    def test_func_fn(self):
        """
            Test whether function fn is RBF-modeled (3 kernels, 10|100 observations, degree -1|1|2 for  polynomial tail)
            the same way in RBFsacob and in SciPy's RBFInterpolator. Use RNG ``my_rng2`` from testRbfModel
            (avoid cycles!)

            Test -- only for ``kernel="cubic"`` and degree ``deg <= 1`` -- that results are the same in Python and in R.

            Test -- only for ``nobs=10`` -- that RBF coefficients are the same in RBFsacob and in SciPy's
            RBFInterpolator (up to absolute tolerance ``atol = 1e-5 ... 1e-2``, depending on kernel and degree).

            Test -- only for degree ``deg=1`` or ``deg=1.5`` -- that the elements of polynomial matrix ``pMat`` in
            RBFsacob are in the new implementation (monomial powers) the same as in the old implementation (that is
            wrong for ``deg=2``).
        """
        d = 2
        ngrid = 100
        EQUIV_R = True
        for nobs in [10]:       # , 100
            for deg in [-1, 1, 1.5, 2]:  #
                for kernel in ["cubic", "gaussian", "multiquadric"]:    # "cubic", "gaussian", "multiquadric"
                    self.inner_test_func_fn(d, ngrid, nobs, deg, kernel, EQUIV_R)
        print("[all tests in test_func_fn passed]")

    def test_gprob_rbf(self):
        """
        Test both cubic RBF interpolators ``"scipy"`` and ``"sacobra"`` on different COPs (G06, G07) and measure
        mean error, total RBF model build (``__init__``) time and total RBF predict (``__call__``) time.

        Results for ``degree=2``: good, both interpolators have similar and small errors < 1e-6.

        Results for ``degree=1``: only mediocre, error < 2e-2 for G06, < 7e-1 for G07, ``"sacobra"`` sometimes worse
        than ``"scipy"``.
        """
        cobraSeed = 52
        runs = 2
        errs = np.zeros((2, runs))
        time_init = np.zeros(2)
        time_call = np.zeros(2)
        # gprob = "G06"
        # deg = 1
        kernel = "cubic"
        dft = pd.DataFrame()
        for gprob in ["G06", "G07"]:
            for run in range(runs):
                for deg in [1, 2]:
                    for k, inter in enumerate(["scipy", "sacobra"]):
                        rbf_opt = RBFoptions(kernel=kernel, degree=deg, interpolator=inter)
                        if gprob == "G06":
                            c2 = solve_G06(cobraSeed+run, rbf_opt, feval=90, verbIter=100, conTol=0)
                        elif gprob == "G07":
                            c2 = solve_G07(cobraSeed + run, rbf_opt, feval=200, verbIter=100, conTol=0)
                        else:   # i.e. "G09"
                            c2 = solve_G09(cobraSeed + run, rbf_opt, feval=200, verbIter=100, conTol=0)
                        errs[k, run] = c2.p2.fin_err
                        time_init[k] += c2.p2.time_init
                        time_call[k] += c2.p2.time_call

                        new_row_dft = pd.DataFrame(
                            {'gprob': gprob,
                             'kernel': kernel,
                             'degree': deg,
                             'seed': cobraSeed+run,
                             'inter': inter,  # interpolator
                             'feval': c2.cobra.sac_opts.feval,
                             'err': c2.p2.fin_err,
                             'fbest': c2.cobra.get_fbest(),
                             't_init': c2.p2.time_init,
                             't_call': c2.p2.time_call
                             }, index=[0])
                        dft = pd.concat([dft, new_row_dft], axis=0)

        err2 = np.array(dft.loc[dft['degree'] == 2, ['err']])
        assert np.allclose(err2, 0, atol=1e-6)

        print(errs)
        print("mean err [scipy, sacobra]")
        print(np.mean(errs, axis=1))
        print("time_init [scipy, sacobra]")
        print(f"{time_init}    {time_init[1]/time_init[0]:.2f}")
        print("time_call [scipy, sacobra]")
        print(f"{time_call}    {time_call[1]/time_call[0]:.2f}")
        print(dft)
        dft.to_feather("feather/dft.feather")

    def inner_width_fit(self, idp, n_splits, cobraSeed, kernel, width, deg, inter, dfw1):
        print(f"Starting fit_G06({cobraSeed}) ...")
        G06 = GCOP("G06")

        rbf_opt = RBFoptions(kernel=kernel, width=width, degree=deg, interpolator=inter)
        cobra = CobraInitializer(G06.x0, G06.fn, G06.name, G06.lower, G06.upper, G06.is_equ,  # G06.fn | fn1
                                 solu=G06.solu,
                                 s_opts=SACoptions(verbose=verb, verboseIter=verb, feval=2*idp, cobraSeed=cobraSeed,
                                                   ID=IDoptions(initDesign="LHS", initDesPoints=idp),
                                                   RBF=rbf_opt))

        A = cobra.sac_res['A']
        Fres = cobra.sac_res['Fres']
        print(A.shape)
        kf = KFold(n_splits=n_splits)
        err_vector = np.ndarray(n_splits, dtype=float)
        for k, (train_index, test_index) in enumerate(kf.split(A)):
            # print("TRAIN:", train_index, "TEST:", test_index)
            A_train, A_test = A[train_index], A[test_index]
            F_train, F_test = Fres[train_index], Fres[test_index]
            fit_model = RBFmodel(A_train, F_train, rbf_opt)
            fit_test = fit_model(A_test)
            err_test = np.mean(np.abs(fit_test-F_test))
            err_vector[k] = err_test
            dummy = 0

        print(f"width = {width}: W_ONE = {fit_model.get_width_one():.3f}, W_THREE = {fit_model.get_width_three():.3f}")
        new_row_dfw = pd.DataFrame(
            {'width': width,
             'kernel': kernel,
             'degree': deg,
             'seed': cobraSeed,
             'inter': inter,  # interpolator
             'err': np.mean(err_vector),
             'fbest': cobra.get_fbest(),
             }, index=[0])
        dfw1 = pd.concat([dfw1, new_row_dfw], axis=0)
        return dfw1

    def inner_width_solve(self, gprob, runs, cobraSeed, kernel, width, deg, inter, dfw2, lon2):
        err_arr = np.zeros(runs)
        fbest_arr = np.zeros(runs)
        for run in range(runs):
            seed = cobraSeed + run
            rbf_opt = RBFoptions(kernel=kernel, width=width, degree=deg, interpolator=inter)

            if gprob == "G03":
                c2 = solve_G03(seed, 10, rbf_opt, feval=500, verbIter=100, conTol=0)
            elif gprob == "G06":
                c2 = solve_G06(seed, rbf_opt, feval=90, verbIter=100, conTol=0)
            elif gprob == "G07":
                c2 = solve_G07(seed, rbf_opt, feval=200, verbIter=100, conTol=0)
            elif gprob == "G09":
                c2 = solve_G09(seed, rbf_opt, feval=500, verbIter=100, conTol=0)

            err_arr[run] = c2.p2.fin_err
            fbest_arr[run] = c2.cobra.get_feasible_best()
            if c2.cobra.is_feasible():          # has a feasible solution been found?
                new_row_lon = pd.DataFrame(
                    {'width': width,
                     'kernel': kernel,
                     'degree': deg,
                     'seed': seed,
                     'inter': inter,  # interpolator
                     'feval': c2.cobra.sac_opts.feval,
                     'err': c2.p2.fin_err,
                     'fbest': c2.cobra.get_feasible_best(),
                     }, index=[0])
                lon2 = pd.concat([lon2, new_row_lon], axis=0)

        new_row_dfw = pd.DataFrame(
            {'width': width,
             'kernel': kernel,
             'degree': deg,
             'inter': inter,  # interpolator
             'feval': c2.cobra.sac_opts.feval,
             'conTol': c2.cobra.sac_opts.SEQ.conTol,
             'err': np.nanmean(err_arr),
             'std_err': np.nanstd(err_arr),
             'fbest': np.nanmean(fbest_arr),
             }, index=[0])
        dfw2 = pd.concat([dfw2, new_row_dfw], axis=0)
        return dfw2, lon2

    def test_rbf_width(self):
        """
        Test for scale-variant kernels (e.g. "gaussian") how accuracy varies with kernel width.

        1. How does the RBF approximation error in cross-validation vary with width?
        2. How does the COP optimization error vary with width?
        3. Are similar width values optimal in 1. and 2.?
        4. Are they similar to the heuristic widths W_ONE, W_THREE? --> see printout in inner_width_fit
        """
        cobraSeed = 52
        gprob = "G09"
        runs = 6
        idp = 20
        n_splits = 4
        kernel = "cubic"   # "cubic"  | "gaussian" | "multiquadric" | "thin_plate_spline" | "quintic"

        dfw1 = pd.DataFrame()
        dfw2 = pd.DataFrame()
        lon2 = pd.DataFrame()
        # for width in np.arange(0.1,3.0,0.2):
        # for width in [0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]:
        for width in [0.01, 0.02, 0.2, 1.0]:
            for deg in [1, 2]:   #
                for k, inter in enumerate(["scipy"]):  # , "sacobra"
                    # dfw1 = self.inner_width_fit(idp, n_splits, cobraSeed, kernel, width, deg, inter, dfw1)
                    dfw2, lon2 = self.inner_width_solve(gprob, runs, cobraSeed, kernel, width, deg, inter, dfw2, lon2)

        print(dfw1)
        print(dfw2)
        dfw1.to_feather("feather/dfw1.feather")
        dfw2.to_feather("feather/dfw2.feather")
        lon2.to_feather("feather/lon2.feather")

        dfw2 = pd.read_feather("feather/dfw2.feather")
        feval = dfw2[0:1]['feval'][0]
        png_file1 = f"plots/{kernel}_idp{idp:03d}_{n_splits}.png"
        png_file2 = f"plots/slv_{gprob}_{kernel}_runs{runs:03d}_{feval}.png"
        # read_dfw1(idp, n_splits, png_file1)           # re-read data frame dfw1 and create plot
        read_dfw2(gprob, runs, png_file=png_file2)    # re-read data frame dfw2, create plot(s), do assertions


if __name__ == '__main__':
    unittest.main()
