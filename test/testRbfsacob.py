import time
import unittest
import numpy as np
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
from testRbfModel import TestRbfModel       # for my_rng2

verb = 1

def fn(x):
    # return ((x[:, 0] - 10) ** 3) + ((x[:, 1] - 20) ** 3)
    return x[:, 0] * 2 + x[:, 1] * 3


class TestRbfSacob(unittest.TestCase):
    """
    Test cases for class RBFsacob, the SACOBRA-internal implementation of RBF models
    """

    def test_svd_inv(self):
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

    def test_func_fn(self):
        """
            test whether function fn is RBF-modeled (cubic, 10/100 observations, with/without polynomial tail)
            the same way in RBFsacob and in SciPy's RBFInterpolator. Uses RNG my_rng2 from testRbfModel (avoid cycles!)
        """
        trm = TestRbfModel()
        d = 2
        ngrid = 100
        for nobs in [10, 100]:       #
            for deg in [-1, 1, 2]:
                trm.val = 24  # setting for my_rng2
                print(f"\n[test_func_fn] started with d = {d}, deg = {deg}, nobs = {nobs}")
                xobs = trm.my_rng2(nobs, d)  # better than my_rng: avoids cycles
                yobs = fn(xobs)
                xgrid = np.mgrid[-1:1:ngrid * 1j, -1:1:ngrid * 1j]
                xflat = xgrid.reshape(d, -1).T
                # print(xflat.shape)  # (ngrid**d, d)
                start = time.perf_counter_ns()
                rbf_model = RBFmodel(xobs, yobs,  interpolator="sacobra", kernel="cubic", degree=deg,
                                     rho=0)
                t1 = time.perf_counter_ns()
                time_sacob_mod = t1 - start
                yflat = rbf_model(xflat)            # most time (1 sec) is spent here (!)
                t2 = time.perf_counter_ns()
                time_sacob_prd = t2 - t1
                ygrid = yflat.reshape(ngrid, ngrid)
                print(f"degree = {deg}: time_sacob (mod/prd)= ({time_sacob_mod*1e-6:.6f},{time_sacob_prd*1e-6:.6f}) [ms]")

                ytrue = fn(xflat).reshape(ngrid, ngrid)
                delta = ytrue - ygrid
                print(f"avg abs |ytrue - ygrid| = {np.mean(np.abs(delta)):.4e}")

                L = ngrid // 2
                ytest = ygrid[50:60, L - 1]

                np.set_printoptions(7)
                #print(ytest)

                # Test numerical equivalence with SciPy implementation.
                #
                start = time.perf_counter_ns()
                sci_model = RBFmodel(xobs, yobs,  interpolator="scipy", kernel="cubic", degree=deg,
                                     rho=0)
                t1 = time.perf_counter_ns()
                time_scipy_mod = t1 - start
                sflat = sci_model(xflat)
                t2 = time.perf_counter_ns()
                time_scipy_prd = t2 - t1
                sgrid = sflat.reshape(ngrid, ngrid)
                delta = ytrue - sgrid
                print(f"avg abs |ytrue - sgrid| = {np.mean(np.abs(delta)):.4e}")
                stest = sgrid[50:60, L - 1]
                print(f"degree = {deg}: time_scipy (mod/prd)= ({time_scipy_mod*1e-6:.6f},{time_scipy_prd*1e-6:.6f}) [ms]")
                delta = ytest - stest
                print(f"avg |ytest - stest| = {np.mean(np.abs(delta)):.4e}")
                assert np.allclose(ytest, stest, rtol=9e-4)

                # Test numerical equivalence with R implementation.
                # Requires fn(x) to return    x[:, 0] * 2 + x[:, 1] * 3.
                # These are the values generated with demo-rbf3.R and my_rng2 (avoid cycles!) on the R-side:
                EQUIV_R = True
                if EQUIV_R and deg <= 1:
                    if nobs == 10:
                        if deg == -1:    # ptail=F in R
                            ytest_from_R = np.array([ 0.0354658,  0.0771876,  0.1187900,  0.1602633,  0.2015979,
                                                      0.2427840,  0.2838119,  0.3246720,  0.3653547,  0.4058505])
                        elif deg == 1:  # ptail=T in R
                            ytest_from_R = np.array([-0.0101010,  0.0303030,  0.0707071,  0.1111111,  0.1515152,
                                                      0.1919192,  0.2323232,  0.2727273,  0.3131313,  0.3535354])
                        else:
                            raise NotImplementedError(f"No data from the R side for nobs={nobs} and deg={deg}")
                    elif nobs == 100:
                        if deg == -1:    # ptail=F in R
                            ytest_from_R = np.array([-0.0100817,  0.0303251,  0.0707322,  0.1111393,  0.1515466,
                                                      0.1919538,  0.2323607,  0.2727671,  0.3131729,  0.3535776])
                        elif deg == 1:  # ptail=T in R
                            ytest_from_R = np.array([-0.0101010,  0.0303030,  0.0707071,  0.1111111,  0.1515152,
                                                      0.1919192,  0.2323232,  0.2727273,  0.3131313,  0.3535354])
                        else:
                            raise NotImplementedError(f"No data from the R side for nobs={nobs} and deg={deg}")
                    else:
                        raise NotImplementedError(f"No data from the R side for nobs={nobs}")

                    delta = ytest - ytest_from_R
                    print(f"avg |ytest - ytest_from_R| = {np.mean(np.abs(delta)):.4e}")
                    # assert np.allclose(ytest, ytest_from_R, rtol=1e-4)

                print("[test_func_fn passed]")

    def test_Gprob_RBF(self):
        """
        Test both RBF interpolators "scipy" and "sacobra" on different RBF problems (G06, G07 or G09) and measure mean
        error and total RBF model build (__init__) and total RBF predict (__call__) time.
        """

        def solve_G06(cobraSeed, rbf_opt, feval=40, verbIter=10, conTol=0):  # conTol=0 | 1e-7
            print(f"Starting solve_G06({cobraSeed}) ...")
            G06 = GCOP("G06")

            cobra = CobraInitializer(G06.x0, G06.fn, G06.name, G06.lower, G06.upper, G06.is_equ,
                                     solu=G06.solu,
                                     s_opts=SACoptions(verbose=verb, verboseIter=verbIter, feval=feval,
                                                       cobraSeed=cobraSeed,
                                                       ID=IDoptions(initDesign="RAND_REP", initDesPoints=6),
                                                       RBF=rbf_opt,
                                                       SEQ=SEQoptions(finalEpsXiZero=True, conTol=conTol)))

            c2 = CobraPhaseII(cobra)
            c2.start()

            # show_error_plot(cobra, G06)

            fin_err = np.array(cobra.get_fbest() - G06.fbest)
            c2.p2.fin_err = fin_err
            print(f"final err: {fin_err}")
            # c2.p2.fe_thresh = 5e-6    # this is for s_opts.SEQ.finalEpsXiZero=False
            c2.p2.fe_thresh = 5e-8  # this is for s_opts.SEQ.finalEpsXiZero=True and s_opts.SEQ.conTol=1e-7
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

            c2 = CobraPhaseII(cobra).start()

            fin_err = np.array(cobra.get_fbest() - G07.fbest)
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

            c2 = CobraPhaseII(cobra).start()

            fin_err = np.array(cobra.get_fbest() - G09.fbest)
            c2.p2.fin_err = fin_err
            print(f"final err: {fin_err}")
            c2.p2.fe_thresh = 5e-02
            c2.p2.dim = G09.dimension
            print(G09.fbest)
            print(cobra.get_fbest())
            c2.p2.conTol = conTol
            return c2

        cobraSeed=52
        runs = 1
        errs = np.zeros((2,runs))
        time_init = np.zeros(2)
        time_call = np.zeros(2)
        for run in range(runs):
            for k, inter in enumerate(["scipy", "sacobra"]):
                rbf_opt = RBFoptions(degree=2, interpolator=inter)
                # c2 = solve_G06(cobraSeed+run, rbf_opt, feval=90, verbIter=100, conTol=0)
                c2 = solve_G07(cobraSeed + run, rbf_opt, feval=200, verbIter=100, conTol=0)
                # c2 = solve_G09(cobraSeed + run, rbf_opt, feval=200, verbIter=100, conTol=0)
                errs[k,run] = c2.p2.fin_err
                time_init[k] += c2.p2.time_init
                time_call[k] += c2.p2.time_call

        print(errs)
        print("mean err [scipy, sacobra]")
        print(np.mean(errs, axis=1))
        print("time_init [scipy, sacobra]")
        print(f"{time_init}    {time_init[1]/time_init[0]:.2f}")
        print("time_call [scipy, sacobra]")
        print(f"{time_call}    {time_call[1]/time_call[0]:.2f}")


if __name__ == '__main__':
    unittest.main()
