import time
import unittest
import numpy as np
import pandas as pd
import nlopt
from scipy.interpolate import RBFInterpolator
from scipy.optimize import fmin_l_bfgs_b, minimize
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

def concat(a, b):
    return np.concatenate((a, b), axis=None)


def fn(x):
    return ((x[:, 0] - 10) ** 3) + ((x[:, 1] - 20) ** 3)
    # return x[:, 0] * 2 + x[:, 1] * 3


class TestRbfSacob(unittest.TestCase):

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
        for nobs in [10]:       # , 100
            for deg in [-1, 1, 2]:
                self.val = 24  # setting for my_rng2
                print(f"\n[test_func_fn] started with d = {d}, deg = {deg}, nobs = {nobs}")
                xobs = trm.my_rng2(nobs, d)  # better: avoids cycles
                yobs = fn(xobs)
                xgrid = np.mgrid[-1:1:ngrid * 1j, -1:1:ngrid * 1j]
                xflat = xgrid.reshape(d, -1).T
                # print(xflat.shape)  # (ngrid**d, d)
                start = time.perf_counter()
                rbf_model = RBFmodel(xobs, yobs,  interpolator="sacobra", kernel="cubic", degree=deg,
                                     rho=0)
                yflat = rbf_model(xflat)            # most time (1 sec) is spent here (!)
                ygrid = yflat.reshape(ngrid, ngrid)
                time_sacob = time.perf_counter() - start
                print(f"degree = {deg}: time_sacob = {time_sacob}")

                ytrue = fn(xflat).reshape(ngrid, ngrid)
                delta = ytrue - ygrid
                print(f"avg abs |ytrue - ygrid| = {np.mean(np.abs(delta)):.4e}")

                L = ngrid // 2
                ytest = ygrid[50:60, L - 1]

                np.set_printoptions(7)
                #print(ytest)

                # Test numerical equivalence with SciPy implementation.
                #
                start = time.perf_counter()
                sci_model = RBFmodel(xobs, yobs,  interpolator="scipy", kernel="cubic", degree=deg,
                                     rho=0)
                sflat = sci_model(xflat)
                sgrid = sflat.reshape(ngrid, ngrid)
                delta = ytrue - sgrid
                print(f"avg abs |ytrue - sgrid| = {np.mean(np.abs(delta)):.4e}")
                stest = sgrid[50:60, L - 1]
                time_scipy = time.perf_counter() - start
                print(f"degree = {deg}: time_scipy = {time_scipy}")
                delta = ytest - stest
                print(f"avg |ytest - stest| = {np.mean(np.abs(delta)):.4e}")
                # assert np.allclose(ytest, stest, rtol=1e-4)


                # Test numerical equivalence with R implementation.
                # These are the values generated with demo-rbf3.R and my_rng2 (avoid cycles!) on the R-side:
                EQUIV_R = False
                if EQUIV_R:
                    if nobs == 10:
                        if deg == 0:    # ptail=F in R
                            ytest_from_R = np.array([ 0.0354658,  0.0771876,  0.1187900,  0.1602633,  0.2015979,
                                                      0.2427840,  0.2838119,  0.3246720,  0.3653547,  0.4058505])
                        elif deg == 1:  # ptail=T in R
                            ytest_from_R = np.array([-0.0101010,  0.0303030,  0.0707071,  0.1111111,  0.1515152,
                                                      0.1919192,  0.2323232,  0.2727273,  0.3131313,  0.3535354])
                        else:
                            raise NotImplementedError(f"No data from the R side for nobs={nobs} and deg={deg}")
                    elif nobs == 100:
                        if deg == 0:    # ptail=F in R
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

    def solve_G06(self, cobraSeed, rbf_opt, feval=40, verbIter=10, conTol=0):        # conTol=0 | 1e-7
        """ Test whether COP G06 has statistical equivalent results to the R side with squares=T, if we set on the
            Python side RBF.degree=2 (which is similar, but not the same).

            We test that the median of 15 final errors is < 5e-6, which is statistically equivalent to the R side
            (see ex_COP.R, function multi_G06)
        """
        print(f"Starting solve_G06({cobraSeed}) ...")
        G06 = GCOP("G06")

        cobra = CobraInitializer(G06.x0, G06.fn, G06.name, G06.lower, G06.upper, G06.is_equ,
                                 solu=G06.solu,
                                 s_opts=SACoptions(verbose=verb, verboseIter=verbIter, feval=feval, cobraSeed=cobraSeed,
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
        c2.p2.fe_thresh = 5e-8      # this is for s_opts.SEQ.finalEpsXiZero=True and s_opts.SEQ.conTol=1e-7
        c2.p2.dim = G06.dimension
        c2.p2.conTol = conTol
        return c2

    def solve_G07(self, cobraSeed, rbf_opt, feval=180, verbIter=10, conTol=0):       # conTol=0 | 1e-7
        print(f"Starting solve_G07({cobraSeed}) ...")
        G07 = GCOP("G07")
        idp = 11*12//2

        cobra = CobraInitializer(G07.x0, G07.fn, G07.name, G07.lower, G07.upper, G07.is_equ,
                                 solu=G07.solu,
                                 s_opts=SACoptions(verbose=verb, verboseIter=verbIter, feval=feval, cobraSeed=cobraSeed,
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

    def solve_G09(self, cobraSeed, rbf_opt, feval=500, verbIter=50, conTol=1e-7):        # conTol=0 | 1e-7
        print(f"Starting solve_G09({cobraSeed}) ...")
        G09 = GCOP("G09")
        idp = 10*11//2
        # G09.x0 = G09.solu + 0.01

        cobra = CobraInitializer(G09.x0, G09.fn, G09.name, G09.lower, G09.upper, G09.is_equ,
                                 solu=G09.solu,
                                 s_opts=SACoptions(verbose=verb, verboseIter=verbIter, feval=feval, cobraSeed=cobraSeed,
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

    def test_Gprob_RBF(self):
        """
        Test both RBF interpolators "scipy" and "sacobra" on different RBF problems (G06, G07 or G09) and measure mean
        error and total RBF model build (__init__) and total RBF predict (__call__) time.
        """
        cobraSeed=52
        runs = 5
        errs = np.zeros((2,runs))
        time_init = np.zeros(2)
        time_call = np.zeros(2)
        for run in range(runs):
            for k, inter in enumerate(["scipy", "sacobra"]):
                rbf_opt = RBFoptions(degree=2, interpolator=inter)
                # c2 = self.solve_G06(cobraSeed+run, rbf_opt, feval=90, verbIter=100, conTol=0)
                c2 = self.solve_G07(cobraSeed + run, rbf_opt, feval=200, verbIter=100, conTol=0)
                # c2 = self.solve_G09(cobraSeed + run, rbf_opt, feval=200, verbIter=100, conTol=0)
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


    def solve_G03(self, cobraSeed, rbf_opt, dimension=8, feval=150, verbIter=10, conTol=0):  # conTol=0 | 1e-7
        print(f"Starting solve_G03({cobraSeed}, dim={dimension}, ...) ...")
        G03 = GCOP("G03", dimension)

        equ = EQUoptions(muDec=1.6, muFinal=1e-7, refinePrint=False, refineAlgo="L-BFGS-B")  # "L-BFGS-B COBYLA"
        # x0 = G03.x0  # None --> a random x0 will be set
        x0 = np.tile([0.5,0.3], dimension//2)
        if dimension % 2 == 1: x0 = np.append(x0, 0.5)
        cobra = CobraInitializer(x0, G03.fn, G03.name, G03.lower, G03.upper, G03.is_equ,
                                 solu=G03.solu,
                                 s_opts=SACoptions(verbose=verb, verboseIter=verbIter, feval=feval, cobraSeed=cobraSeed,
                                                   ID=IDoptions(initDesign="RAND_REP", initDesPoints=2*dimension+1, rescale=True),
                                                   RBF=rbf_opt,
                                                   EQU=equ,
                                                   SEQ=SEQoptions(finalEpsXiZero=True, conTol=conTol)))
        print(f"idp = {cobra.sac_opts.ID.initDesPoints}")
        c2 = CobraPhaseII(cobra).start()

        fin_err = np.array(cobra.get_fbest() - G03.fbest)
        print(f"final err: {fin_err}")
        c2.p2.fin_err = fin_err
        c2.p2.fe_thresh = 1e-9
        c2.p2.dim = G03.dimension
        c2.p2.conTol = conTol
        return c2

    def test_G03(self):
        """
        Test for the G03 Python problem
        """
        cobraSeed=42
        runs = 1
        errs = np.zeros((2,runs))
        for run in range(runs):
            for k, inter in enumerate(["sacobra"]):
                rbf_opt = RBFoptions(degree=1.5, interpolator=inter)
                c2 = self.solve_G03(cobraSeed + run, rbf_opt, dimension=10, feval=300, verbIter=50, conTol=0)
                errs[k,run] = c2.p2.fin_err

        print(errs)
        print("mean err [sacobra]")
        print(np.mean(errs, axis=1))

    def test_lfbgs(self):
        dimension = 7
        cobraSeed = 42
        G03 = GCOP("G03", dimension)
        # equ_ind is the index to all equality constraints in p2.constraintSurrogates:
        self.equ_ind = np.flatnonzero(G03.is_equ)
        # ine_ind is the index to all inequality constraints in p2.constraintSurrogates:
        self.ine_ind = np.flatnonzero(G03.is_equ is False)

        rbf_opt = RBFoptions(degree=1.5, interpolator="sacobra")
        equ = EQUoptions(muDec=1.6, muFinal=1e-7, refinePrint=False, refineAlgo="L-BFGS-B")  # "L-BFGS-B COBYLA"
        # x0 = G03.x0  # None --> a random x0 will be set
        x0 = np.tile([0.5,0.3], dimension//2)
        if dimension % 2 == 1: x0 = np.append(x0, 0.5)

        # BFGS_METH = 0: 'fmin_l_bfgs_b' (from scipy.optimize),
        # BFGS_METH = 1: 'minimize' (from scipy.optimize) with method 'L-BFGS-B',
        df_dist = pd.DataFrame()
        for BFGS_METH in range(1+1):
            print(f"Starting test_lfbgs with seed={cobraSeed}, dim={dimension}, BFGS_METH={BFGS_METH}, ...) ...")
            cobra = CobraInitializer(x0, G03.fn, G03.name, G03.lower, G03.upper, G03.is_equ,
                                     solu=G03.solu,
                                     s_opts=SACoptions(verbose=verb, verboseIter=10, feval=100, cobraSeed=cobraSeed,
                                                       ID=IDoptions(initDesign="RAND_REP", initDesPoints=2*dimension+1, rescale=True),
                                                       RBF=rbf_opt,
                                                       EQU=equ,
                                                       SEQ=SEQoptions(finalEpsXiZero=True, conTol=0)))
            print(f"idp = {cobra.sac_opts.ID.initDesPoints}")
            s_res = cobra.sac_res
            s_opts = cobra.sac_opts

            xNew = np.array([-0.24730173, -1.,  0.63872554, -1.,  0.01828732, -1., 1.])
            xNew = np.maximum(-0.9,xNew)
            xNew = np.minimum(+0.9,xNew)
            xNew = np.repeat(-0.2,dimension)  # dim=7, all values a \in [-0.99,0.05] find the minimum, but a=-1.0 or a>0.05 does not, they all go to the wrong min [-1,-1,...]

            con_mdl = RBFmodel(s_res['A'], s_res['Gres'], interpolator=s_opts.RBF.interpolator,
                            kernel=s_opts.RBF.model,
                            degree=s_opts.RBF.degree, rho=s_opts.RBF.rho)

            def myf(x, grad):
                # conR = s_res['fn'](x)[1:]
                conR = con_mdl(x)
                return np.sum(concat(np.maximum(0, conR[self.ine_ind]) ** 2, conR[self.equ_ind] ** 2))

            def myf2(x):
                return myf(x, None)

            def approx_fprime(x: np.ndarray):
                eps = 1e-7
                app_grad = np.zeros(x.size)
                for i in range(x.size):
                    xp = x.copy()
                    xn = x.copy()
                    xp[i] += eps
                    xn[i] -= eps
                    app_grad[i] = (myf2(xp) - myf2(xn))/(2*eps)
                return app_grad

            # lbfgs_bounds = zip(s_res['lower'].tolist(), s_res['upper'].tolist())
            lbfgs_bounds = [(s_res['lower'][i],s_res['upper'][i]) for i in range(s_res['upper'].size)]
            xNewVals = np.arange(-1,1.05,0.1)
            for nplus in range(0,2+1):
                for nminus in range(2,2+1):
                    for xNewVal in xNewVals:
                        xNew = np.repeat(xNewVal, dimension)
                        xNew[0:nminus] = -1
                        xNew[(dimension-nplus):dimension] = +1
                        if nplus >= 1:
                            dummy = 0
                        # xNew = np.array([-0.24730173, -1., 0.63872554, -1., 0.01828732, -1., 1.])
                        if BFGS_METH == 0:
                            x_opt, f_opt, info = fmin_l_bfgs_b(myf2, x0=xNew, fprime=None,      # fprime = None | approx_fprime
                                                               bounds=list(lbfgs_bounds), maxiter=s_opts.EQU.refineMaxit,
                                                               factr=10, approx_grad=True)
                            self.refi = {'x': x_opt,
                                         'minf': f_opt,
                                         'res_code': info['warnflag'],   # the return code
                                         'res_msg': info['task'],
                                         'feMax': info['funcalls'],
                                         }
                        elif BFGS_METH == 1:
                            res = minimize(myf2, xNew, method='L-BFGS-B', bounds=lbfgs_bounds)
                            self.refi = {'x': res.x,
                                         'minf': res.fun,
                                         'res_code': res.status,    # the return code
                                         'res_msg': res.message,
                                         'feMax': res.nfev,
                                         }
                        elif BFGS_METH == 2:   # COBYLA
                            opt = nlopt.opt(nlopt.LN_COBYLA, xNew.size)
                            opt.set_lower_bounds(s_res['lower'])
                            opt.set_upper_bounds(s_res['upper'])
                            opt.set_min_objective(myf)
                            opt.set_xtol_rel(-1)  # this may give an NLopt Roundoff-Limited error in opt.optimize
                            # opt.set_xtol_rel(1e-20)  #  (1e-8) #
                            opt.set_maxeval(s_opts.EQU.refineMaxit)
                            # opt.set_exceptions_enabled(False)       # not found!

                            try:
                                x = opt.optimize(xNew)
                            except nlopt.RoundoffLimited:
                                print(f"WARNING: refine:  nlopt.RoundoffLimited exception "
                                      f"(result code {opt.last_optimize_result()})")
                                x = xNew.copy()

                            minf = opt.last_optimum_value()

                            self.refi = {'x': x,
                                         'minf': minf,
                                         'res_code': opt.last_optimize_result(),  # the return code
                                         'feMax': opt.get_numevals(),
                                         }

                        print(self.refi)
                        if dimension == 7:
                            desired_refine_solu = np.repeat(-0.2440716973266758, 7)
                        dist_solu = np.sqrt(np.sum((desired_refine_solu-self.refi['x'])**2))
                        new_row_df = pd.DataFrame({
                            'xNewVal': xNewVal,
                            'nplus': nplus,
                            'nminus': nminus,
                            'BFGS_METH': BFGS_METH,
                            'dist_solu': dist_solu,
                            'myf2': myf2(xNew),
                            'myfres': myf2(self.refi['x']),
                        }, index=[0])
                        df_dist = pd.concat([df_dist, new_row_df], axis=0)
            myf2_from_Py = df_dist['myf2'][0:21]
            myf2_from_R = np.array([1.0000000000, 0.9653062500, 0.8649000000, 0.7098062500, 0.5184000000, 0.3164062500, 0.1369000000, 0.0203062500, 0.0144000000, 0.1743062500, 0.5625000000, 1.2488062500, 2.3104000000, 3.8318062500, 5.9049000000, 8.6289062500, 12.1104000000, 16.4633062500, 21.8089000000, 28.2758062500, 36.0000000000])
            if all(df_dist['nminus'][0:21] == 0) and all(df_dist['nplus'][0:21] == 0):
                assert np.allclose(myf2_from_R,myf2_from_Py)
                print("Assertion myf2 valid.")
        dummy = 0

if __name__ == '__main__':
    unittest.main()
