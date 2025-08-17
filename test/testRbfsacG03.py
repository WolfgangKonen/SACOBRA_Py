import time
import unittest
import numpy as np
import pandas as pd
import nlopt
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

verb = 1

def concat(a, b):
    return np.concatenate((a, b), axis=None)


class TestRbfSacG03(unittest.TestCase):

    def solve_G03(self, cobraSeed, rbf_opt, dimension=8, feval=150, verbIter=10, conTol=0):  # conTol=0 | 1e-7
        print(f"Starting solve_G03({cobraSeed}, dim={dimension}, ...) ...")
        G03 = GCOP("G03", dimension)

        equ = EQUoptions(muDec=1.6, muFinal=1e-7, refinePrint=False, refineAlgo="BFGS_1")  # "L-BFGS-B COBYLA"
        # x0 = G03.x0  # None --> a random x0 will be set
        x0 = np.tile([0.5,0.3], dimension//2)
        if dimension % 2 == 1: x0 = np.append(x0, 0.5)
        cobra = CobraInitializer(x0, G03.fn, G03.name, G03.lower, G03.upper, G03.is_equ,
                                 solu=G03.solu,
                                 s_opts=SACoptions(verbose=verb, verboseIter=verbIter, feval=feval, cobraSeed=cobraSeed,
                                                   ID=IDoptions(initDesign="RAND_REP", initDesPoints=2*dimension+1, rescale=True),
                                                   RBF=rbf_opt,
                                                   EQU=equ,
                                                   SEQ=SEQoptions(finalEpsXiZero=True, trueFuncForSurrogates=True, conTol=conTol)))
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
        cobraSeed=54
        runs = 10
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

            con_mdl = RBFmodel(s_res['A'], s_res['Gres'], s_opts.RBF)

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
