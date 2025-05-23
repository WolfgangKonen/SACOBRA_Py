import nlopt
import numpy as np
import pandas as pd
from scipy.optimize import fmin_l_bfgs_b
from scipy.optimize import minimize

from cobraInit import CobraInitializer
from phase2Vars import Phase2Vars
from innerFuncs import plogReverse


def concat(a, b):
    return np.concatenate((a, b), axis=None)


class EvaluatorReal:
    """
        Evaluate the new infill point (found via surrogates) on the real functions.
    """
    def __init__(self, cobra: CobraInitializer, p2: Phase2Vars):
        s_opts = cobra.sac_opts
        nConstraints = cobra.sac_res['nConstraints']
        self.CONSTRAINED = nConstraints > 0
        if p2.num == s_opts.ID.initDesPoints:
            self.predY = np.repeat(np.nan, s_opts.ID.initDesPoints)
            self.predVal = np.repeat(np.nan, s_opts.ID.initDesPoints)
            if self.CONSTRAINED:
                # matrix to store predicted constraint values (TODO: is it needed if we have Gres?)
                self.predC = np.zeros((s_opts.ID.initDesPoints, nConstraints))
                # feasibility of initial design:
                self.feas = np.apply_along_axis(lambda x: np.max(x) <= 0, axis=1, arr=cobra.sac_res['Gres'])

            self.feasPred = np.repeat(False, s_opts.ID.initDesPoints)
            self.optimConv = np.repeat(1, s_opts.ID.initDesPoints)  # vector to store optimizer convergence
            self.optimTime = np.repeat(0, s_opts.ID.initDesPoints)
            self.feval = np.repeat(np.nan, s_opts.ID.initDesPoints)  # vector to store function evaluations on surrogate

        else:
            df = cobra.df
            self.predY = df.predY
            self.predVal = df.predVal
            self.predC = cobra.sac_res['predC']
            self.feas = df.feasible
            self.feasPred = df.feasPred
            self.optimConv = df.optimConv
            self.optimTime = df.optimTime
            self.feval = df.FEval

        self.is_equ = cobra.sac_res['is_equ']
        # equ_ind is the index to all equality constraints in p2.constraintSurrogates:
        self.equ_ind = np.flatnonzero(self.is_equ)
        # ine_ind is the index to all inequality constraints in p2.constraintSurrogates:
        self.ine_ind = np.flatnonzero(self.is_equ is False)
        # DON'T change here to 'self.is_equ is False' as the PEP hint suggest --> strange error in NLopt (!)

        # add fields that will be filled later (e.g. in update, equ_refine, equ_num_max_viol):
        self.x_0 = None           # the state before refine
        self.x_1 = None           # the state after refine
        self.xNew = None
        self.xNewEval = None
        self.refi = None          # dict with results from refine step optimization
        self.newNumViol = None    # number of violated constraints for xNew (equ constraints: artificial with margin mu)
        self.newMaxViol = None
        self.trueNumViol = None   # number of violated constraints for xNew (equ constraints: no margin mu)
        self.trueMaxViol = None
        self.newNumPred = None    # number of violated surrogates for xNew (equ constraints: artificial with margin mu)
        self.nv_trueA = None
        self.nv_trueB = None
        self.nv_conA = None
        self.nv_conB = None

        self.state = "initialized"

    def update(self, xNew: np.ndarray, cobra: CobraInitializer, p2: Phase2Vars, currentEps,
               fitnessSurrogate=None, f_value=None):
        """
        Evaluate ``xNew`` on the real functions + do refine step (if ``cobra.sac_opts.EQU.active`` and
        ''self.state == "optimized"``) + calculate various feasibility indicators (numViol, maxViol, ...)

        :param xNew:    the result vector from sequential optimization on surrogates (the new best x)
        :param cobra:
        :param p2:
        :param currentEps:  artificial margin 'mu' for equality constraints
        :param fitnessSurrogate:
        :param f_value:
        :return: nothing, but many elements of self are updated: xNew, xNewEval, x0, x1, ...
        """
        # update() corresponds to evalReal() in R, which is called in three places:
        # after seq.opt, after repair and after TR-step

        if fitnessSurrogate is None:
            fitnessSurrogate = p2.fitnessSurrogate
        if f_value is None:
            f_value = p2.opt_res['minf']
        # fn = cobra$fn
        self.xNew = xNew
        self.xNew = np.maximum(self.xNew, cobra.sac_res['lower'])
        self.xNew = np.minimum(self.xNew, cobra.sac_res['upper'])

        if cobra.sac_opts.EQU.active:
            if cobra.sac_opts.EQU.refine and self.state == "optimized":
                # do refine step only after surrogate optimizer (not after repair or TR)
                self.equ_refine(cobra, p2, currentEps)

        if cobra.sac_opts.SEQ.trueFuncForSurrogates:
            newPredY = cobra.sac_res['fn'](self.xNew)[0]
        else:
            newPredY = getPredY0(self.xNew, fitnessSurrogate, p2)
        self.predY = concat(self.predY, newPredY)  # bug fix: now predY is the fitness surrogate value /WK/
        self.predVal = concat(self.predVal, f_value)  # fitness + penalty (in case of NMKB et al.) /WK/
        self.feval = concat(self.feval, p2.opt_res['feval'])
        self.optimConv = concat(self.optimConv, p2.opt_res['res_code'])
        self.optimTime = concat(self.optimTime, p2.opt_res['time_ms'])
        newPredC = np.array([])
        if self.CONSTRAINED:
            newPredC = p2.constraintSurrogates(self.xNew)
            self.predC = np.vstack((self.predC, newPredC))
            newPredC = newPredC[0]    # why [0]? - constraintSurrogates returns a
            # (1,nC)-matrix, but we want a (nc,)-vector here (nC = nConstraints)

        self.xNewEval = cobra.sac_res['fn'](self.xNew)
        # TODO later:
        # if (cobra$CA$active & & cobra$TFlag){
        #     xNewT < -(ev1$xNew-cobra$tCenter) % * %cobra$TM
        #     xNewT < -xNewT+cobra$tCenter
        #     ev1$xNewEvalT < -fn(xNewT)
        # }

        if self.CONSTRAINED:
            if cobra.sac_opts.EQU.active:
                self.equ_num_max_viol(cobra, currentEps, newPredC)
            else:
                self.ine_num_max_viol(cobra, newPredC)

        else:
            self.newNumViol = 0
            self.newMaxViol = 0
            self.trueNumViol = 0
            self.trueMaxViol = 0
            self.feas = concat(self.feas, True)
            self.feasPred = concat(self.feasPred, True)

    def equ_refine(self, cobra: CobraInitializer, p2: Phase2Vars, currentEps):
        """
        Do refine step for equality constraints on the new point stored in self.xNew.

        :param cobra:
        :param p2:
        :param currentEps:
        :return: nothing, but these elements of self are changed: (a) vectors xNew, x_0, x_1; (b) dict refi;
                 (c) numbers nv_conB, nv_conA, nv_trueB, nv_trueA; (d) string state
        """
        s_opts = cobra.sac_opts
        s_res = cobra.sac_res
        # if (cobra$trueMaxViol[cobra$ibest] > cobra$equHandle$equEpsFinal) {
        if True:  # testwise, enforce refine in every step
            # cat("[evalReal] starting refine ...\n")
            #
            # refine step, special for equality constraints:
            # Due to currentEps, xNew will not fulfill the equality constraints exactly.
            # Search with s_opts.EQU.refineAlgo (i.e. COBYLA, BFGS or similar) a point near to xNew which
            # fulfills the equality constraints: Minimize the square sum of s_i(x)**2 where s_i is the surrogate
            # model for the ith constraint. The solution self.refi['x'] from refine replaces xNew.
            #
            # The old version (deprecated): minimize equality violations only
            #
            #       sum ( h_j(x)^2 )
            #
            # The new version: minimize *all* constraint violations (inequalities + equalities):
            #
            #        sum( max(0,g_i(x))^2 )  + sum ( h_j(x)^2 )
            #

            def myf(x, grad):
                conR = p2.constraintSurrogates(x)[0]        # why [0]? - constraintSurrogates
                # returns a (1,nC)-matrix, but we want a (nc,)-vector here (nC = nConstraints)
                return np.sum(concat(np.maximum(0, conR[self.ine_ind]) ** 2, conR[self.equ_ind] ** 2))
                # return np.sum(conR[self.equ_ind] ** 2)     # deprecated

            if s_opts.SEQ.trueFuncForSurrogates:
                def myf(x, grad):
                    conR = s_res['fn'](x)[1:]
                    return np.sum(concat(np.maximum(0, conR[self.ine_ind]) ** 2, conR[self.equ_ind] ** 2))

            if s_opts.EQU.refineAlgo == "L-BFGS-B":
                def myf2(x):
                    return myf(x, None)

                BFGS_METH = 1       # 0: fmin_l_bfgs_b,
                                    # 1: minimize with method 'L-BFGS-B',
                                    # 2: nlopt with method 'nlopt.LD_LBFGS' (does not work)
                lbfgs_bounds = zip(s_res['lower'].tolist(), s_res['upper'].tolist())
                if BFGS_METH == 0:
                    x_opt, f_opt, info = fmin_l_bfgs_b(myf2, x0=self.xNew, bounds=list(lbfgs_bounds),
                                                       maxiter=s_opts.EQU.refineMaxit,
                                                       approx_grad=True)
                    self.refi = {'x': x_opt,
                                 'minf': f_opt,
                                 'res_code': info['warnflag'],    # the return code
                                 'feval': 0,
                                 }
                elif BFGS_METH == 1:
                    res = minimize(myf2, self.xNew, method='L-BFGS-B', bounds=lbfgs_bounds)
                    self.refi = {'x': res.x,
                                 'minf': res.fun,
                                 'res_code': res.message,    # the return code
                                 'feval': res.nfev,
                                 }
                else:  # i.e. BFGS_METH == 2
                    raise RuntimeError(f"BFGS_METH={BFGS_METH} does not work (raises nlopt.runtime_error)")
                    # opt = nlopt.opt(nlopt.LD_LBFGS, self.xNew.size)
                    # opt.set_lower_bounds(s_res['lower'])
                    # opt.set_upper_bounds(s_res['upper'])
                    # opt.set_min_objective(myf)
                    # opt.set_xtol_rel(-1)
                    # opt.set_maxeval(s_opts.EQU.refineMaxit)
                    # try:
                    #     x = opt.optimize(self.xNew)
                    # except nlopt.runtime_error: # nlopt.RoundoffLimited:
                    #     print(f"WARNING: refine [{p2.num}] nlopt.runtime_error exception " # RoundoffLimited
                    #           f"(result code {opt.last_optimize_result()})")
                    #     x = self.xNew.copy()
                    # minf = opt.last_optimum_value()
                    # self.refi = {'x': x,
                    #              'minf': minf,
                    #              'res_code': opt.last_optimize_result(),  # the return code
                    #              'feval': opt.get_numevals(),
                    #              }

                    # # just a debug test: would a G03-specific normalization to sphere surface solve the refine issue?
                    # # Answer: No: It works for d<=7, but fails for d>7, as the other methods.
                    # x = self.xNew.copy()
                    # x = x / np.sqrt(np.sum(x*x))
                    # self.refi = {'x': x,
                    #              'minf': 0,
                    #              'res_code': 0,  # the return code
                    #              'feval': 0}

            else:  # i.e. "COBYLA"
                opt = nlopt.opt(nlopt.LN_COBYLA, self.xNew.size)
                opt.set_lower_bounds(s_res['lower'])
                opt.set_upper_bounds(s_res['upper'])
                opt.set_min_objective(myf)
                opt.set_xtol_rel(-1)    # this may give an NLopt Roundoff-Limited error in opt.optimize
                # opt.set_xtol_rel(1e-20)  #  (1e-8) #
                opt.set_maxeval(s_opts.EQU.refineMaxit)
                # opt.set_exceptions_enabled(False)       # not found!

                try:
                    x = opt.optimize(self.xNew)
                except nlopt.RoundoffLimited:
                    print(f"WARNING: refine [{p2.num}] nlopt.RoundoffLimited exception "
                          f"(result code {opt.last_optimize_result()})")
                    x = self.xNew.copy()

                minf = opt.last_optimum_value()
                # COBYLA with xtol_rel=-1 is the recommended choice. If xtol_rel were missing, the default value would
                # be xtol_rel=1e-4 which causes the optimization to stop when the change in any parameter x_i is smaller
                # than xtol_rel * |x_i|. This would cause the refine mechanism to return too early and to lose solutions
                # while the equality margin is shrinking (e.g. in benchmark problem G13).

                # print(c(cg$convergence, cg$message))
                # -- optimization usually stops with cg$convergence=5 ("... because maxeval (above) was reached")
                self.refi = {'x': x,
                             'minf': minf,
                             'res_code': opt.last_optimize_result(),    # the return code
                             'feval': opt.get_numevals(),
                             }

            # if (cg$convergence == 1) {  # iteration limit maxit has been reached
            # warning(sprintf("Refine step: optim did not converge within maxit=%d iterations. %s",
            # cobra$equHandle$refineMaxit,
            # "Consider to increase cobra$equHandle$refineMaxit"))
            # }
            # if (any( is.na(cg$par))){browser()}
            #
            self.x_0 = self.xNew       # the state before refine
            self.x_1 = self.refi['x']  # the state after refine

            if s_opts.EQU.refinePrint:
                # ------ The following is only debug/diagnostics: ----------------
                # /WK/ The debug-printout of the three lines below shows that optim succeeds in all cases to
                #      turn the equality mismatch in cgbefore, which may be large (>1) in initial iterates,
                #      down to cg$value = 1e-8 or better. And cgtrue, the evaluation of cg$par on the true
                #      equality constraints (not the RBF models) gives usually 1e-8 or better.
                cgbefore = myf(self.x_0, None)
                cgafter = myf(self.x_1, None)
                if self.refi['res_code'] >= 0:
                    assert self.refi['minf'] == cgafter
                conA = p2.constraintSurrogates(self.x_1)[0]
                conA[self.equ_ind] = abs(conA[self.equ_ind])
                aMaxV = np.max(concat(np.maximum(0, conA[self.ine_ind]), conA[self.equ_ind]))
                conTrue = s_res['fn'](self.x_1)[1:]  # true constraints after refine
                conTrue[self.equ_ind] = abs(conTrue[self.equ_ind])
                cgtrue = np.sum(concat(np.maximum(0, conTrue[self.ine_ind]) ** 2, conTrue[self.equ_ind] ** 2))
                trueMaxV = np.max(concat(np.maximum(0, conTrue[self.ine_ind]), conTrue[self.equ_ind]))
                # #### printout, only for debugging
                # /WK/ the name "cg" is just a reminiscence  of initially used optim with method="CG" (conjugate grad)
                print(f"cg-values (before,after,true) = ({cgbefore:.1g}, {cgafter:.1g}, {cgtrue:.1g})  "
                      f"[aMaxV, trueMaxV, mu] = [{aMaxV:.2e}, {trueMaxV:.2e}, {p2.currentEps:.2e}]")
                if aMaxV == 0:
                    dummy = 0
            #
            # this is just more detailed constraint information, which may be inspected in browser:
            if s_opts.SEQ.trueFuncForSurrogates:
                conB = s_res['fn'](self.x_0)[1:]  # true constraints before refine
                conA = s_res['fn'](self.x_1)[1:]  # true constraints after refine
            else:
                conB = p2.constraintSurrogates(self.x_0)   # constraint surrogates before refine
                conA = p2.constraintSurrogates(self.x_1)   # constraint surrogates after refine

            conB = conB.reshape(conB.size)
            conA = conA.reshape(conA.size)
            trueB = s_res['fn'](self.x_0)[1:]
            trueA = s_res['fn'](self.x_1)[1:]
            conB[self.equ_ind] = abs(conB[self.equ_ind]) - currentEps
            conA[self.equ_ind] = abs(conA[self.equ_ind]) - currentEps
            trueB[self.equ_ind] = abs(trueB[self.equ_ind]) - currentEps
            trueA[self.equ_ind] = abs(trueA[self.equ_ind]) - currentEps
            conTol = s_opts.SEQ.conTol
            pos_conB = np.flatnonzero(conB > conTol)
            pos_conA = np.flatnonzero(conA > conTol)
            pos_trueB = np.flatnonzero(trueB > conTol)
            pos_trueA = np.flatnonzero(trueA > conTol)
            self.nv_conB = pos_conB.size  # will be added to df2 in updateSaveCobra
            self.nv_conA = pos_conA.size
            self.nv_trueB = pos_trueB.size
            self.nv_trueA = pos_trueA.size
            check_df = pd.DataFrame({'conB': conB, 'conA': conA,
                                     'trueB': trueB, 'trueA': trueA,
                                     'diffB': np.maximum(0, trueB)-np.maximum(0, conB),
                                     'diffA': np.maximum(0, trueA)-np.maximum(0, conA)
                                     })
            #  ------ end debug/diagnostics ----------------------------------

            self.xNew = self.x_1
            self.state = "refined"

        # equ_refine(self,...) returns in self.xNew a refined solution + several other diagnostic infos on self.xyz
        # which can be retrieved by the caller as p2.ev1.xyz
        # equ_refine(self,...) corresponds to equRefineStep in evalReal.R

    def equ_num_max_viol(self, cobra: CobraInitializer, currentEps, newPredC):
        """
        Calculate self.newNumViol, .newMaxViol, ... for the equality case (sac_opts.EQU.active=True)

        :param cobra:   object of class CobraInitializer
        :param currentEps: current artificial equality margin :math:`\mu`
        :param newPredC: prediction for xNew on constraint surrogates
        :return: nothing, but these elements of self are changed: (a) numbers newNumViol, newMaxViol, newNumPred,
                     trueNumViol, trueMaxViol; (b) vectors feas, feasPred
        """
        s_opts = cobra.sac_opts
        s_res = cobra.sac_res
        conTol = s_opts.SEQ.conTol
        # WK: the new version: we check whether
        #
        #          g_i(x) <= 0,  h_j(x) - currentEps <= 0,    -h_j(x) - currentEps <= 0
        #
        # for the real surrogates and set self.newNumViol to the number of violated constraints.
        # NOTE that temp is also used for self.newMaxViol below.
        temp = s_res['fn'](self.xNew)[1:].copy()
        temp = concat(temp, -temp[self.equ_ind])
        equ2Index = concat(self.equ_ind, s_res['nConstraints'] + np.arange(self.equ_ind.size))
        temp[equ2Index] = temp[equ2Index] - currentEps
        self.newNumViol = np.flatnonzero(temp > conTol).size
        # number of constraint violations for new point  (we changed former thresh 0 changed to conTol)

        # just a debug check:
        if s_opts.EQU.refine:
            if self.state == "refined" or self.state == "optimized":
                assert self.newNumViol == self.nv_trueA
            # If state is not "optimized" at start of evalReal, then the branch that
            # computes nv_trueA will not been executed and the assertion would fail. Otherwise, it should hold.

        self.feas = concat(self.feas, self.newNumViol < 1)

        # WK: brought here currentEps and equ2Index into play as well
        ptemp = concat(newPredC, -newPredC[self.equ_ind])
        ptemp[equ2Index] = ptemp[equ2Index] - currentEps
        self.newNumPred = np.flatnonzero(ptemp > conTol).size  # the same on constraint surrogates
        self.feasPred = concat(self.feasPred, self.newNumPred < 1)

        # WK: changed self.newMaxViol back to hold the artificial constraint max violation (currentEps-
        #     margin for equality constraints). This is one condition for entering repair (cobraPhaseII)
        M = max(0, max(temp))  # maximum violation
        self.newMaxViol = M

        # SB: it is also interesting to observe and save the information about the real maximum violation
        # instead of the maximum distance to the artificial constraints.
        # WK: the difference is that we do not subtract currentEps here
        temp = s_res['fn'](self.xNew)[1:].copy()
        temp[self.equ_ind] = np.abs(temp[self.equ_ind])
        self.trueNumViol = np.flatnonzero(temp > conTol).size
        M = max(0, max(temp * s_res['GRfact']))   # true maximum violation, weighted with GRfact
        # /WK/2025/05/04: this is the version used in R as well: GRfact (see adCon()) is a vector of length
        # nConstraints and tempG.shape = (idp,nConstraints). With the trick "... @ np.diag(GRfact)" we multiply each
        # row of tempG elementwise with GRfact. The violations are weighted with GRfact, then we take the max.
        # The simpler alternative would be:
        #       M = max(0, max(temp))
        # but this is not the way it is in R. (The former remark, that we do not need to multiply with GRfact because
        # it is 1.0 anyhow, is *wrong* because GRfact becomes a vector of length nConstraints in the EQU.active branch)

        if M <= conTol: M = 0
        self.trueMaxViol = M

    def ine_num_max_viol(self, cobra: CobraInitializer, newPredC):
        """
        Calculate self.newNumViol, .newMaxViol, ... for the inequality case (sac_opts.EQU.active=False)

        :param cobra:   object of class CobraInitializer
        :param newPredC: prediction for xNew on constraint surrogates
        :return: nothing, but these elements of self are changed: (a) numbers newNumViol, newMaxViol, newNumPred,
                     trueNumViol, trueMaxViol; (b) vectors feas, feasPred
        """
        conTol = cobra.sac_opts.SEQ.conTol
        # number of constraint violations for new point:
        self.newNumViol = np.flatnonzero(self.xNewEval[1:] > conTol).size
        # the same on constraint surrogates:
        self.newNumPred = np.flatnonzero(newPredC > conTol).size
        self.feas = concat(self.feas, self.newNumViol < 1)
        self.feasPred = concat(self.feasPred, self.newNumPred < 1)

        if (max(0, max(self.xNewEval[1:]))) > conTol:  # maximum violation
            self.newMaxViol = max(0, max(self.xNewEval[1:]))
        else:
            self.newMaxViol = 0

        self.trueNumViol = self.newNumViol
        self.trueMaxViol = self.newMaxViol


def getPredY0(xNew, fitnessSurrogate, p2: Phase2Vars):
    pred_y = fitnessSurrogate(xNew)
    if p2.PLOG[-1]:
        pred_y = plogReverse(pred_y, p2.pshift[-1])
    return pred_y

# getPredY is nothing else than calling getPredY0 with fitnessSurrogate=p2.fitnessSurrogate
