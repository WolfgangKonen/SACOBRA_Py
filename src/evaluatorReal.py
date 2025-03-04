import numpy as np
from cobraInit import CobraInitializer
from phase2Vars import Phase2Vars
from innerFuncs import plogReverse


class EvaluatorReal:
    """
        Evaluate the new infill point (found via surrogates) on the real functions.
    """
    def __init__(self, cobra: CobraInitializer,p2: Phase2Vars):
        s_opts = cobra.sac_opts
        nConstraints = cobra.sac_res['nConstraints']
        self.CONSTRAINED = nConstraints > 0
        self.xNew = None
        if p2.num == s_opts.ID.initDesPoints:
            self.predY = np.repeat(np.nan, s_opts.ID.initDesPoints)
            self.predVal = np.repeat(np.nan, s_opts.ID.initDesPoints)
            if self.CONSTRAINED:
                # matrix to store predicted constraint values (TODO: is it needed if we have Gres?)
                self.predC = np.zeros((s_opts.ID.initDesPoints, nConstraints))
                # feasibility of initial design:
                self.feas = np.apply_along_axis(lambda x: np.max(x) > 0, axis = 1, arr=cobra.sac_res['Gres'])

            self.feasPred = np.repeat(np.nan, s_opts.ID.initDesPoints)
            self.optimConv = np.repeat(1, s_opts.ID.initDesPoints)  # vector to store optimizer convergence
            self.optimTime = np.repeat(0, s_opts.ID.initDesPoints)
            self.feval = np.repeat(np.nan, s_opts.ID.initDesPoints)  # structure to store function evaluations on surrogate

        else:
            df = cobra.df
            self.predY = df.predY
            self.predVal = df.predVal
            self.predC = cobra.sac_res['predC']
            self.feas = df.feasible
            self.feasPred = df.feasPred
            self.optimConv = df.conv
            self.optimTime = df.optimizationTime
            self.feval = df.FEval

        self.state = "initialized"

    def update(self, cobra: CobraInitializer, p2: Phase2Vars, currentEps=0,
               fitnessSurrogate=None, f_value=None):
        """
        The result from sequential optimization on surrogates is the new best x is in ``xNew = self.p2.opt_res['x']``.

        Evaluate ``xNew`` on the real functions + do refine step (if ``cobra.sac_opts.EQU.active``).

        :param cobra:
        :param p2:
        :param currentEps:
        :param fitnessSurrogate:
        :param f_value:
        """
        # update() corresponds to evalReal() in R, which is called after seq.opt, after repair and after TR-step
        def concat(a,b):
            return np.concatenate((a,b), axis=None)

        if fitnessSurrogate is None:
            fitnessSurrogate = p2.fitnessSurrogate
        if f_value is None:
            f_value = p2.opt_res['minf']
        # fn = cobra$fn
        self.xNew = p2.opt_res['x']
        self.xNew = np.maximum(self.xNew, cobra.sac_res['lower'])
        self.xNew = np.minimum(self.xNew, cobra.sac_res['upper'])
        if cobra.sac_opts.EQU.active:
            raise NotImplementedError("[EvaluatorReal] Branch EQU.active (line 70-142 evalReal.R) not yet implemented!")

        if cobra.sac_opts.SEQ.trueFuncForSurrogates:
            newPredY = cobra.sac_res['fn'](self.xNew)[0]
        else:
            newPredY = getPredY0(self.xNew, fitnessSurrogate, p2)
        self.predY = concat(self.predY, newPredY)  # bug fix: now predY is the fitness surrogate value /WK/
        self.predVal = concat(self.predVal, f_value)  # fitness + penalty (in case of NMKB et al.) /WK/
        self.feval = concat(self.feval, p2.opt_res['feval'])
        self.optimConv = concat(self.optimConv, p2.opt_res['res_code'])
        self.optimTime = concat(self.optimTime, p2.opt_res['time_ms'] )
        if self.CONSTRAINED:
            newPredC = p2.constraintSurrogates(self.xNew)
            self.predC = np.vstack((self.predC, newPredC))


        self.xNewEval = cobra.sac_res['fn'](self.xNew)
        # TODO later:
        # if (cobra$CA$active & & cobra$TFlag){
        #     xNewT < -(ev1$xNew-cobra$tCenter) % * %cobra$TM
        #     xNewT < -xNewT+cobra$tCenter
        #     ev1$xNewEvalT < -fn(xNewT)
        # }

        if self.CONSTRAINED:
            if cobra.sac_opts.EQU.active:
                raise NotImplementedError(
                    "[EvaluatorReal] Branch EQU.active (line 177-225 evalReal.R) not yet implemented!")
            else: # i.e. if not cobra.sac_opts.EQU.active
                conTol = cobra.sac_opts.SEQ.conTol
                # number of constraint violations for new point:
                self.newNumViol = np.flatnonzero(self.xNewEval[1:] > conTol).size
                # the same on constraint surrogates:
                self.newNumPred = np.flatnonzero(newPredC > conTol).size
                self.feas = concat(self.feas, self.newNumViol < 1 )
                self.feasPred = concat(self.feasPred, self.newNumPred < 1 )

                if (max(0, max(self.xNewEval[1:]))) > conTol:  # maximum violation
                    self.newMaxViol = max(0, max(self.xNewEval[1:]) )
                else:
                    self.newMaxViol = 0
                self.trueMaxViol = self.newMaxViol

        else: # i.e. if not self.CONSTRAINED
            self.newNumViol = 0
            self.newMaxViol = 0
            self.trueMaxViol = 0
            self.feas = True

        self.newCurrentEps = currentEps
        self.state = "optimized"


def getPredY0(xNew, fitnessSurrogate, p2: Phase2Vars):
    pred_y = fitnessSurrogate(xNew)
    if p2.PLOG[-1]:
        pred_y = plogReverse(pred_y, p2.pshift[-1])
    return pred_y

# getPredY is nothing else than calling getPredY0 with fitnessSurrogate=p2.fitnessSurrogate
