# need to specify SACOBRA_Py.src as source folder in File - Settings - Project Structure,
# then the following import statements will work:
from cobraInit import CobraInitializer
from phase2Vars import Phase2Vars
import phase2Funcs as pf2
from randomStarter import RandomStarter
from trainSurrogates import trainSurrogates, calcPEffect
from seqOptimizer import SeqOptimizer, check_if_cobra_optimizable
from evaluatorReal import EvaluatorReal
from updateSaveCobra import updateSaveCobra


class CobraPhaseII:
    """
        SACOBRA phase II executor.

        Information is communicated via object :class:`CobraInitializer` ``cobra`` (with elements sac_opts and sac_res)
        and via object :class:`Phase2Vars` ``p2`` (internal variables needed in phase II).
    """
    def __init__(self, cobra: CobraInitializer):
        # initial settings of all phase-II-related variables:
        self.p2 = Phase2Vars(cobra)
        self.p2.ev1 = EvaluatorReal(cobra, self.p2)
        self.p2.rs1 = RandomStarter(cobra.sac_opts)

        cobra.phase = "phase2"
        self.cobra = cobra

    def get_cobra(self):
        return self.cobra

    def get_p2(self):
        return self.p2

    def start(self):
        s_opts = self.cobra.sac_opts
        s_res = self.cobra.sac_res
        assert self.p2.ev1.state == "initialized"
        self.p2.currentEps = s_res['muVec'][0]
        first_pass = True
        final_gama = None
        while self.p2.num < s_opts.feval:
            self.p2.gama = s_opts.XI[(self.p2.globalOptCounter % s_opts.XI.size)]
            if final_gama is not None:      # final_gama is set at the end of while loop if s_opts.finalEpsXiZero is
                self.p2.gama = final_gama   # TRUE and if it is just before the last iter

            # TODO: MS (model-selection) part

            # train RBF surrogate models:
            self.cobra = trainSurrogates(self.cobra, self.p2)

            if first_pass:
                # needed just for assertion check in testCOP.test_G06_R:
                self.p2.fp1 = self.p2.fitnessSurrogate(s_res['xbest'] + 1)
                self.p2.gp1 = self.p2.constraintSurrogates(s_res['xbest'] + 1)
                first_pass = False

            if s_opts.EQU.mu4inequality:
                # The internal parameter p2.mu4 (will become currentMu in seqOptimizer.py) is normally 0.
                # It will be set to the last element of cobra.sac_res['muVec'] (cobra$currentEps in R)
                # if mu4inequality is TRUE.
                self.p2.mu4 = s_res['muVec'][-1]

            # TODO: CA (conditioning analysis, whitening part), if(cobra$CA$active)  [OPTIONAL]

            self.p2.ro = self.p2.gama * s_res['l']  # 0.001 #
            # s_res['l'] is set in cobraInit (length of smallest side of search space)
            # TODO: take the EPS set by adjustMargins:   cobra$EPS < - EPS

            # select xStart: either xbest or [conditional, if flag ISA.RS] random start (see randomStarter.py)
            xStart = self.p2.rs1.random_start(self.cobra, self.p2)

            check_if_cobra_optimizable(self.cobra, self.p2)

            # start sequential optimizer on surrogates and write result to self.p2.opt_res
            # (e.g. the new best x is in xNew = self.p2.opt_res['x']):
            SeqOptimizer(xStart, self.cobra, self.p2)
            self.p2.globalOptCounter += 1       # a counter which counts all main iterates, excluding repair or TR
            self.p2.ev1.state = "optimized"     # flag ev1 as being in the state after sequential optimization

            # evaluate xNew on the real functions + do refine step (if cobra.sac_opts.EQU.active).
            # Result is the updated EvaluatorReal object self.p2.ev1:
            xNew = self.p2.opt_res['x']
            self.p2.ev1.update(xNew, self.cobra, self.p2, self.p2.currentEps)

            # [conditional] calcPEffect (SACOBRA) for onlinePLOG
            calcPEffect(self.p2, self.p2.ev1.xNew, self.p2.ev1.xNewEval)

            # update cobra information (A, Fres, Gres and others)
            pf2.updateInfoAndCounters(self.cobra, self.p2)
            self.p2.num = self.cobra.sac_res['A'].shape[0]

            # update and save cobra: data frames df, df2, keys xbest, fbest, ibest in sac_res
            updateSaveCobra(self.cobra, self.p2, self.p2.EPS, pf2.fitFuncPenalRBF, pf2.distRequirement)

            # adjust margin self.p2.EPS and adjust counters (self.p2.Cfeas, self.p2.Cinfeas):
            pf2.adjustMargins(self.cobra, self.p2)

            # TODO: [conditional] repairInfeasible

            # TODO: [conditional] trustRegion

            if s_opts.finalEpsXiZero:
                if self.p2.num == s_opts.feval-1:  # last iter: exploit maximally with EPS=gama=0.0 (might require
                    self.p2.EPS = 0.0              # s_opts.SEQ.conTol=1e-7)
                    final_gama = 0.0
                    # s_opts.EQU.refine = False

        # end while self.p2.num

        # TODO: some final settings to self.cobra, self.p2
        return self.cobra
