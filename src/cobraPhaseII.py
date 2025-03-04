import numpy as np
# need to specify SACOBRA_Py.src as source folder in File - Settings - Project Structure,
# then the following import statements will work:
from cobraInit import CobraInitializer
from phase2Vars import Phase2Vars
import phase2Funcs as pf2
import seqOptimizer as if2
from innerFuncs import verboseprint
from trainSurrogates import trainSurrogates
from rescaleWrapper import RescaleWrapper
from initDesigner import InitDesigner
from opt.sacOptions import SACoptions
from opt.isaOptions import ISAoptions
from seqOptimizer import SeqOptimizer, check_if_cobra_optimizable
from evaluatorReal import EvaluatorReal

class CobraPhaseII:
    """
        SACOBRA phase II executor.

        Information is communicated via object :class:`CobraInitializer` ``cobra`` (with elements sac_opts and sac_res)
        and via object :class:`Phase2Vars` ``p2`` (internal variables needed in phase II).
    """
    def __init__(self, cobra: CobraInitializer):

        #
        # STEP 0: first settings and checks
        #
        s_opts = cobra.get_sac_opts()
        s_opts.ISA.TGR = 42

        # initial settings of all phase-II-related variables:
        self.p2 = Phase2Vars(cobra)
        self.p2.ev1 = EvaluatorReal(cobra, self.p2)
        # print(f"executing dummyFunc: {if2.dummyFunc(cobra)}")

        #
        # STEP 4: update structures
        #
        self.phase = "phase2"
        self.cobra = cobra

    def get_cobra(self):
        return self.cobra

    def get_p2(self):
        return self.p2

    def __call__(self, *args, **kwargs):
        return self.cobra

    def start(self):
        s_opts = self.cobra.sac_opts
        s_res = self.cobra.sac_res
        assert self.p2.ev1.state == "initialized"
        while self.p2.num < s_opts.feval:
            self.p2.gama = s_opts.XI[(self.p2.globalOptCounter % s_opts.XI.size)]

            # TODO: MS (model-selection) part

            self.cobra = trainSurrogates(self.cobra, self.p2)

            # TODO: CA (conditioning analysis, whitening part), if(cobra$CA$active)  [OPTIONAL]

            self.p2.ro = self.p2.gama * s_res['l'] # 0.001 #
            # s_res['l'] is set in cobraInit (length of smallest side of search space)
            # TODO: take the EPS set by adjustMargins:   cobra$EPS < - EPS

            # self.cobra = pf2.selectXStart(self.cobra)
            # # TODO: select xStart (either xbest or RandomStart, see SACOBRA.R)
            # if s_opts.RS:
            #     # cobra = RandomStart(cobra)   # TODO
            #     xStart = s_res['xStart']
            #     # if(any(cobra$xbest!=xStart)) cobra$noProgressCount<-0
            # else:
            #     xStart = s_res['xbest']
            xStart = s_res['xStart']

            check_if_cobra_optimizable(self.cobra, self.p2)

            # start sequential optimizer on surrogates and write result to self.p2.opt_res
            # (e.g. the new best x is in xNew = self.p2.opt_res['x']):
            SeqOptimizer(xStart, self.cobra, self.p2)
            self.p2.globalOptCounter += 1  # this is a counter which counts all main iterates, excluding repair or TR
            self.p2.ev1.state = "optimized"     # signal ev1 that it is updated after seq. optimization

            # evaluate xNew on the real functions + do refine step (if cobra.sac_opts.EQU.active).
            # Result is the updated EvaluatorReal object self.p2.ev1:
            self.p2.ev1.update(self.cobra, self.p2)

            # TODO: [conditional] calcPEffect (SACOBRA) for onlinePLOG

            pf2.updateInfoAndCounters(self.cobra, self.p2, self.phase)
            self.p2.num = self.cobra.sac_res['A'].shape[0]

            # TODO: updateSaveCobra

            # TODO: adjustMargins

            # TODO: [conditional] repairInfeasible

            # TODO: [conditional] trustRegion


        # end while
        # TODO: some final settings to self.cobra, self.p2
        return self.cobra
