from opt.isaOptions import O_LOGIC
from cobraInit import CobraInitializer
from cobraPhaseII import Phase2Vars
from surrogator1 import Surrogator1   # for AdFitter
from surrogator2 import Surrogator2


class Surrogator:

    @staticmethod
    def trainSurrogates(cobra: CobraInitializer, p2: Phase2Vars) -> Phase2Vars:
        if cobra.sac_opts.ISA.onlinePLOG == O_LOGIC.MIDPTS:
            p2 = Surrogator2.trainSurrogates(cobra, p2)
        else:  # i.e. O_LOGIC.NONE or O_LOGIC.XNEW:
            p2 = Surrogator1.trainSurrogates(cobra, p2)
        return p2
