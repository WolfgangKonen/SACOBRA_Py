import numpy as np
# need to specify SACOBRA_Py.src as source folder in File - Settings - Project Structure,
# then the following import statements will work:
from cobraInit import CobraInitializer


class Phase2Vars:
    """
    Variables needed by :class:`CobraPhaseII` (in addition to the :class:`CobraInitializer` object)
    """
    def __init__(self, cobra: CobraInitializer):
        self.EPS = cobra.sac_opts.epsilonInit
        self.currentEps = 0.0       # dummy, TODO: needs to be computed by equality-branch
        self.num = cobra.sac_res['A'].shape[0]      # counter for real function evaluations
        self.globalOptCounter = 0   # counter for only for the global optimization steps in phase II,
                                    # excluding repair and trust region
        self.Cfeas = 0          # how many feasible infills in a row (see adjustMargins, updateInfoAndCounters)
        self.Cinfeas = 0        # how many infeasible infills in a row (see adjustMargins, updateInfoAndCounters)
        self.fitnessSurrogate = None
        self.constraintSurrogates = None
        self.fitnessSurrogate1 = None
        self.fitnessSurrogate2 = None
        self.PLOG = np.array([], dtype=np.bool)
        self.pshift = np.array([], dtype=np.float64)
        self.err1 = np.array([], dtype=np.float64)
        self.err2 = np.array([], dtype=np.float64)
        self.noProgressCount = 0
        self.printP = True
        self.write_XI = True
        self.rs1 = None         # gets RandomStarter object in cobraPhaseII
        self.ev1 = None         # gets EvaluatorReal object in cobraPhaseII
        self.opt_res = None     # gets the SeqOptimizer results in cobraPhaseII

