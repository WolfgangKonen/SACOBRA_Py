import numpy as np
# need to specify SACOBRA_Py.src as source folder in File - Settings - Project Structure,
# then the following import statements will work:
from cobraInit import CobraInitializer


class Phase2Vars:
    """
    Variables needed by :class:`CobraPhaseII` (in addition to :class:`CobraInitializer` ``cobra``)
    """
    def __init__(self, cobra: CobraInitializer):
        eps_init = cobra.sac_opts.epsilonInit
        self.EPS = eps_init         # number, the current safety margin EPS in constraint surrogates
        self.currentEps = 0.0       # number, the current equality margin \mu, see equHandling.py
        self.num = cobra.sac_res['A'].shape[0]      # the number of real function evaluations
        self.globalOptCounter = 0   # counter for only for the global optimization steps in phase II,
                                    # excluding repair and trust region
        self.Cfeas = 0          # how many feasible infills in a row (see adjustMargins, updateInfoAndCounters)
        self.Cinfeas = 0        # how many infeasible infills in a row (see adjustMargins, updateInfoAndCounters)
        self.fitnessSurrogate = None
        self.constraintSurrogates = None
        self.fitnessSurrogate1 = None
        self.fitnessSurrogate2 = None
        self.err1 = np.array([], dtype=np.float64)
        self.err2 = np.array([], dtype=np.float64)
        self.noProgressCount = 0
        self.printP = True
        self.write_XI = True
        self.rs_done = None     # bool to indicate whether random_start was done (randomStarter.py)
        self.rs1 = None         # gets RandomStarter object in cobraPhaseII
        self.ev1 = None         # gets EvaluatorReal object in cobraPhaseII
        self.opt_res = None     # gets the SeqOptimizer results in cobraPhaseII
        self.gama = None        # number, will be set in cobraPhaseII.py
        self.ro = None          # number, will be set in cobraPhaseII.py
        self.mu4 = 0            # number, will be conditionally set in cobraPhaseII.py
        self.pEffect = None     # number, will be set in trainSurrogates.py
        self.PLOG = np.array([], dtype=np.bool)
        self.pshift = np.array([], dtype=np.float64)
