import numpy as np
# need to specify SACOBRA_Py.src as source folder in File - Settings - Project Structure,
# then the following import statements will work:
from cobraInit import CobraInitializer


class Phase2Vars:
    """
    This class is just a container for variables needed by :class:`.CobraPhaseII` (in addition to :class:`.CobraInitializer` ``cobra``).
    These variables include:

    - **EPS**           number, the current safety margin EPS in constraint surrogates
    - **currentMu**     number, the current equality margin :math:`\\mu`, see equHandling.py
    - **num**           the number of real function evaluations carried out
    - **globalOptCounter** counter of the global optimization steps in phase II, excluding repair and trust region
    - **Cfeas**          how many feasible infills in a row (see :meth:`.adjustMargins`, updateInfoAndCounters)
    - **Cinfeas**        how many infeasible infills in a row (see :meth:`.adjustMargins`, updateInfoAndCounters)
    - **fitnessSurrogate** the objective surrogate model
    - **constraintSurrogates** the constraint surrogate models
    - **pEffect**       number, calculated in calcPEffect in each iteration: If > 1, apply plog(Fres).
    - **PLOG**          boolean vector: whether plog(Fres) was applied in iteration i
    - **pshift**        float vector: with which p-shift was plog(Fres) applied (if at all) in iteration i

    Example: ``p2 = Phase2Vars(); print(p2.num);``
    """
    def __init__(self, cobra: CobraInitializer):
        self.EPS = cobra.sac_opts.SEQ.epsilonInit      # number, the current safety margin EPS in constraint surrogates
        self.currentMu = 0.0       #
        self.num = cobra.sac_res['A'].shape[0]      # the number of real function evaluations carried out
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
        self.errRatio = None
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
        self.pEffect = cobra.sac_opts.ISA.pEffectInit     # number, will be recalculated in trainSurrogates.py
        self.PLOG = np.array([], dtype=np.bool)
        self.pshift = np.array([], dtype=np.float64)
        self.fin_err = None     # number, will be set in ex_COP.py
        self.fe_thresh = None   # number, will be set in ex_COP.py

