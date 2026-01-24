import numpy as np
# need to specify SACOBRA_Py.src as source folder in File - Settings - Project Structure,
# then the following import statements will work:
from cobraInit import CobraInitializer
from gCOP import GCOP


class Phase2Vars:
    """
    This class is just a container for variables needed by :class:`.CobraPhaseII` (in addition to :class:`.CobraInitializer` ``cobra``).
    These variables include:

    - **EPS**           number, the current safety margin :math:`\\epsilon` in constraint surrogates, see :ref:`safety margin <safety_margin-label>`
    - **currentMu**     number, the current equality margin :math:`\\mu`, see :ref:`refine step <refineStep-label>`
    - **num**           the number of real function evaluations carried out
    - **globalOptCounter**      counter of the global optimization steps in phase II, excluding repair and trust region
    - **Cfeas**         how many feasible infills in a row (see :meth:`phase2Funcs.adjustMargins`, ``updateInfoAndCounters``)
    - **Cinfeas**       how many infeasible infills in a row (see :meth:`phase2Funcs.adjustMargins`, ``updateInfoAndCounters``)
    - **fitnessSurrogate**      the objective surrogate model
    - **constraintSurrogates**  the constraint surrogate models
    - **pEffect**       number, calculated in :meth:`.Surrogator1.calcPEffect` in each iteration: If > 0, apply plog(Fres).
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
        self.adFit = None       # class object, will be set in surrogator1.py
        self.printP = True
        self.write_XI = True
        self.rs_done = None     # bool to indicate whether random_start was done (randomStarter.py)
        self.rs1 = None         # gets RandomStarter object in cobraPhaseII
        self.ev1 = None         # gets EvaluatorReal object in cobraPhaseII
        self.ri2 = None         # gets RI2 (repair infeasible) object in cobraPhaseII
        self.opt_res = None     # gets the SeqOptimizer results in cobraPhaseII
        self.gama = None        # number, will be set in cobraPhaseII.py
        self.ro = None          # number, will be set in cobraPhaseII.py
        self.mu4 = 0            # number, will be conditionally set in cobraPhaseII.py
        self.pEffect = cobra.sac_opts.ISA.pEffectInit     # number, pEffect will be recalculated in calcPEffect...
        self.PLOG = np.array([], dtype=np.bool)
        self.pshift = np.array([], dtype=np.float64)
        self.midpts = None      # ndarray, will be set in surrogator2.py
        self.midptsEval = None  # ndarray, will be set in surrogator2.py
        self.fe_thresh = 0.1    # number (just for diagnostic printout in multi_gfnc), may be overwritten in ex_COP.py
        self.time_init = 0.0
        self.time_call = 0.0

        # The following members will be set via p2.fill() which is called at the end of cobraPhaseII::start:
        self.fin_err = None     # number, the final error = f(feasible best solution) - f(true solu)
        self.f_solu = None      # f(true solu) = objective at true solution (if provided)
        self.gcop = None
        self.ncall = 0          # number, how often was gcop.fn called?
        self.conTol = 0         # number, constraint tolerance
        self.constr = None      # array, constraint values at best solution found
        self.maxViol = None     # number, maximum constraint violation (given muFinal and conTol)
        self.dim = None


    def fill(self, cobra: CobraInitializer, gcop: GCOP=None):
        """
        Fill in member settings. Called at the end of SACOBRA phase II.

        Detail: Bundling all the final ``p2``-fills in a common helper function (instead of having scattered assignments
        ``c2.p2...=...``) has the advantage that one cannot forget a specific setting in one place.

        :param cobra:
        :param gcop: (optional) a COP object, just needed to retrieve ``ncall`` after optimization phase II. If
                     ``None`` then the default ``self.ncall=0`` remains.
        """
        s_res = cobra.sac_res

        # compute final error
        if cobra.solu is None:
            self.f_solu = None
        else:
            first_solu = cobra.solu[0, :] if cobra.solu.ndim == 2 else cobra.solu
            self.f_solu = s_res['originalfn'](first_solu)[0]     # objective at (first) solution point
        self.fin_err = np.array(cobra.get_feasible_best() - self.f_solu)

        # Compute constraint values at best solution found (which might be infeasible)
        xbest = cobra.get_xbest()
        self.constr = s_res['originalfn'](xbest)[1:]

        self.dim = s_res['dimension']
        self.conTol = cobra.sac_opts.SEQ.conTol
        self.maxViol = s_res['trueMaxViol'][s_res['ibest']]

        if gcop is not None:
            self.gcop = gcop
            self.ncall = gcop.ncall
