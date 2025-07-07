import pandas as pd
# need to specify SACOBRA_Py.src as source folder in File - Settings - Project Structure,
# then the following import statements will work:
from cobraInit import CobraInitializer
from phase2Vars import Phase2Vars
import phase2Funcs as pf2
from randomStarter import RandomStarter
from surrogator import Surrogator
# from trainSurrogates import trainSurrogates, calcPEffect
from seqOptimizer import SeqOptimizer, check_if_cobra_optimizable
from evaluatorReal import EvaluatorReal
from updateSaveCobra import updateSaveCobra
from opt.equOptions import EQUoptions


class CobraPhaseII:
    """
        SACOBRA phase II executor.

        Information is communicated via object :class:`.CobraInitializer` ``cobra`` (with elements sac_opts and sac_res)
        and via object :class:`.Phase2Vars` ``p2`` (internal variables needed in phase II).
    """
    def __init__(self, cobra: CobraInitializer):
        # initial settings of all phase-II-related variables:
        self.p2 = Phase2Vars(cobra)
        self.p2.ev1 = EvaluatorReal(cobra, self.p2)
        self.p2.rs1 = RandomStarter(cobra.sac_opts)

        cobra.phase = "phase2"
        self.cobra = cobra

    def get_cobra(self):
        """
        :return: COBRA variables
        :rtype: CobraInitializer
        """
        return self.cobra

    def get_p2(self):
        """
        :return: phase II variables
        :rtype: Phase2Vars
        """
        return self.p2

    def start(self):
        """
        Start the main optimization loop of phase II

        Perform in a loop, until the budget ``feval`` is exhausted:

           - select cyclically an element ``p2.ro`` from :ref:`DRC <DRC-label>` ``XI``. Add the appropriate DRC-constraint as extra constraint to the set of constraints
           - train RBF surrogate models on the current set of infill points
           - select start point ``xStart``: either current-best ``xbest`` or random start (see :class:`.RandomStarter`)
           - perform sequential optimization, starting from ``xStart``, subject to the ``nConstraint + 1`` constraints. Result is ``xNew = p2.opt_res['x']``
           - evaluate ``xNew`` on the real functions +  (if ``EQU.active``) do :ref:`refine step <refineStep-label>`. Result is the updated :class:`.EvaluatorReal` object ``p2.ev1``
           - calculate :ref:`p-effect <pEffect-label>` for onlinePLOG (see :meth:`.Surrogator.calcPEffect`)
           - update cobra information: The new infill point ``xNew`` and its evaluation on the real functions is added to the ``cobra``'s arrays  ``A``, ``Fres``, ``Gres``
           - update and save ``cobra``: data frames :ref:`df <df-label>`, :ref:`df2 <df2-label>`, elements ``sac_res['xbest', 'fbest', 'ibest']`` of  dict :ref:`cobra.sac_res <sacres-label>`
           - adjust margins ``p2.EPS``, :math:`\\mu` (see :class:`.EQUoptions`), adjust  :math:`\\rho` (see :class:`.RBFoptions`) and ``p2.Cfeas``, ``p2.Cinfeas`` (see :meth:`phase2Funcs.adjustMargins`)

        The result is a modified object ``cobra`` (detailed results in dict :ref:`sac_res <sacres-label>` and detailed diagnostic info in data frames :ref:`df <df-label>`, :ref:`df2 <df2-label>`). The
        optimization results can be retrieved from ``cobra`` with methods :meth:`.get_fbest`, :meth:`.get_xbest`
        and  :meth:`.get_xbest_cobra`.

        :return: ``self``
        """
        s_opts = self.cobra.sac_opts
        s_res = self.cobra.sac_res
        assert self.p2.ev1.state == "initialized"
        self.p2.currentMu = s_res['muVec'][0]
        first_pass = True
        final_gama = None
        while self.p2.num < s_opts.feval:
            self.p2.gama = s_opts.XI[(self.p2.globalOptCounter % s_opts.XI.size)]
            if final_gama is not None:      # final_gama is set at the end of while loop if s_opts.SEQ.finalEpsXiZero is
                self.p2.gama = final_gama   # TRUE and if it is just before the last iter

            # TODO: MS (model-selection) part

            # train RBF surrogate models:
            self.p2 = Surrogator.trainSurrogates(self.cobra, self.p2)

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
            self.p2.ev1.update(xNew, self.cobra, self.p2, self.p2.currentMu)

            # calcPEffect (SACOBRA) for onlinePLOG
            Surrogator.calcPEffect(self.p2, self.p2.ev1.xNew, self.p2.ev1.xNewEval)

            # update cobra information (A, Fres, Gres and others)
            pf2.updateInfoAndCounters(self.cobra, self.p2)
            self.p2.num = self.cobra.sac_res['A'].shape[0]

            # update and save cobra: data frames df, df2, keys xbest, fbest, ibest in sac_res
            updateSaveCobra(self.cobra, self.p2, self.p2.EPS, pf2.fitFuncPenalRBF, pf2.distRequirement)

            # adjust margin self.p2.EPS, self.p2.currentMu, cobra.sac_opts.RBF.rho and adjust counters (self.p2.Cfeas, self.p2.Cinfeas):
            pf2.adjustMargins(self.cobra, self.p2)

            # TODO: [conditional] repairInfeasible

            # TODO: [conditional] trustRegion

            if s_opts.SEQ.finalEpsXiZero:
                if self.p2.num == s_opts.feval-1:  # last iter: exploit maximally with EPS=gama=0.0 (might require
                    self.p2.EPS = 0.0              # s_opts.SEQ.conTol=1e-7)
                    final_gama = 0.0
                    # s_opts.EQU.refine = False

            self.p2.time_init += self.p2.fitnessSurrogate.time_init
            self.p2.time_init += self.p2.fitnessSurrogate1.time_init
            self.p2.time_init += self.p2.fitnessSurrogate2.time_init
            self.p2.time_init += self.p2.constraintSurrogates.time_init
            self.p2.time_call += self.p2.fitnessSurrogate.time_call
            self.p2.time_call += self.p2.fitnessSurrogate1.time_call
            self.p2.time_call += self.p2.fitnessSurrogate2.time_call
            self.p2.time_call += self.p2.constraintSurrogates.time_call
        # end while self.p2.num

        # TODO: some final settings to self.cobra, self.p2
        return self

    # NOTE: the purpose of this function is just to supply a docstring (used in appendix of Sphinx docu):
    def get_df(self) -> pd.DataFrame:
        """
        Return data frame ``cobra.df`` with the following elements, accessible with e.g. ``cobra.df['iter']``.
        Data frame ``cobra.df`` contains ``feval`` rows, one for each true function evaluation.

        The contents of a specific row of ``cobra.df`` holds the results of a specific iteration, the *current* iteration:

        - **iter**: the iteration number
        - **y**: the objective function value at the current infill point
        - **predY**: the fitness surrogate value at the current infill point
        - **predSolu**: the fitness surrogate value at the true solution (if provided, else none)
        - **feasible**: is the current infill point feasible on the true objective?
        - **feasPred**: is the current infill point *predicted* to be feasible by the surrogate models?
        - **nViolations**: the number of violations in the constraint surrogates' prediction at the infill point
        - **trueNViol**: the number of violations in the true constraints at the infill point
        - **maxViolation**: the maximum violation of the constraint surrogates' prediction at the infill point
        - **trueMaxViol**: the maximum violation of the true constraints at the infill point
        - **feMax**: the number of iterates that the sequential optimizer took when producing this infill point
        - **fBest**: the all-time best feasible objective value. As long as no feasible point is found, the fitness of the one with the least maximum violation
        - **dist**: distance of the true solution to infill point, in rescaled space. Minimum distance for multiple solu's, None if no solu is provided
        - **distOrig**: the same, but in original space
        - **RS**: True if it is an iteration with a random start
        - **XI**: :ref:`DRC <DRC-label>` for this iteration
        - **optimConv**: convergence of sequential optimization
        - **optimTime**: time of this sequential optimization
        - **optimizer**: optimization algorithm
        - **seed**: ```sac_opts.cobraSeed``
        - ...

        :return: SACOBRA diagnostic information
        :rtype: pd.DataFrame
        """
        return self.df

    # NOTE: the purpose of this function is just to supply a docstring (used in appendix of Sphinx docu):
    def get_df2(self) -> pd.DataFrame:
        """
        Return data frame ``cobra.df2`` with the following elements, accessible with e.g. ``cobra.df2['iter']``.
        Data frame ``cobra.df2`` contains ``feval - initDesPoints`` rows, one for each true function evaluation *in phase II*.

        The contents of a specific row of ``cobra.df2`` holds the results of a specific iteration, the *current* iteration:

        - **iter**: the iteration number
        - **predY**: the fitness surrogate value at the current infill point
        - **predVal**: surrogate fitness + penalty at xNew
        - **predSolu**: the fitness surrogate value at the true solution (if provided, else none)
        - **predSoluPenal**: surrogate fitness + penalty at the true solu (only diagnostics)
        - **sigmaD**: ...
        - **penaF**: ...
        - **XI**: :ref:`DRC <DRC-label>` for this iteration
        - **rho**: smoothing factor :math:`\\rho` (interpolating / approximating RBFs)
        - **fBest**: the all-time best feasible objective value. As long as no feasible point is found, the fitness of the one with the least maximum violation
        - **EPS**: the current safety margin EPS in constraint surrogates
        - **muVec**: artificial equality constraint enlargement :math:`\\mu`
        - **PLOG**: whether :math:`plog` transformation is used in the current iteration (see :ref:`p-Effect <pEffect-label>`)
        - **pshift**: shift used in :math:`plog` transformation (see :ref:`p-Effect <pEffect-label>`)
        - **pEffect**: the :ref:`p-Effect <pEffect-label>` value
        - **state**: name of iteration state ['initialized' | 'optimized' | 'refined']

        Only if the COP contains equality constraints and if equality handling is active (:class:`.EQUoptions`
        ``EQU.active==True``), then the following row elements are not ``None``, instead they contain these
        attributes evaluated *for the current infill point*:

        - **nv_cB**: number of (artificial) constraint violations (``> conTol``) before refine on surrogates
        - **nv_cA**: number of (artificial) constraint violations (``> conTol``) after refine on surrogates
        - **nv_tB**: number of (artificial) constraint violations (``> conTol``) before refine on true constraints
        - **nv_tA**: number of (artificial) constraint violations (``> conTol``) after refine on true constraints

        :return: SACOBRA diagnostic information
        :rtype: pd.DataFrame
        """
        return self.df2
