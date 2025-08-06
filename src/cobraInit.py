import numpy as np
import pandas as pd

# need to specify SACOBRA_Py.src as source folder in File - Settings - Project Structure,
# then the following import statements will work:
from rescaleWrapper import RescaleWrapper
from initDesigner import InitDesigner
from innerFuncs import distLine, verboseprint, plogReverse
from opt.sacOptions import SACoptions
from opt.isaOptions import ISAoptions0, ISAoptions2

# long and short DRC:
# DRCL: Distance Requirement Cycle, long version:
DRCL = np.array([0.3, 0.05, 0.001, 0.0005, 0.0])
# DRCS: Distance Requirement Cycle, short version:
DRCS = np.array([0.001, 0.0])


class CobraInitializer:
    """
        Initialize SACOBRA optimization:

        - problem formulation: x0, fn, lower, upper, is_equ, solu
        - parameter settings: s_opts via :class:`.SACoptions`
        - create initial design: A, Fres, Gres via :class:`.InitDesigner`

        :param x0: start point, if given, then its dim has to be the same as ``lower``. If it  is/has NaN or None
                   on input, it is replaced by a random point from ``[lower, upper]``.
        :type x0: np.ndarray or None
        :param fn:  function returning ``(1+nConstraints)``-dim vector: [objective to minimize, constraints]
        :param fName: function name
        :param lower: lower bound, its dimension defines input space dimension
        :param upper: upper bound (same dim as lower)
        :param is_equ: boolean vector with dim ``nConstraints``: which constraints are equality constraints?
        :param solu:  (optional, for diagnostics) true solution vector or solution matrix (one solution per row):
                      one or several feasible x that deliver the minimal objective value
        :type solu: np.ndarray or None
        :param s_opts: the options. If not specified, take the default :class:`.SACoptions` object
        :type s_opts: SACoptions

        Objects of class ``CobraInitializer`` (usually named ``cobra`` below) have the following useful attributes:

        - **phase**     name of the optimization phase ['init' | 'phase1' | 'phase2']
        - **rng**       random number generator
        - **rw**        RescaleWrapper
        - **sac_opts**  the realized :class:`.SACoptions` object containing all the options
        - **sac_res**   dictionary with the SACOBRA results from initialization and optimization, see :ref:`cobra.sac_res <sacres-label>` in the appendix for details
        - **df**        data frame with diagnostic information from optimization, see :ref:`cobra.df <df-label>` in the appendix for details
        - **df2**       data frame with further diagnostic information from optimization, see :ref:`cobra.df2 <df2-label>` in the appendix for details

        See :ref:`appendix-label` for more details on :ref:`cobra.sacres <sacres-label>`, :ref:`cobra.df <df-label>`, :ref:`cobra.df2 <df2-label>`.

    """

    def __init__(self, x0, fn: object, fName: str, lower: np.ndarray, upper: np.ndarray,
                 is_equ: np.ndarray,
                 solu = None,
                 s_opts: SACoptions = SACoptions(),
                 ) -> object:
        """
        docstring for __init__
        """
        #
        # STEP 0: first settings and checks
        #
        dimension = lower.size
        print(f"*** Starting run with seed {s_opts.cobraSeed} ***")
        self.rng = np.random.default_rng(seed=s_opts.cobraSeed)    # moved up here (before potential x0 generation)
        # set.seed(s_opts.cobraSeed)   # TODO: proper RNG seeding
        if s_opts.ID.initDesPoints is None:
            # s_opts.ID.initDesPoints = 2 * dimension + 1   # /WK/2025/05/06: old and wrong (from R side)
            # these are the necessary minimum initDesPoints for kernel="cubic":
            if s_opts.RBF.degree == 1:
                s_opts.ID.initDesPoints = dimension + 1
            else:   # i.e. degree==2
                s_opts.ID.initDesPoints = (dimension + 1) * (dimension + 2) // 2
        if s_opts.XI is None:
            s_opts.XI = DRCL
        # The threshold parameter for the number of consecutive iterations that yield ...
        s_opts.Tfeas = np.floor(2 * np.sqrt(dimension))  # ... feasible solutions before the margin is reduced
        s_opts.Tinfeas = np.floor(2 * np.sqrt(dimension))  # ... infeasible solutions before the margin is increased

        lower = np.array(lower)
        upper = np.array(upper)
        if x0 is None or np.isnan(x0).any():
            x0 = self.rng.random(size=dimension) * (upper - lower) + lower
        assert lower.size == upper.size, "[CobraInitializer] size of lower and upper differs"
        assert (lower < upper).all(), "[CobraInitializer] lower < upper violated"
        assert x0.ndim == 1, "[CobraInitializer] x0 is not 1-dimensional"
        assert x0.size == lower.size, "[CobraInitializer] size of x0 and lower differs"
        assert s_opts.ID.initDesPoints < s_opts.feval, "CobraInitializer: Too many init design points"
        x0 = np.maximum(x0, lower)
        x0 = np.minimum(x0, upper)
        #
        # STEP 1: (optional) rescaling
        #
        originalfn = fn
        originalx0 = x0
        originalL = lower
        originalU = upper
        lb = np.repeat(s_opts.ID.newLower, dimension)
        ub = np.repeat(s_opts.ID.newUpper, dimension)
        self.rw = RescaleWrapper(originalfn, originalL, originalU, lb, ub)
        self.solu = solu

        if s_opts.ID.rescale:
            x0 = self.rw.forward(x0)
            fn = self.rw.apply
            lower = lb
            upper = ub

        #
        # STEP 2: second settings and checks
        #
        fn_x0 = fn(x0)
        nConstraints = fn_x0.size - 1
        assert nConstraints >= 0
        # if is_equ is None:      # the default assumes that all constraints are inequality constraints:
        #     is_equ = np.zeros(nConstraints, dtype=bool)
        assert is_equ.size == nConstraints, f"Wrong size is_equ.size = {is_equ.size} (nConstraints = {nConstraints})"

        CONSTRAINED = (nConstraints > 0)
        # assert not CONSTRAINED or not conditioningAnalysis.active, \
        #        "This version does not support conditioning analysis for constrained problems "
        # if not CONSTRAINED:
        #   assert not TrustRegion, \
        #          "cobraInit: This version does not support trust Region functionality for unconstrained Problems"

        if not CONSTRAINED:
            verboseprint(verbose=s_opts.verbose, important=True, message="An unconstrained problem is being addressed")

        ell = min(upper - lower)  # length of smallest side of search space
        if s_opts.SEQ.epsilonInit is None: s_opts.SEQ.epsilonInit = 0.005 * ell
        if s_opts.SEQ.epsilonMax is None:  s_opts.SEQ.epsilonMax = 2 * 0.005 * ell
        if s_opts.ID.initDesOptP is None: s_opts.ID.initDesOptP = s_opts.ID.initDesPoints
        s_opts.phase1DesignPoints = None

        #
        # STEP 3: create initial design
        #
        # TODO archive functionality
        A, Fres, Gres = InitDesigner(x0, fn, self.rng, lower, upper, s_opts)()
        verboseprint(s_opts.verbose, important=False,
                     message=f"Shapes of A, Fres, Gres: {A.shape}, {Fres.shape}, {Gres.shape}")
        # print(A[-1,:])
        # fe = A.shape[0]  # number of function evaluations = number of rows of A (never used, we have p2.num)

        #
        # STEP 4: update structures
        #
        self.sac_res = {'fn': fn,
                        'lower': lower,
                        'upper': upper,
                        'x0': x0,
                        'originalfn': originalfn,
                        'originalL': originalL,
                        'originalU': originalU,
                        'originalx0': originalx0,
                        'xStart': x0.copy(),  # will be overwritten with xbest below and in phase II
                        'dimension': dimension,
                        'is_equ': is_equ,
                        # 'iteration': 0,   # never used
                        # 'fe': fe,         # never used, we have p2.num
                        'A': A,
                        'Fres': Fres,
                        'Gres': Gres,
                        'nConstraints': nConstraints,
                        'l': ell
                        }
        self.for_rbf = {'A': A,
                        'Fres': Fres,
                        'Gres': Gres
                        }

        # TODO: default settings DEBUG_RBF, CA, MS, RI, TR
        self.sac_res['GRfact'] = 1  # will be later overwritten by adCon, when the TODO 'equalityConstraints' is done.
        self.sac_res['finMarginCoef'] = 1  # also set in adCon, used in equHandling.py
        self.sac_opts = s_opts
        if s_opts.ISA.aCF and nConstraints != 0:
            verboseprint(s_opts.verbose, important=False, message="adjusting constraint functions")
            self.adCon()
            # cobra$fn = fn

        #
        # STEP 5: compute numViol, maxViol
        #
        z0 = np.zeros(Gres.shape[0])
        numViol = np.sum(Gres > 0, axis=1) if Gres.size > 0 else z0   # number of initial constraint violations
        maxViol = np.max(np.maximum(Gres, 0), axis=1) if Gres.size > 0 else z0    # max. of initial constraint violation
        trueNumViol = numViol
        trueMaxViol = maxViol
        self.sac_res['muVec'] = np.repeat(0.0, s_opts.ID.initDesPoints)
        # just to have a valid entry, will be replaced below in case s_opts.EQU.active==True
        # (self.sac_res['muVec'] is what is named cobra$currentEps in R)

        equ_ind = np.flatnonzero(is_equ)
        if equ_ind.size == 0:
            s_opts.EQU.active = False
        if s_opts.EQU.active:
            # equality-related changes to numViol, maxViol, trueNumViol, trueMaxViol, self.sac_res['muVec']
            # [this is the branch starting in cobraInit.R with "else if(equHandle$active)" (line 821 ff)]
            # equ2Index = np.concat(equ_ind, nConstraints + np.arange(0, equ_ind.size))
            #
            tempG = Gres.copy()     # .copy important here, otherwise changes to tempG would change Gres as well (!)
            tempG[:, equ_ind] = abs(tempG[:, equ_ind])
            # z = self.sac_res['GRfact']
            # tempG2 = tempG * z if nConstraints == 1 else tempG @ np.diag(z)
            # trueMaxViol = np.maximum(0, np.max(tempG2, axis=1))
            trueMaxViol = np.maximum(0, np.max(tempG, axis=1))
            # /WK/2025/05/04: the version with tempG2 is used in R as well: GRfact (see adCon()) is a vector of length
            # nConstraints and tempG.shape = (idp,nConstraints). With the trick "... @ np.diag(GRfact)" we multiply each
            # row of tempG elementwise with GRfact. The violations are weighted with GRfact, then we take the maximum.
            # The simpler alternative would be:
            #       trueMaxViol = np.maximum(0, np.max(tempG, axis=1))
            # but this is not the way it is in R.

            def tav_func(temp_g):
                temp_g = np.maximum(temp_g, 0)
                return np.median(np.sum(temp_g, axis=1))
            switcher = {
                "useGrange": self.sac_res['GrangeEqu'],
                "TAV": tav_func(tempG),
                "TMV": np.median(trueMaxViol),
                "EMV": np.median(np.max(tempG[:, equ_ind], axis=1))
            }
            currentMu = switcher.get(s_opts.EQU.initType, "Invalid initType")
            assert currentMu != "Invalid initType", f"[cobraInit] Wrong s_opts.EQU.initType = {s_opts.EQU.initType}"
            currentMu = max(currentMu, s_opts.EQU.muFinal)
            if s_opts.EQU.muType == "CONS":            # /WK/2025/05/01: bug fix: "CONS" overrides initType
                currentMu = s_opts.EQU.muFinal
            self.sac_res['muVec'] = np.repeat(currentMu, s_opts.ID.initDesPoints)
            tempG[:, equ_ind] = tempG[:, equ_ind] - currentMu  # /WK/2025/03/23: bug fix
            conTol = s_opts.SEQ.conTol
            maxViol = np.maximum(conTol, np.max(tempG, axis=1))     # /WK/2025/03/23: bug fix conTol
            numViol = np.sum(tempG > conTol, axis=1)                # /WK/2025/03/23: bug fix conTol

        #
        # STEP 6: best feasible/infeasible solution, based on numViol
        #
        if 0 in numViol:
            # if there are feasible points, select among them the one with minimal Fres:
            fbest = min(Fres[numViol == 0])
            xbest = A[Fres == fbest, :]
            ibest = np.flatnonzero(Fres == fbest)[0]
        else:
            # if there is no feasible point yet: take the set of points with min number of violated constraints ...
            minNumIndex = np.flatnonzero(numViol == min(numViol))
            # ... and select from them the one with minimal Fres:
            FresMin = Fres[minNumIndex]
            ind = np.flatnonzero(FresMin == min(FresMin))[0]
            index = minNumIndex[ind]
            fbest = Fres[index]
            xbest = A[index, :]
            ibest = index
        xbest = xbest.reshape(xbest.size)   # bug fix 2025-06-24: reshape xbest to 1D-array of length xbest.size
        fbestArray = np.repeat(fbest, s_opts.ID.initDesPoints)
        xbestArray = np.tile(xbest, (s_opts.ID.initDesPoints, 1))

        #
        # STEP 7: update structures
        #
        self.phase = "init"
        phaseVec = np.repeat(self.phase, s_opts.ID.initDesPoints)
        sac_res2 = {'numViol': numViol,
                    'trueNumViol': trueNumViol,
                    'maxViol': maxViol,
                    'trueMaxViol': trueMaxViol,
                    'predC': None,
                    'fbest': fbest,
                    'xbest': xbest,
                    'ibest': ibest,
                    'fbestArray': fbestArray,
                    'xbestArray': xbestArray,
                    'phase': phaseVec,
                    'xStart': xbest.reshape(dimension,),         # bug fix
                    'rs': np.array([])      # (only for debug logging, see RandomStarter.decide_about_random_start)
                    }
        self.sac_res.update(sac_res2)       # update dict sac_res with the content of sac_res2
        self.df = None
        self.df2 = pd.DataFrame()

        # TODO:
        # if (is.character(cobra$CA$ITER)) if (cobra$CA$ITER == "all") cobra$CA$ITER = seq(initDesPoints, feMax, 1)
        #
        # cobra$radi = np.repeat(cobra$TRlist$radiInit, initDesPoints)

        #
        # STEP 8: SACOBRA initialization (depending on s_opts.ISA, perform adDRC and adCon)
        #
        # --- obsolete, we set s_opts.ISA directly to the right class (ISAoptions, ISAoptions0 or ISAoptions2) ---
        # if s_opts.isa_ver == 0: s_opts.ISA = ISAoptions0()
        # elif s_opts.isa_ver == 2: s_opts.ISA = ISAoptions2()
        # ---
        if s_opts.ISA.isa_ver > 0:
            verboseprint(s_opts.verbose, important=False, message="Parameter and function adjustment phase")
            # s_opts.pEffect = s_opts.ISA.pEffectInit    # obsolete, we have p2.pEffect

            if s_opts.ISA.aDRC:
                if s_opts.XI.size != DRCL.size:
                    print("Warning: XI is different from default (DRCL), but sac$aDRC==TRUE, "
                          "so XI will be set by automatic DRC adjustment!")
                elif np.any(s_opts.XI != DRCL):
                    print("Warning: XI is different from default (DRCL), but sac$aDRC==TRUE, "
                          "so XI will be set by automatic DRC adjustment!")

                verboseprint(s_opts.verbose, important=False, message="adjusting DRC")
                DRC = self.adDRC()     # max(self.sac_res['Fres']), min(self.sac_res['Fres'])
                s_opts.XI = DRC

            # --- adFit is now called in *each* iteration of cobraPhaseII (adaptive plog) ---

            # --- never used: ---
            # self.sac_res['RSDONE'] = np.repeat(np.nan, s_opts.ID.initDesPoints)

        self.sac_opts = s_opts

    def get_sac_opts(self) -> SACoptions:
        return self.sac_opts

    def get_sac_res(self) -> dict:
        return self.sac_res

    def get_xbest_cobra(self):
        """
        :return: best solution in COBRA space (maybe rescaled)
        :rtype: np.ndarray
        """
        return self.sac_res['xbest']

    def get_xbest(self):
        """
        :return: best solution in original space
        :rtype: np.ndarray
        """
        if self.sac_opts.ID.rescale:
            return self.rw.inverse(self.sac_res['xbest'])
        return self.get_xbest_cobra()

    def get_fbest(self):
        """
        Return the original objective function value at the best feasible solution.

        Note: We cannot take the best function value via ``sac_res['fn']``, because this
        may be modified by PLOG or others.
        """
        return self.sac_res['originalfn'](self.get_xbest())[0]

    def adDRC(self):        # , maxF, minF
        """ Adjust :ref:`DRC <DRC-label>` (distance requirement cycle), based on range of ``Fres`` """
        FRange = (max(self.sac_res['Fres']) - min(self.sac_res['Fres']))
        if FRange > 1e+03:
            DRC = DRCS
            print(f"FR={FRange} is large, XI is set to Short DRC")
        else:
            DRC = DRCL
            print(f"FR={FRange} is large, XI is set to Long DRC")
        return DRC

    def adCon(self):
        """
            Adjust several elements according to constraint range.

            The following elements of ``self.sac_res`` may be changed: 'fn', 'Gres', 'Grange', 'GrangeEqu'
        """
        s_opts = self.sac_opts
        fnold = self.sac_res['fn']
        Gres = self.sac_res['Gres']
        assert not np.any(np.isnan(Gres)), "[adCon] self.sac_res['Gres'] contains NaN elements"
        equ_ind = np.flatnonzero(self.sac_res['is_equ'])

        GRL = np.apply_along_axis(self.maxMinLen, axis=0, arr=Gres)
        # axis=0 means that arr is sliced along axis 0, i.e. maxMinLen is applied to the columns of Gres
        if min(GRL) == 0:   # pathological case where at least one constraint is constant:
            GR = -np.inf    # inhibit constraint normalization
        else:
            GR = max(GRL) / min(GRL)
        self.sac_res['GR'] = GR

        if GR > s_opts.ISA.TGR:
            verboseprint(s_opts.verbose, True, f"GR={GR} is large --> normalizing constraint functions ...")
            # GRfact = np.hstack((1, GRL * (1 / np.mean(GRL))))   # probably buggy: at least for G10, some constraint ranges get bigger than before (!)
            GRfact = np.hstack((1, GRL))                          # fix 2025/06/12: seems to give smaller Gres ranges

            # finding the normalizing coefficient of the equality constraints
            if equ_ind.size != 0:
                GRF = GRfact[1:]
                EQUfact = GRF[equ_ind]
                # define a coefficient for the final equality margin
                muFinal = s_opts.EQU.muFinal
                finMarginCoef = min([min(muFinal / EQUfact), muFinal]) / muFinal
                # --- /WK/2025/05/04: disabled the following, because it is also disabled (overwritten by the initial
                # --- setting for muFinal) on the R side: ---
                # s_opts.EQU.muFinal = finMarginCoef * muFinal
                self.sac_res['GRfact'] = GRF
                self.sac_res['finMarginCoef'] = finMarginCoef

            def fn(x):
                return fnold(x) / GRfact

            self.sac_res['fn'] = fn

            self.sac_res['Gres'] = Gres @ np.diag(1/GRfact[1:])
            self.for_rbf['Gres'] = self.for_rbf['Gres'] @ np.diag(1/GRfact[1:])
            # bug fix 2025/06/12: the line with 'for_rbf' was missing before and this led to wrong constraint surrogate
            # models whenever 'normalizing constraint functions' was active(mind-buggingly high and wrong maxViol
            # --> no feasible points were found). With this 'for_rbf'-line, everything is OK.

        self.sac_res['Grange'] = np.mean(GRL)
        self.sac_res['GrangeEqu'] = np.mean(GRL[equ_ind]) if equ_ind.size > 0 else np.mean(GRL)

    def maxMinLen(self, x):
        maxL = max(x)
        minL = min(x)
        return maxL - minL

    def maxMinLen2(self, x):
        # never used
        maxL = np.quantile(x, 0.9)
        minL = np.quantile(x, 0.1)
        return (maxL - minL) / 2

    # NOTE: the purpose of this function is just to supply a docstring (used in appendix of Sphinx docu):
    def get_sac_res(self):
        """
        Return dictionary ``sac_res`` with the following elements, accessible with e.g. ``sac_res['fn']``:

        - **fn**: function returning ``(1+nConstraints)``-dim vector: [objective to minimize, constraints], optionally rescaled
        - **originalfn**: the same, but before rescaling
        - **lower**: lower-bound vector with dimension :math:`d`, optionally rescaled
        - **originalL**: the same, but before rescaling
        - **upper**: upper-bound vector with dimension :math:`d`, optionally rescaled
        - **originalU**: the same, but before rescaling
        - **x0**: initial start point, either given with the problem formulation or a random point, optionally resacled
        - **originalx0**: the same, but before rescaling
        - **xStart**: start point for each sequential optimization: initially ``x0``, but in later iterations replaced by ``xbest`` or random start
        - **dimension**: input space dimension :math:`d`
        - **nConstraints**: number of constraints
        - **is_equ**: boolean vector of size ``nConstraints``: ``True`` if ``n``'th constraint is equality constraint
        - **A**: matrix ``(nIter, dimension)`` holding the initial design + all infill points
        - **Fres**: vector of size ``nIter``: objective value for each point in ``A``
        - **Gres**: matrix ``(nIter, nConstraints)``: constraint values for each point in ``A``
        - **predC**: matrix ``(nIter, nConstraints)``: *predicted* constraint values for each point in ``A``. The prediction of the current infill point is made with the current constraint surrogates which are formed from the ``nIter - 1`` previous points.
        - **muVec**: vector of size ``nIter``: :math:`\\mu` in the ``i``'th iteration
        - **numViol, trueNumViol, maxViol, trueMaxViol**: same as columns *nViolations, trueNViol, maxViolation, trueMaxViol* of :ref:`cobra.df <df-label>`
        - **fbestArray**: vector of size ``nIter``: best feasible objective value in the ``i``'th iteration
        - ...

        :return: SACOBRA results
        :rtype: dict
        """
        return self.sac_res
