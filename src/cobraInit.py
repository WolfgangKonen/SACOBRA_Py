from typing import Union
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
        - parameter settings: s_opts via :class:`SACoptions`
        - create initial design: A, Fres, Gres via :class:`InitDesigner`
    """
    def __init__(self, x0, fn, fName, lower: np.ndarray, upper: np.ndarray,
                 is_equ: np.ndarray[bool],
                 solu: Union[np.ndarray, None] = None, s_opts=SACoptions(50),
                 ):
        """

        :param x0: start point, if given, then its dim has to be the same as ``lower``. If it  is/has NaN or None
                    on input, it is replaced by a random point from ``[lower, upper]``.
        :param fn:  function returning ``(1+nConstraints)``-dim vector: [objective to minimize, constraints]
        :param fName: function name
        :param lower: lower bound, its dimension defines input space dimension
        :param upper: upper bound (same dim as lower)
        :param is_equ: boolean vector with dim ``nConstraints``: which constraints are equality constraints?
        :param solu:  (optional, for diagnostics) true solution vector or solution matrix (one solution per row):
                      one or several feasible x that deliver minimal objective value
        :param s_opts: the options, see :class:`SACoptions`
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
        originalXStart = x0
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
        if s_opts.epsilonInit is None: s_opts.epsilonInit = 0.005 * ell
        if s_opts.epsilonMax is None:  s_opts.epsilonMax = 2 * 0.005 * ell
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
        fe = A.shape[0]  # number of function evaluations = number of rows of A

        #
        # STEP 4: update structures
        #
        self.sac_res = {'fn': fn,
                        'lower': lower,
                        'upper': upper,
                        'x0': x0,
                        'xStart': x0.copy(),  # will be overwritten with xbest below and in phase II
                        'dimension': dimension,
                        'originalfn': originalfn,
                        'originalL': originalL,
                        'originalU': originalU,
                        'originalXStart': originalXStart,
                        'is_equ': is_equ,
                        'iteration': 0,
                        'fe': fe,
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
            currentEps = switcher.get(s_opts.EQU.initType, "Invalid initType")
            assert currentEps != "Invalid initType", f"[cobraInit] Wrong s_opts.EQU.initType = {s_opts.EQU.initType}"
            currentEps = max(currentEps, s_opts.EQU.equEpsFinal)
            if s_opts.EQU.epsType == "CONS":            # /WK/2025/05/01: bug fix: "CONS" overrides initType
                currentEps = s_opts.EQU.equEpsFinal
            self.sac_res['muVec'] = np.repeat(currentEps, s_opts.ID.initDesPoints)
            tempG[:, equ_ind] = tempG[:, equ_ind] - currentEps  # /WK/2025/03/23: bug fix
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
                    'rs': np.array([])      # will be populated by random_start
                    }
        self.sac_res.update(sac_res2)       # update dict sac_res with the content of sac_res2
        self.df = None
        self.df2 = pd.DataFrame()

        # TODO:
        # if (is.character(cobra$CA$ITER)) if (cobra$CA$ITER == "all") cobra$CA$ITER = seq(initDesPoints, feval, 1)
        #
        # cobra$radi = np.repeat(cobra$TRlist$radiInit, initDesPoints)

        #
        # STEP 8: SACOBRA initialization (set s_opts.ISA, perform adDRC and adCon)
        #
        if s_opts.DOSAC == 0: s_opts.ISA = ISAoptions0()
        elif s_opts.DOSAC == 2: s_opts.ISA = ISAoptions2()
        if s_opts.DOSAC > 0:
            verboseprint(s_opts.verbose, important=False, message="Parameter and function adjustment phase")
            s_opts.pEffect = s_opts.ISA.pEffectInit

            if s_opts.ISA.aDRC:
                if s_opts.XI.size != DRCL.size:
                    print("Warning: XI is different from default (DRCL), but sac$aDRC==TRUE, "
                          "so XI will be set by automatic DRC adjustment!")
                elif np.any(s_opts.XI != DRCL):
                    print("Warning: XI is different from default (DRCL), but sac$aDRC==TRUE, "
                          "so XI will be set by automatic DRC adjustment!")

                verboseprint(s_opts.verbose, important=False, message="adjusting DRC")
                DRC = self.adDRC(max(self.sac_res['Fres']), min(self.sac_res['Fres']))
                s_opts.XI = DRC

            # --- adFit is now called in *each* iteration of cobraPhaseII (adaptive plog) ---

            self.sac_res['RSDONE'] = np.repeat(np.nan, s_opts.ID.initDesPoints)

        self.sac_opts = s_opts

    def get_sac_opts(self) -> SACoptions:
        return self.sac_opts

    def get_sac_res(self) -> dict:
        return self.sac_res

    def get_xbest_cobra(self):
        """
        :return: best solution in COBRA space (maybe rescaled)
        """
        return self.sac_res['xbest']

    def get_xbest(self):
        """
        :return: best solution in original space
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

    def adDRC(self, maxF, minF):
        """ Adjust DRC (distance requirement cycle) """
        FRange = (maxF - minF)
        if FRange > 1e+03:
            DRC = DRCS
            print(f"FR={FRange} is large, XI is set to Short DRC")
        else:
            DRC = DRCL
            print(f"FR={FRange} is large, XI is set to Long DRC")
        return DRC

    def adCon(self):
        s_opts = self.sac_opts
        fnold = self.sac_res['fn']
        Gres = self.sac_res['Gres']
        assert not np.any(np.isnan(Gres)), "[adCon] self.sac_res['Gres'] contains NaN elements"
        equ_ind = np.flatnonzero(self.sac_res['is_equ'])

        GRL = np.apply_along_axis(self.maxMinLen, axis=0, arr=Gres)
        # axis=0 means that arr is sliced along axis 0, i.e. detLen is applied to the columns of Gres
        if min(GRL) == 0:   # pathological case where at least one constraint is constant:
            GR = -np.inf    # inhibit constraint normalization
        else:
            GR = max(GRL) / min(GRL)

        if GR > s_opts.ISA.TGR:
            verboseprint(s_opts.verbose, True, "Normalizing Constraint Functions ...")
            GRfact = np.hstack((1, GRL * (1 / np.mean(GRL))))

            # finding the normalizing coefficient of the equality constraints
            if equ_ind.size != 0:
                GRF = GRfact[1:]
                EQUfact = GRF[equ_ind]
                # define a coefficient for the final equality margin
                equEpsFinal = s_opts.EQU.equEpsFinal
                finMarginCoef = min([min(equEpsFinal / EQUfact), equEpsFinal]) / equEpsFinal
                # --- /WK/2025/05/04: disabled the following, because it is also disabled (overwritten by the initial
                # --- setting for equEpsFinal) on the R side: ---
                # s_opts.EQU.equEpsFinal = finMarginCoef * equEpsFinal
                self.sac_res['GRfact'] = GRF
                self.sac_res['finMarginCoef'] = finMarginCoef

            def fn(x):
                return fnold(x) / GRfact

            self.sac_res['fn'] = fn

            self.sac_res['Gres'] = Gres @ np.diag(1/GRfact[1:])

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

