from typing import Union
import numpy as np
import pandas as pd
# need to specify SACOBRA_Py.src as source folder in File - Settings - Project Structure,
# then the following import statements will work:
from rescaleWrapper import RescaleWrapper
from initDesigner import InitDesigner
from innerFuncs import verboseprint
from opt.sacOptions import SACoptions
from opt.isaOptions import ISAoptions, ISAoptions0, ISAoptions2

# long and short DRC:
# DRCL: Distance Requirement Cycle, long version:
DRCL = np.array([0.3,0.05, 0.001, 0.0005,0.0])
# DRCS: Distance Requirement Cycle, short version:
DRCS = np.array([0.001,0.0])


class CobraInitializer:
    """
        Initialize SACOBRA optimization:

        - problem formulation: xStart, fn, lower, upper, is_equ, solu
        - parameter settings: s_opts
        - create initial design: A, Fres, Gres
    """
    def __init__(self, xStart, fn, fName, lower, upper,
                 is_equ: Union[np.ndarray, None] = None,
                 solu: Union[np.ndarray, None] = None, s_opts=SACoptions(50),
                 ):
        """

        :param xStart: start point, its dimension defines input space dimension
        :param fn:  function returning (1+nConstraints)-dim vector: [objective to minimize, constraints]
        :param fName: function name
        :param lower: lower bound (same dim as xStart)
        :param upper: upper bound (same dim as xStart)
        :param is_equ: nConstraints-dim boolean vector: which constraints are equality constraints
        :param solu:  (optional, for diagnostics) true solution vector or solution matrix: which x are feasible
                    and deliver minimal objective value
        :param s_opts: the options, see :class:`SACoptions`
        """
        #
        # STEP 0: first settings and checks
        #
        dimension = xStart.size
        if s_opts.ID.initDesPoints is None:
            s_opts.ID.initDesPoints = 2 * dimension + 1
        if s_opts.XI is None:
            s_opts.XI = DRCL
        # The threshold parameter for the number of consecutive iterations that yield ...
        s_opts.Tfeas = np.floor(2 * np.sqrt(dimension))  # ... feasible solutions before the margin is reduced
        s_opts.Tinfeas=np.floor(2 * np.sqrt(dimension))  # ... infeasible solutions before the margin is increased

        lower = np.array(lower)
        upper = np.array(upper)
        assert (lower < upper).all(), "CobraInitializer: lower < upper violated"
        assert not np.isnan(xStart).any(), "CobraInitializer: xStart contains NaNs"
        assert len(xStart.shape) == 1, "CobraInitializer: xStart is not 1-dimensional"
        assert s_opts.ID.initDesPoints < s_opts.feval, "CobraInitializer: Too many init design points"

        #
        # STEP 1: (optional) rescaling
        #
        originalfn = fn
        originalXStart = xStart
        originalSolu = solu
        originalL = lower
        originalU = upper
        lb = np.repeat(s_opts.ID.newLower, dimension)
        ub = np.repeat(s_opts.ID.newUpper, dimension)
        self.rw = RescaleWrapper(originalfn, originalL, originalU, lb, ub)

        if s_opts.ID.rescale:
            xStart = np.array([np.interp(xStart[i],(lower[i],upper[i]), (lb[i],ub[i]))
                               for i in range(dimension)])
            if solu is not None:
                solu = np.array([np.interp(solu[i], (lower[i], upper[i]), (lb[i], ub[i]))
                                for i in range(dimension)])
            fn = self.rw.apply
            lower = lb
            upper = ub

        #
        # STEP 2: second settings and checks
        #
        fn_xStart = fn(xStart)
        nConstraints = fn_xStart.size - 1
        CONSTRAINED = (nConstraints > 0)
        if is_equ is None:      # the default assumes that all constraints are inequality constraints:
            is_equ = np.zeros(nConstraints, dtype=bool)
        assert is_equ.size == nConstraints, f"Wrong size is_equ.size = {is_equ.size}"
        ##assert not CONSTRAINED or not conditioningAnalysis.active, \
        #        "This version does not support conditioning analysis for constrained problems "

        # if not CONSTRAINED:
            # assert not TrustRegion, \
            #     "cobraInit: This version does not support trust Region functionality for unconstrained Problems"

        if not CONSTRAINED:
            verboseprint(verbose=s_opts.verbose, important=True, message="An unconstrained problem is being addressed")

        l = min(upper - lower)  # length of smallest side of search space
        if s_opts.epsilonInit is None: s_opts.epsilonInit = 0.005 * l
        if s_opts.epsilonMax is None:  s_opts.epsilonMax = 2 * 0.005 * l
        if s_opts.ID.initDesOptP is None: s_opts.ID.initDesOptP = s_opts.ID.initDesPoints
        s_opts.phase1DesignPoints = None

        #
        # STEP 3: create initial design
        #
        # set.seed(s_opts.cobraSeed)    # TODO: RNG seeding
        # TODO archive functionality
        A, Fres, Gres = InitDesigner(xStart, fn, lower, upper, s_opts)()
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
                        'xStart': xStart,
                        'dimension': dimension,
                        'originalfn': originalfn,
                        'originalL': originalL,
                        'originalU': originalU,
                        'originalXStart': originalXStart,
                        'originalSolu': originalSolu,
                        'solu': solu,
                        'is_equ': is_equ,
                        'iteration': 0,
                        'fe': fe,
                        'A': A,
                        'Fres': Fres,
                        'Gres': Gres,
                        'nConstraints': nConstraints,
                        'l': l
                        }
        self.for_rbf = {'A': A,
                        'Fres': Fres,
                        'Gres': Gres
                        }

        # TODO: cobra$equIndex
        # TODO: default settings DEBUG_RBF, CA, MS, RI, TR

        #
        # STEP 5: compute numViol, maxViol
        #
        z0 = np.zeros(Gres.shape[0])
        numViol = np.sum(Gres>0, axis=1) if Gres.size>0 else z0   # number of initial constraint violations
        maxViol = np.max(np.maximum(Gres,0), axis=1) if Gres.size>0 else z0    # maximum of initial constraint violation
        trueMaxViol = maxViol
        # currentEps = # TODO: clarify if we need sac_res['currentEps']

        equ_index = np.flatnonzero(self.sac_res['is_equ'] == True)
        if equ_index.size == 0:
            s_opts.EQU.active = False
        if s_opts.EQU.active:
            raise NotImplementedError("[CobraInit] Branch s_opts.EQU.active not yet implemented!")
            # TODO: equality-related changes to numViol, maxViol, trueMaxViol (if(equHandle$active))
            # [this is the branch starting in cobraInit.R with "else if(equHandle$active)" (line 721 ff)]

        #
        # STEP 6: best feasible/infeasible solution, based on numViol
        #
        if 0 in numViol:
            # if there are feasible points, select among them the one with minimal Fres:
            fbest = min(Fres[numViol==0])
            xbest = A[Fres == fbest,:]
            ibest = np.flatnonzero(Fres==fbest)[0]
        else:
            # if there is no feasible point yet: take the set of points with min number of violated constraints ...
            minNumIndex = np.flatnonzero(numViol == min(numViol))
            # ... and select from them the one with minimal Fres:
            FresMin = Fres[minNumIndex]
            ind = np.flatnonzero(FresMin == min(FresMin))[0]
            index = minNumIndex[ind]
            fbest = Fres[index]
            xbest = A[index,:]
            ibest = index
        fbestArray = np.repeat(fbest, s_opts.ID.initDesPoints)
        xbestArray = np.tile(xbest, (s_opts.ID.initDesPoints, 1))

        #
        # STEP 7: update structures
        #
        self.phase = "init"
        self.sac_opts = s_opts
        phaseVec = np.repeat(self.phase, s_opts.ID.initDesPoints)
        sac_res2 = {'numViol': numViol,
                    'maxViol': maxViol,
                    'trueMaxViol': trueMaxViol,
                    'predC': None,
                    'fbest': fbest,
                    'xbest': xbest,
                    'ibest': ibest,
                    'fbestArray': fbestArray,
                    'xbestArray': xbestArray,
                    'phase': phaseVec,
                    'xStart': xbest,         # bug fix
                    'rs': np.array([])      # will be populated by random_start
                    }
        self.sac_res.update(sac_res2)       # update dict sac_res with the content of sac_res2
        self.df = None
        self.df2 = pd.DataFrame()

        # TODO:
        # cobra$equIndex = which(colnames(Gres) == "equ")
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
                    print("Warning: XI is different from default (DRCL), but sac$aDRC==TRUE, so XI will be set by automatic DRC adjustment!")
                elif np.any(s_opts.XI != DRCL):
                    print("Warning: XI is different from default (DRCL), but sac$aDRC==TRUE, so XI will be set by automatic DRC adjustment!")


                verboseprint(s_opts.verbose, important=False, message="adjusting DRC")
                DRC = self.adDRC(max(self.sac_res['Fres']), min(self.sac_res['Fres']))
                s_opts.XI = DRC

            # --- adFit is now called in *each* iteration of cobraPhaseII (adaptive plog) ---

            self.sac_res['GRfact'] = 1   # will be later overwritten by adCon, when the TODO 'equalityConstraints' is done.
            self.sac_res['finMarginCoef'] = 1  # also set in adCon, used in modifyEquCons.R
            if s_opts.ISA.aCF and nConstraints != 0:
                verboseprint(s_opts.verbose, important=False, message="adjusting constraint functions")
                self.adCon()
                # cobra$fn = fn

            self.sac_res['RSDONE'] = np.repeat(np.nan, s_opts.ID.initDesPoints)

        self.sac_opts = s_opts

    def get_sac_opts(self) -> SACoptions:
        return self.sac_opts

    def get_sac_res(self) -> dict:
        return self.sac_res

    def adDRC(self, maxF, minF):
        """ Adjust DRC (distance requirement cycle) """
        FRange = (maxF - minF)
        if (FRange > 1e+03):
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

        GRL = np.apply_along_axis(self.detLen, axis=0, arr=Gres)
        # axis=0 means that arr is sliced along axis 0, i.e. detLen is applied to the columns of Gres
        if min(GRL) == 0:   # pathological case where at least one constraint is constant:
            GR = -np.inf    # inhibit constraint normalization
        else:
            GR = max(GRL) / min(GRL)

        if GR > s_opts.ISA.TGR:
            verboseprint(s_opts.verbose, True, "Normalizing Constraint Functions ...")
            GRfact = np.hstack((1, GRL * (1 / np.mean(GRL))))
            # TODO
            #
            # # finding the normalizing coefficient of the equality constraints
            # if (length(cobra$equIndex) != 0:
            #     GRF = GRfact[-1]
            #     EQUfact = GRF[cobra$equIndex]
            #     # define a coefficient for the final equality margin
            #     equEpsFinal = cobra$equHandle$equEpsFinal
            #     finMarginCoef = min(c(equEpsFinal / EQUfact, equEpsFinal)) / equEpsFinal
            #     cobra$equHandle$equEpsFinal = finMarginCoef * equEpsFinal
            #     cobra$GRfact = GRF
            #     cobra$finMarginCoef = finMarginCoef

            def fn(x):
                return fnold(x) / GRfact

            self.sac_res['fn'] = fn

            self.sac_res['Gres'] = Gres @ np.diag(1/GRfact[1:])

        # TODO
        #
        # cobra$Grange = mean(GRL)
        # equIndex = which(names(GRL) == "equ")
        # cobra$GrangeEqu = mean(GRL[equIndex])

    def detLen(self, x):
        maxL = max(x)
        minL = min(x)
        return maxL - minL

    def detLen2(x):
        # never used
        maxL = np.quantile(x, 0.9)
        minL = np.quantile(x, 0.1)
        return (maxL - minL) / 2

