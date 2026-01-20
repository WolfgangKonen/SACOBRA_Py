import numpy as np
# need to specify SACOBRA_Py.src as source folder in File - Settings - Project Structure,
# then the following import statements will work:
from cobraInit import CobraInitializer
from phase2Vars import Phase2Vars
from innerFuncs import verboseprint, distLine
from equHandling import modifyMu
from surrogator import Surrogator
from updateSaveCobra import updateSaveCobra


def fitFuncPenalRBF(x):
    # --- should later go into innerFuncs, but think about EPS and ro and fn
    return np.array([0])
    # TODO (from cobraPhaseII.R):
    # if (any( is.nan(x))){
    #     warning("fitFuncPenalRBF: x value is NaN, returning Inf")
    #     return (Inf)
    # }
    # y = interpRBF(x, cobra$fitnessSurrogate)
    # if (cobra$trueFuncForSurrogates) y < -fn(x)[1]
    # penalty < -0
    # if (cobra$CONSTRAINED){
    #     constraintPrediction < -  interpRBF(x, cobra$constraintSurrogates) +EPS ^ 2
    #     if (cobra$trueFuncForSurrogates) constraintPrediction < -  fn(x)[-1]+EPS ^ 2
    #     violatedConstraints = which(constraintPrediction > 0)
    #     penalty = sum(constraintPrediction[violatedConstraints])
    # }
    #
    # penalty = penalty + distRequirement(x, cobra$fitnessSurrogate, cobra$ro)$sumViol * sigmaD[1]
    # return (y + penalty * penaF[1])


def selectXStart(cobra: CobraInitializer):
    dummy = 0


def distRequirement(x,fitnessSurrogate,ro):
    dummy = 0
    # TODO:
    # distRequirement<- function(x,fitnessSurrogate,ro) {
    #     ed = ro - distLine(x,fitnessSurrogate$xp)
    #     violatedDist = which(ed>0)
    #     sumViol = sum(ed[violatedDist])
    #     return(list(ed=ed,   # vector of euclidean distances
    #                 violatedDist=violatedDist,
    #                 sumViol=sumViol))
    # }


def updateInfoAndCounters(cobra: CobraInitializer, p2: Phase2Vars, currentMu=0):
    """
        Update cobra information (A, Fres, Gres and others) and update counters (Cfeas, Cinfeas).
    """
    def concat(a, b):
        return np.concatenate((a, b), axis=None)

    # LinAlgError ('Singular Matrix') is raised by RBFInterpolator if matrix A contains identical rows
    # (identical infill points). We avoid this with cobra.for_rbf['A'] (instead of cobra.sac_res['A']),
    # where a new infill point is added in updateInfoAndCounters ONLY if min(xNewDist), the minimum Euclid distance
    # of the new infill points to all rows of cobra.for_rbf['A'] is greater than a small thresh:
    xNewDist = distLine(p2.ev1.xNew, cobra.sac_res['A'])
    if min(xNewDist) >  1e-9: # 0.0:     # a value 1e-9 is needed by G04 to avoid LinAlgError
        cobra.for_rbf['A'] = np.vstack((cobra.for_rbf['A'], p2.ev1.xNew))
        cobra.for_rbf['Fres'] = concat(cobra.for_rbf['Fres'], p2.ev1.xNewEval[0])
        cobra.for_rbf['Gres'] = np.vstack((cobra.for_rbf['Gres'], p2.ev1.xNewEval[1:]))
        # The new elements of dict cobra.for_rbf are used in trainSurrogates as a safe replacement for the
        # former elements of cobra.sac_res.
    # The elements cobra.sac_res['A', 'Fres', 'Gres'] are filled in any case to keep track of every iteration.
    cobra.sac_res['A'] = np.vstack((cobra.sac_res['A'], p2.ev1.xNew))
    # cobra$TA = rbind(cobra$TA,xNew)
    cobra.sac_res['Fres'] = concat(cobra.sac_res['Fres'], p2.ev1.xNewEval[0])
    cobra.sac_res['Gres'] = np.vstack((cobra.sac_res['Gres'], p2.ev1.xNewEval[1:]))
    cobra.sac_res['muVec'] = concat(cobra.sac_res['muVec'], p2.currentMu)
    cobra.sac_res['numViol'] = concat(cobra.sac_res['numViol'], p2.ev1.newNumViol)
    cobra.sac_res['trueNumViol'] = concat(cobra.sac_res['trueNumViol'], p2.ev1.trueNumViol)
    cobra.sac_res['maxViol'] = concat(cobra.sac_res['maxViol'], p2.ev1.newMaxViol)
    cobra.sac_res['trueMaxViol'] = concat(cobra.sac_res['trueMaxViol'], p2.ev1.trueMaxViol)
    cobra.sac_res['phase'] = concat(cobra.sac_res['phase'], cobra.phase)
    cobra.sac_res['predC'] = p2.ev1.predC

    p2.num = cobra.sac_res['A'].shape[0]
    curr_important = p2.num % cobra.sac_opts.verboseIter == 0
    cobra.sac_opts.important =curr_important

    xNewIndex = cobra.sac_res['numViol'].size - 1
    DEBUGequ = (cobra.sac_opts.EQU.active and cobra.sac_opts.verbose == 2)
    verbose = cobra.sac_opts.verbose
    verboseprint(verbose, important = DEBUGequ,
                 message = f"{cobra.phase}.[{p2.num}]: {cobra.sac_res['A'][xNewIndex, 0]} | "
                           f"{cobra.sac_res['Fres'][-1]} | {p2.ev1.newMaxViol} | {currentMu}")

    dim = cobra.sac_res['A'].shape[1]
    realXbest = cobra.rw.inverse(cobra.sac_res['xbest'].reshape(dim,))
    if cobra.sac_opts.EQU.active:
        verboseprint(verbose, important = cobra.sac_opts.important,
                     message = f"Best Result.[{p2.num}]: {realXbest[0]} {realXbest[1]} | {cobra.sac_res['fbest']} | "
                               f"{cobra.sac_res['trueMaxViol'][cobra.sac_res['ibest']]} |  {currentMu}")

    else:
        # TODO: add the part with 'nrow(get("ARCHIVE",envir=intern.archive.env))' to the following message:
        verboseprint(verbose, important=cobra.sac_opts.important,
                     message=f"Best Result.[{p2.num}]: {realXbest[0]} {realXbest[1]} | {cobra.sac_res['fbest']} | "
                             f"{cobra.sac_res['trueMaxViol'][cobra.sac_res['ibest']]}")

    if cobra.sac_res['numViol'][-1] == 0:
        p2.Cfeas += 1
        p2.Cinfeas = 0
    else:
        p2.Cinfeas += 1
        p2.Cfeas = 0


# NOTE: We cannot have adjustMargins(self, cobra) in phase2Vars, because adjustMargins needs to import equHandling
# and equHandling needs to import phase2Vars. The way to avoid this circular import is to have adjustMargins in
# this separate module phase2Funcs.
def adjustMargins(cobra: CobraInitializer, p2: Phase2Vars):
    """
    Adjust margins :math:`\\epsilon =` ``p2.EPS``, :math:`\\mu =` ``p2.currentMu`` and
    :math:`\\rho =` ``cobra.sac_opts.RBF.rho``; conditionally reset counters ``p2.Cfeas``, ``p2.Cinfeas``.

    If ``p2.Cfeas >= Tfeas``, then halve :math:`\\epsilon`.

    If ``p2.Cinfeas >= Tinfeas``, then double :math:`\\epsilon` and clip it at ``epsilonMax``.

    ``Tfeas``, ``Tinfeas``, ``epsilonMax`` are members of  ``cobra.sac_opts.SEQ``.

    :param cobra:   SACOBRA settings and results
    :param p2:      these members may be changed : ``EPS``, ``currentMu``, ``Cfeas``, ``Cinfeas``
    """
    Tfeas = cobra.sac_opts.SEQ.Tfeas
    Tinfeas = cobra.sac_opts.SEQ.Tinfeas
    verbose = cobra.sac_opts.verbose
    if p2.Cfeas >= Tfeas:
        p2.EPS = p2.EPS / 2
        verboseprint(verbose, important = False, message=f"reducing epsilon to {p2.EPS}")
        verboseprint(verbose, important = False, message=f"reducing equality margin to {p2.currentMu}")

        p2.Cfeas = 0

    if p2.Cinfeas >= Tinfeas:
        p2.EPS = min(2 * p2.EPS, cobra.sac_opts.SEQ.epsilonMax)
        verboseprint(verbose, important=False, message=f"increasing epsilon to {p2.EPS}")
        verboseprint(verbose, important=False, message=f"increasing equality margin to {p2.currentMu}")

        p2.Cinfeas = 0

    if cobra.sac_opts.EQU.active:
        p2.currentMu = modifyMu(p2.Cfeas, p2.Cinfeas, Tfeas, p2.currentMu, cobra, p2)

    if cobra.sac_opts.RBF.rhoGrow > 0:
        if p2.num % cobra.sac_opts.RBF.rhoGrow == 0:
            cobra.sac_opts.RBF.rho = cobra.df2['rho'].values[0]  # every rhoGrow (e.g. 100) iterations, re-enlarge rho

    cobra.sac_opts.RBF.rho /= cobra.sac_opts.RBF.rhoDec


# def check_gReal_eps1(cobra: CobraInitializer, p2: Phase2Vars):
#     GRfact = cobra.sac_res['GRfact']
#     gReal = p2.ev1.xNewEval[1:] * GRfact
#     equ_ind = np.flatnonzero(cobra.sac_res['is_equ'])  # index to all equality constraints
#     equ2Index = np.concatenate((equ_ind, cobra.sac_res['nConstraints'] + np.arange(equ_ind.size)), axis=None)
#     gReal = np.concatenate((gReal, -gReal[equ_ind]), axis=None)
#     gReal[equ2Index] -= p2.currentMu
#     g_arti = constraint_to_artif(p2.ev1.xNewEval[1:], cobra, p2)
#     assert np.allclose(g_arti, gReal)
#     eps1 = p2.ri2.s_opts.RI.eps1
#     if p2.ev1.newNumViol > 0 and not np.any(gReal + eps1 > 0):
#         dummy = 0


def constraint_to_artif(g_val, cobra: CobraInitializer, p2: Phase2Vars):
    """
    Given a vector ``g_val`` with ``n_constraints`` constraint values, return an artificial constraint
    vector ``g_arti`` with ``n_constraints+n_equ`` artificial constraint values: multiplied by ``GRfact``, with
    ``n_equ`` equ-inequalities appended and with ``currentMu`` subtracted from all equ-inequalities.

    :param g_val: constraint vector (size ``n_constraints``)
    :param cobra:
    :param p2:
    :return: artificial constraint vector ``g_arti`` (size ``n_constraints+n_equ``)
    """
    GRfact = cobra.sac_res['GRfact']
    g_arti = (g_val * GRfact).copy()
    equ_ind = np.flatnonzero(cobra.sac_res['is_equ'])  # index to all equality constraints
    equ2Index = np.concatenate((equ_ind, cobra.sac_res['nConstraints'] + np.arange(equ_ind.size)), axis=None)
    g_arti = np.concatenate((g_arti, -g_arti[equ_ind]), axis=None)
    g_arti[equ2Index] -= p2.currentMu
    return g_arti


def conditions_for_repair_met(cobra: CobraInitializer, p2: Phase2Vars) -> bool:
    s_opts = cobra.sac_opts
    ri = s_opts.RI
    if not s_opts.RI.repairInfeas: return False
    if not p2.num < s_opts.feval: return False      # no repair, if we are in last iteration (would result in too many iterations)
    if p2.ev1.newNumViol == 0: return False         # no repair if xNew is feasible anyway
    # print(p2.ev1.newMaxViol)
    if p2.ev1.newMaxViol >= s_opts.RI.repairMargin: return False  # no repair if xNew has a too large max violation

    # check_gReal_eps1(cobra, p2)

    fbest = cobra.get_feasible_best()
    if ri.repairOnlyFresBetter and fbest != np.nan:
        # if we arrive here, we repair only if fitness < so-far-best-fitness + marFres
        do_repair = (cobra.sac_res['Fres'][-1] < fbest + ri.marFres)
    else:
        # if we arrive here, we repair unconditionally
        do_repair = True
    return do_repair


def do_repair_step(cobra: CobraInitializer, p2: Phase2Vars):
    """
    Do repair step for an infeasible point and necessary surrounding updates (surrogates, ``ev1``, ``cobra``,
    info & counters).

    The most important result is that the new infill point ``p2.ev1.xNew`` (infeasible) is replaced with a
    repaired version (hopefully feasible).

    The success of repair (on the surrogates) can be read off from
    ``p2.ev1.state = "repairFailed" | "repaired" | "repairSuccess"``.

    :param cobra:   SACOBRA settings and results (A, Fres, Gres, xbest, fbest, df, df2, ...)
    :param p2:      these members may be changed : ``ev1``, ``Cfeas``, ``Cinfeas``
    """
    # print("repair is called")   # there is a verboseprint in repairInfeasRI2

    # Build surrogate anew, based on current A, Gres
    # This is important for accurate constraint surrogates models near current infeasible point
    p2 = Surrogator.trainSurrogates(cobra, p2)

    x = p2.ev1.xNew
    gReal = p2.ev1.xNewEval[1:]
    RIopt = p2.ri2.s_opts.RI
    z = p2.ri2.repairInfeasRI2(x, gReal, p2.constraintSurrogates, cobra, p2, p2.currentMu, RIopt.checkIt)
    z_is_feas = p2.ri2.is_epsilon_feasible(z, RIopt.eps2, p2.constraintSurrogates)

    if np.all(z == x):
        # verboseprint(cobra.sac_opts.verbose, important=True, message="cannot repair")  # printout already in while
        p2.ev1.state = "repairFailed"
    else:
        p2.ev1.state = "repairSuccess" if z_is_feas else "repaired"
        p2.ev1.update(z, cobra, p2, p2.currentMu)   # set ev1.xNew=z; do update w/o refine, because state!="optimized"
        updateInfoAndCounters(cobra, p2)            # includes increment p2.num
        updateSaveCobra(cobra, p2, p2.EPS, fitFuncPenalRBF, distRequirement)
        if p2.ev1.state == "repairSuccess":
            if p2.num >= 362:
                dummy = 0

        # --- questionable: should a repair step be followed by additional adjustMargins (eps, mu, rho)? - I think NO!
        #     The right way to do adjustMargins is in cobraRepairII, just after the (conditional) repair
        # adjustMargins(cobra, p2)
