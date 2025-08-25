import numpy as np
# need to specify SACOBRA_Py.src as source folder in File - Settings - Project Structure,
# then the following import statements will work:
from cobraInit import CobraInitializer
from phase2Vars import Phase2Vars
from innerFuncs import verboseprint, distLine
from equHandling import modifyMu


def fitFuncPenalRBF(x):
    ### --- should later go into innerFuncs, but think about EPS and ro and fn
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

    :param cobra:   SACOBRA settings and results
    :param p2:      these members may be changed : ``EPS``, ``currentMu``, ``Cfeas``, ``Cinfeas``
    """
    Tfeas = cobra.sac_opts.Tfeas
    Tinfeas = cobra.sac_opts.Tinfeas
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