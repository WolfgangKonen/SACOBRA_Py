import numpy as np
# need to specify SACOBRA_Py.src as source folder in File - Settings - Project Structure,
# then the following import statements will work:
from cobraInit import CobraInitializer
from phase2Vars import Phase2Vars
from innerFuncs import verboseprint


def fitFuncPenalRBF(x):
    ### --- should later go into innerFuncs, but think about EPS and ro and fn
    dummy = 0


def selectXStart(cobra: CobraInitializer):
    dummy = 0


def updateInfoAndCounters(cobra: CobraInitializer, p2: Phase2Vars, phase, currentEps=0):
    """
        Update cobra information (A, Fres, Gres and others) and update counters (Cfeas, Cinfeas).
    """
    def concat(a, b):
        return np.concatenate((a, b), axis=None)

    cobra.sac_res['A'] = np.vstack((cobra.sac_res['A'], p2.ev1.xNew))
    # cobra$TA = rbind(cobra$TA,xNew)
    cobra.sac_res['Fres'] = concat(cobra.sac_res['Fres'], p2.ev1.xNewEval[0])
    cobra.sac_res['Gres'] = np.vstack((cobra.sac_res['Gres'], p2.ev1.xNewEval[1:]))
    # cobra$currentEps<-c(cobra$currentEps,currentEps)  # TODO: clarify if we need cobra$currentEps
    cobra.sac_res['numViol'] = concat(cobra.sac_res['numViol'], p2.ev1.newNumViol)
    cobra.sac_res['maxViol'] = concat(cobra.sac_res['maxViol'], p2.ev1.newMaxViol)
    cobra.sac_res['trueMaxViol'] = concat(cobra.sac_res['trueMaxViol'], p2.ev1.trueMaxViol)
    cobra.sac_res['phase'] = concat(cobra.sac_res['phase'], phase)
    cobra.sac_res['predC'] = p2.ev1.predC

    num = cobra.sac_res['A'].shape[0]
    curr_important = num % cobra.sac_opts.verboseIter == 0
    cobra.sac_opts.important =curr_important

    xNewIndex = cobra.sac_res['numViol'].size - 1
    DEBUGequ = (cobra.sac_opts.EQU.active and cobra.sac_opts.verbose == 2)
    verbose = cobra.sac_opts.verbose
    verboseprint(verbose, important = DEBUGequ,
                 message = f"{phase}.[{num}]: {cobra.sac_res['A'][xNewIndex, 0]} | {cobra.sac_res['Fres'][-1]} |"
                           f"{p2.ev1.newMaxViol} | {currentEps}")

    dim = cobra.sac_res['A'].shape[1]
    realXbest = cobra.rw.inverse(cobra.sac_res['xbest'].reshape(dim,))
    if cobra.sac_opts.EQU.active:
        verboseprint(verbose, important = cobra.sac_opts.important,
                     message = f"Best Result.[{num}]: {realXbest[0]} {realXbest[1]} | {cobra.sac_res['fbest'][0]} | "
                               f"{cobra.sac_res['trueMaxViol'][cobra.sac_res['ibest']]} |  {currentEps}")

    else:
        verboseprint(verbose, important=cobra.sac_opts.important,
                     message=f"Best Result.[{num}](TODO: archive): {realXbest[0]} {realXbest[1]} | {cobra.sac_res['fbest']} | "
                             f"{cobra.sac_res['trueMaxViol'][cobra.sac_res['ibest']]}")

    if cobra.sac_res['numViol'][-1] == 0:
        p2.Cfeas += 1
        p2.Cinfeas = 0
    else:
        p2.Cinfeas += 1
        p2.Cfeas = 0

