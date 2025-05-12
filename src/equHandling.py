import numpy as np
from cobraInit import CobraInitializer
from phase2Vars import Phase2Vars


def updateCobraEqu(cobra: CobraInitializer, p2: Phase2Vars, xNew):
    """
    Calculate ``cobra.sac_res['xbest', 'fbest', 'ibest']`` given the currently evaluated points.

    First, we calculate the set S of points being feasible in the inequalities and feasible
    with margin currentEps in the equalities. If S is non-empty, take the point from S with
    the best objective as ``cobra.sac_res['xbest']``. If S is empty (there are no feasible points): If the new point
    has lower maxViol than the maxViol of ``cobra.sac_res['xbest']``, take it as the new ``cobra.sac_res['xbest']``.

    :param cobra: an object of class :class:`CobraInitializer`
    :param xNew:  the last evaluated point
    :return: ``cobra``, an object of class :class:`CobraInitializer` , with potentially
             modified ``cobra.sac_res['xbest', 'fbest', 'ibest']``
    """
    s_opts = cobra.sac_opts
    s_res = cobra.sac_res
    equ_ind = np.flatnonzero(s_res['is_equ'])
    conTol = s_opts.SEQ.conTol

    equMargin = p2.currentEps   # new 2025/04/02
    temp = s_res['Gres'].copy()
    # We check whether
    #
    #          g_i(x) <= 0,  h_j(x) - equMargin <= 0,    -h_j(x) - equMargin <= 0
    #
    # for each row of cobra$Gres is valid and set only rows fulfilling this to currentFeas==TRUE
    # New /2025/04/02: replace 0 by cobra$conTol
    temp[:, equ_ind] = abs(temp[:, equ_ind]) - equMargin
    currentMaxViols = np.maximum(0, np.max(temp, axis=1))
    # --- this former approach is not bug-free, it has problems with array shape (!): ---
    # temp = np.array([temp, -temp[:, equ_ind]])
    # equ2Index = np.concat((equ_ind, s_res['nConstraints'] + np.arange(0, equ_ind.size)))
    # temp[:, equ2Index] = temp[:, equ2Index] - equMargin
    # currentMaxViols = np.maximum(0, np.max(temp,axis=1))    # np.max(..,axis=1): take the row maximum in each row

    currentFeas = np.flatnonzero(currentMaxViols <= conTol)  # new 2025/04/02: replaces "<=0"

    #   If length(currentFeas)==0, we do the same thing as in updateSaveCobra:
    #   If the new point has a smaller maxViol then we take it as xbest, otherwise
    #   we leave the triple (xbest,fbest,ibest) as before (e.g. as set by cobraInit.R,
    #   line 400: From all points with minimum number of violated constraints, take
    #   the one with smallest Fres.)
    if currentFeas.size == 0:
        ibest = np.flatnonzero(currentMaxViols == np.min(currentMaxViols))[0]
        cobra.sac_res['ibest'] = ibest
        cobra.sac_res['fbest'] = s_res['Fres'][ibest]
        cobra.sac_res['xbest'] = s_res['A'][ibest]
        return cobra

    # Otherwise, if currentFeas.size>0, we take among the feasible solutions the one with
    # minimum objective Fres:
    fminInd = np.flatnonzero(s_res['Fres'][currentFeas] == np.min(s_res['Fres'][currentFeas]))
    # The new cobra$ibest might refer to a previous solution which differs from cobra$ibest
    # so far and is NOT the ibest of the new point!
    # Why? - The so-far ibest might be a solution for an older equMargin band which is no longer
    # valid in the current iteration. Then fminInd searches among the now valid solutions
    # the new best Fres. An older solution might come into play.
    # This is the reason why there can be in cobra$df a line where df$Best changes to a new value,
    # but this value is NOT the df$y of the current iteration (as it used to be the case
    # for inequality constraint handling).
    ibest = currentFeas[fminInd[0]]
    cobra.sac_res['ibest'] = ibest
    cobra.sac_res['xbest'] = s_res['A'][ibest,:]
    cobra.sac_res['fbest'] = s_res['Fres'][ibest]
    cobra.sac_res['currentFeas'] = currentFeas

    return cobra


def modifyMu(Cfeas, Cinfeas, Tfeas, currentEps, cobra: CobraInitializer, p2: Phase2Vars):
    """
    Modify equality margin :math:`\mu` = ``currentEps``.

    :param Cfeas: counter feasible iterates
    :param Cinfeas: counter infeasible iterates
    :param Tfeas: threshold counts
    :param currentEps: current value for :math:`\mu`
    :param cobra: an object of class :class:`CobraInitializer`
    :param p2: an object of class :class:`Phase2Vars`
    :return: ``currentEps``, the modified value for :math:`\mu`
    """
    s_opts = cobra.sac_opts
    s_res = cobra.sac_res
    # s_res['muVec'] holds the vector named cobra$currentEps in R
    if s_opts.EQU.muGrow > 0:
        if p2.num % s_opts.EQU.muGrow == 0:
            currentEps = s_res['muVec'][0]  # every muGrow (e.g. 100) iterations, re-enlarge the \mu-band

    switcher = {
        'expFunc':    # exponentially decaying func
            max(currentEps / s_opts.EQU.dec, s_opts.EQU.equEpsFinal),
        'SAexpFunc':  # self-adjusting expFunc
            max(np.mean([s_res['muVec'][-1]/s_opts.EQU.dec,
                         s_res['trueMaxViol'][s_res['ibest']] * s_res['finMarginCoef'] ]),
                s_opts.EQU.equEpsFinal),
        'funcDim': (s_res['muVec'][0] * (1 / s_opts.EQU.dec) ** ((p2.num - 3 * s_res['A'].shape[1]) /
                                                                 ((Tfeas ** 2) / 2 - 1))
                    ) + s_opts.EQU.equEpsFinal,
        'funcSDim': (s_res['muVec'][0] * (1 / s_opts.EQU.dec) ** ((p2.num - 3 * s_res['A'].shape[1]) / Tfeas)
                     ) + s_opts.EQU.equEpsFinal,
        'Zhang': max(max(s_res['muVec'] * (1 - p2.ev1.feas.size / p2.num)), s_opts.EQU.equEpsFinal),
        'CONS': s_opts.EQU.equEpsFinal
    }
    currentEps = switcher.get(s_opts.EQU.epsType, "Invalid epsType")
    assert currentEps != "Invalid epsType", f"[modifyMu] invalid s_opts.EQU.epsType = {s_opts.EQU.epsType}"

    return currentEps


