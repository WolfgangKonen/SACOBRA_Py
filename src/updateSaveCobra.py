import numpy as np
from cobraInit import CobraInitializer
from cobraPhaseII import CobraPhaseII
from phase2Vars import Phase2Vars
from evaluatorReal import getPredY0
from innerFuncs import plogReverse


def updateSaveCobra(cobra: CobraInitializer, p2: Phase2Vars, gama, EPS,
                    fitFuncPenalRBF, distRequirement, fitnessSurrogate=None, sigmaD=None, penaF=None):
    """
        Update and save cobra.

        Helper for :class:`CobraPhaseII`: make some assertion tests,
        update elements in object ``cobra``, including data frames ``df`` and ``df2``.
        If ``cobra.sac_opts.saveIntermediate==True``, save cobra and p2 in subdirectory ``results/``.

        Most importantly, the items with keys  ``xbest, fbest, ibest`` in dictionary ``cobra.sac_res`` are updated.
        They characterize the best feasible point (least violating point if no feasible point was found so far) and
        influence the next starting point.

        Note: the items with keys ``A, Fres, Gres`` in dictionary ``cobra.sac_res`` are set in
        ``updateInfoAndCounters``, an internal function of :class:`CobraPhaseII`.

    """

    def concat(a, b):
        return np.concatenate((a, b), axis=None)

    if fitnessSurrogate is None:
        fitnessSurrogate = p2.fitnessSurrogate

    s_opts = cobra.sac_opts
    s_res = cobra.sac_res
    xNew = p2.ev1.xNew
    feas = p2.ev1.feas
    feasPred = p2.ev1.feasPred
    feval = p2.ev1.feval
    optimConv = p2.ev1.optimConv
    predY = p2.ev1.predY
    predVal = p2.ev1.predVal
    diff = s_res['A'].shape[0] - predY.size

    df_RS  = cobra.df.RS
    df_RS  =  concat(df_RS, not np.all(s_res['xbest'] == s_res['xStart']))
    # TODO:
    # if (cobra$DEBUG_XI) {
    #     df_fxStart <- c(cobra$df$fxStart,cobra$fn(cobra$xStart)[1])
    #     df_fxbest <-  c(cobra$df$fxbest,cobra$fn(cobra$xbest)[1])
    #     df_RS2 <-  c(cobra$df$RS2,cobra$DEBUG_RS)
    # }


    # TODO:
    # if (cobra$WRITE_XI) {
    #     ro < -gama * cobra$l
    #     sumViol < - distRequirement(xNew, cobra$fitnessSurrogate, ro)$sumViol
    #     if ( is.null(cobra$df)) {
    #         df_XI < - c(rep(NA, cobra$initDesPoints), gama)
    #         df_XIsumViol < - c(rep(NA, cobra$initDesPoints), sumViol)
    #     } else {
    #         df_XI < - c(cobra$df$XI, gama)
    #         df_XIsumViol < - c(cobra$df$XIsumViol, sumViol)
    #     }
    # }

    xNewIndex = cobra.sac_res['numViol'].size

    if s_opts.EQU.active and np.flatnonzero(s_res['is_equ']).size != 0:
        # ... if we handle equality constraints by 'equMargin & two inequality constraints'
        # and have equality constraints in the actual problem:
        # calculate cobra$xbest,fbest,ibest via updateCobraEqu:
        raise NotImplementedError("[updateSaveCobra] updateCobraEqu not (yet) implemented")
        # TODO:
        # cobra < -updateCobraEqu(cobra, xNew)  # in modifyEquCons.R
    elif s_res['numViol'][s_res['ibest']] == 0:  # if the so-far best is feasible...
        assert s_res['Fres'][s_res['ibest']] == s_res['fbest']
        if s_res['numViol'][xNewIndex] == 0 and s_res['Fres'][xNewIndex] < s_res['fbest']:
            # If xNew is feasible and even better:
            cobra.sac_res['xbest'] = xNew
            cobra.sac_res['fbest'] = s_res['Fres'][xNewIndex]
            cobra.sac_res['ibest'] = xNewIndex
    else:  # if we have not yet an everBestFeasible ...
        if s_res['numViol'][xNewIndex] == 0:
            # the new point is feasible then we select it
            cobra.sac_res['xbest'] = xNew  # ... take xNew, if it is feasible
            cobra.sac_res['fbest'] = s_res['Fres'][xNewIndex]
            cobra.sac_res['ibest'] = xNewIndex
        else:  # new solution is infeasible: look for the best infeasible solu
            if s_res['maxViol'][xNewIndex] < s_res['maxViol'][s_res['ibest']]:
                cobra.sac_res['xbest'] = xNew  # ... take xNew, if it has smaller maxViol
                cobra.sac_res['fbest'] = s_res['Fres'][xNewIndex]
                cobra.sac_res['ibest'] = xNewIndex
    # If we do not have a feasible point AND xNew has a larger maxMivol than ibest, then leave the
    # triple (xbest,fbest,ibest) at the setting of cobraInit.py, line 151-159: From all points
    # with minimum number of violated constraints, take the one with smallest Fres.


    cobra.sac_res['fbestArray'] = concat(s_res['fbestArray'], s_res['fbest'])
    cobra.sac_res['xbestArray'] = np.vstack((s_res['xbestArray'], s_res['xbest']))

    # --- commented out since feasibleIndices and xbestIndex are never used:
    # feasibleIndices = np.flatnonzero(np.max(s_res['Gres'],axis=1) <= 0)
    # xbestIndex = which.min(cobra$Fres[feasibleIndices])  # finding index of the best point so far

    # only diagnostics, needed for cobra$df & cobra$df2 /WK/
    solu = s_res['solu']
    if solu is None:
        solu= p2.opt_res['x']  # p2.opt_res['x']: the optimal solution found so far
        soluOrig = cobra.rw.inverse(p2.opt_res['x'], cobra)
    else:
        if s_opts.ID.rescale:
            if solu.ndim==2:
                solu = np.apply_along_axis(lambda x: cobra.rw.forward(x, cobra), axis=1, arr=solu)
            else:
                solu = cobra.rw.forward(solu, cobra)
        soluOrig = s_res['solu']
    # now solu is always in *rescaled* input space

    predSoluFunc = lambda x: getPredY0(x, fitnessSurrogate, p2)
    if solu.ndim==2:      # in case of multiple global optima in solu:
        predSolu = np.apply_along_axis(predSoluFunc, axis=1, arr=solu)
        predSoluPenal = np.apply_along_axis(fitFuncPenalRBF, axis=1, arr=solu)
    else:
        predSolu = predSoluFunc(solu);
        predSoluPenal = fitFuncPenalRBF(solu);

    predSolu = min(predSolu)    # Why min? - In case of multiple global optima: predSolu is the
                                # value of predSoluFunc at the best solution solu
    predSoluPenal = min(predSoluPenal)
    if cobra.df is None:
        df_predSolu = concat(np.repeat(np.nan, s_opts.ID.initDesPoints), predSolu)
    else:
        df_predSolu = concat(cobra.df.predSolu, predSolu)
