import numpy as np
import pandas as pd

from soluContainer import SoluContainer
from cobraInit import CobraInitializer
from phase2Vars import Phase2Vars
from evaluatorReal import getPredY0
from equHandling import updateCobraEqu
# from innerFuncs import distLine


def updateSaveCobra(cobra: CobraInitializer, p2: Phase2Vars, EPS,
                    fitFuncPenalRBF, distRequirement, fitnessSurrogate=None):
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
    optimTime = p2.ev1.optimTime
    predY = p2.ev1.predY
    predVal = p2.ev1.predVal
    diff = s_res['A'].shape[0] - predY.size
    CONSTRAINED = s_res['nConstraints'] > 0

    if cobra.df is None:
        df_RS = concat(np.repeat(False, s_opts.ID.initDesPoints), p2.rs_done)
    else:
        df_RS = concat(cobra.df.RS, p2.rs_done)  # not np.all(s_res['xbest'] == s_res['xStart']))
    # TODO:
    # if (cobra$DEBUG_XI) {
    #     df_fxStart <- c(cobra$df$fxStart,cobra$fn(cobra$xStart)[1])
    #     df_fxbest <-  c(cobra$df$fxbest,cobra$fn(cobra$xbest)[1])
    #     df_RS2 <-  c(cobra$df$RS2,cobra$DEBUG_RS)
    # }

    # TODO:
    if p2.write_XI:
        ro = p2.gama * s_res['l']
        # s_res['l'] is set in cobraInit (length of smallest side of search space)
        # TODO: distRequirement
        # sumViol = pf2.distRequirement(xNew, p2.fitnessSurrogate, ro)$sumViol
        if cobra.df is None:
            df_XI = concat(np.repeat(np.nan, s_opts.ID.initDesPoints), p2.gama)
            # df_XIsumViol = c(rep(NA, cobra$initDesPoints), sumViol)
        else:
            df_XI = concat(cobra.df.XI, p2.gama)
            # df_XIsumViol = c(cobra$df$XIsumViol, sumViol)

    xNewIndex = cobra.sac_res['numViol'].size - 1

    if s_opts.EQU.active and np.flatnonzero(s_res['is_equ']).size != 0:
        # ... if we handle equality constraints by 'equMargin & two inequality constraints'
        # and have equality constraints in the actual problem:
        # calculate cobra$xbest,fbest,ibest via updateCobraEqu:
        cobra = updateCobraEqu(cobra, p2, xNew)  # in equHandling.py
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
        else:  # new solution is infeasible: look for the best infeasible solution
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
    cobra.solution2 = SoluContainer(cobra.solu, cobra)
    predSolu, predSoluPenal = cobra.solution2.predict_at_solu(p2, fitnessSurrogate, fitFuncPenalRBF)
    #
    # --- OLD version: ---
    # solu = s_res['solu']
    # if solu is None:
    #     # TODO: questionable, better leave it as None
    #     solu = p2.opt_res['x']  # p2.opt_res['x']: the optimal solution found so far
    #     soluOrig = cobra.rw.inverse(p2.opt_res['x'])
    # else:
    #     # --- the following was wrong, since solu was already rescaled in cobraInit, if s_opts.ID.rescale
    #     # if s_opts.ID.rescale:
    #     #     if solu.ndim == 2:
    #     #         solu = np.apply_along_axis(lambda x: cobra.rw.forward(x), axis=1, arr=solu)
    #     #     else:
    #     #         solu = cobra.rw.forward(solu)
    #     # soluOrig = s_res['solu']
    #     soluOrig = s_res['originalSolu']
    # # now solu is always in *rescaled* input space
    #
    # predSoluFunc = lambda x: getPredY0(x, fitnessSurrogate, p2)
    # if solu.ndim == 2:      # in case of multiple global optima in solu:
    #     predSolu = np.apply_along_axis(predSoluFunc, axis=1, arr=solu)
    #     predSoluPenal = np.apply_along_axis(fitFuncPenalRBF, axis=1, arr=solu)
    # else:
    #     predSolu = predSoluFunc(solu)
    #     predSoluPenal = fitFuncPenalRBF(solu)
    #
    # predSolu = min(predSolu)    # Why min? - In case of multiple global optima: predSolu is the
    #                             # value of predSoluFunc at the best solution solu
    # predSoluPenal = min(predSoluPenal)

    if cobra.df is None:
        df_predSolu = concat(np.repeat(np.nan, s_opts.ID.initDesPoints), predSolu)
    else:
        df_predSolu = concat(cobra.df.predSolu, predSolu)

    # calculate distA and distOrig:
    distA, distOrig = cobra.solution2.distance_to_solu(cobra)
    #
    # --- OLD version: ---
    # A = s_res['A']
    # origA = np.apply_along_axis(lambda x: cobra.rw.inverse(x), axis=1, arr=s_res['A'])
    # # if (cobra$dimension == 1) origA = t(origA)
    # if solu.ndim == 2:   # this is for the case with multiple solutions (like in G11)
    #     distA, distOrig = cobra.solution.distanceSolution()
    # else:
    #     distA = distLine(solu, s_res['A'])  # distance in rescaled space, distLine: see RbfInter.R
    #     distOrig = distLine(soluOrig, origA)  # distance in original space

    # several assertions
    assert s_res['Fres'].shape[0] == predY.size, "[updateSaveCobra] predY"
    assert df_predSolu.size == optimConv.size, "[updateSaveCobra] optimConv"
    assert s_res['Fres'].shape[0] == optimConv.size, "[updateSaveCobra] optimConv 2"
    assert s_res['Fres'].shape[0] == optimTime.size, "[updateSaveCobra] optimTime"
    assert s_res['Fres'].shape[0] == s_res['fbestArray'].size, "[updateSaveCobra] fbestArray"
    if CONSTRAINED:
        assert s_res['Fres'].shape[0] == feas.size, "[updateSaveCobra] feas"
        assert s_res['Fres'].shape[0] == feasPred.size, "[updateSaveCobra] feasPred"
        assert s_res['Fres'].shape[0] == s_res['numViol'].size, "[updateSaveCobra] numViol"
        assert s_res['Fres'].shape[0] == s_res['maxViol'].size, "[updateSaveCobra] maxViol"

    # result data frame df:
    if CONSTRAINED:
        cobra.df = pd.DataFrame(
            {'iter': np.arange(predY.size),
             'y': s_res['Fres'],
             'predY': predY,  # surrogate fitness
             'predSolu': df_predSolu,
             'feasible': feas,
             'feasPred': feasPred,
             'nViolations': s_res['numViol'],
             'trueNViol': s_res['trueNumViol'],
             'maxViolation': s_res['maxViol'],
             'trueMaxViol': s_res['trueMaxViol'],
             'FEval': feval,
             'Best': s_res['fbestArray'],
             'optimizer': np.repeat(s_opts.SEQ.optimizer, s_res['Fres'].shape[0]),
             'optimConv': optimConv,
             'optimTime': optimTime,
             'dist': distA,         # distance of solu to infill points, rescaled space (min dist for multiple solu's)
             'distOrig': distOrig,  # the same, but in original space
             'RS': df_RS,       # TRUE, if it is an iteration with random start point
             })
    else:   # i.e. if not CONSTRAINED:
        raise NotImplementedError("[updateSaveCobra] Branch df for CONSTRAINED==False not (yet) implemented")
        # TODO:
        # if (is.null(cobra$df$realfeval))
        #     realfeval < -c()
        # else
        #     realfeval < -cobra$df$realfeval
        #
        # df < - data.frame(
        #     y=cobra$Fres,
        #     predY = predY,  # surrogate fitness
        #     predSolu = df_predSolu,
        #     feasible = T,
        #     FEval = feval,
        #     realfeval = c(realfeval, nrow(get("ARCHIVE", envir=intern.archive.env))),
        #     Best = cobra$fbestArray,
        #     optimizer = rep(cobra$seqOptimizer, length(cobra$Fres)),
        #     optimizationTime = ev1$optimizationTime,
        #     conv = optimizerConvergence,
        #     dist = distA,
        #     distOrig = distOrig,
        #     RS = df_RS,  # TRUE, if it is an iteration with random start point
        #     row.names = NULL
        # )

    if p2.write_XI:
        cobra.df['XI'] = df_XI
        # cobra.df.XIsumViol=df_XIsumViol

    # TODO: write extra columns related to debugging of XI
    # if (cobra$DEBUG_XI) {
    #     firstSolu < - solu;
    #     if ( is.matrix(solu)) firstSolu < - solu[1, ];
    #     optimum < - cobra$fn(firstSolu)[1];
    #
    #     df$fxStart=df_fxStart  # objective function at xStart
    #     df$fxbest=df_fxbest  # objective function at xbest
    #     df$exbest=df_fxbest - optimum  # error (objective function - optimum) at xbest
    #     df$RS2=df_RS2  # the same
    #     df$iter2=1:nrow(df)
    #     df$errFy = df$y - optimum  # the error of the optimizer result in every iteration
    #     # if(tail(df_RS,1)==TRUE) browser()
    #     # browser()
    #     testit::assert (df$RS == df_RS2)
    #     if (any(df$fxbest[!df$RS] != df$fxStart[!df$RS])) {
    #         browser()
    #         df$fxbest[!df$RS]=df$fxStart[!df$RS]  # symptomatic fix for the next assert
    #     }
    #     testit::assert (df$fxbest[!df$RS] == df$fxStart[!df$RS])
    # }

    cobra.df['seed'] = s_opts.cobraSeed

    # result data frame df2:
    last = cobra.df.shape[0] - 1
    new_row_df2 = pd.DataFrame(
        {'iter': cobra.df.iter[last],
         'predY': predY[-1],           # surrogate fitness at current point xNew
         'predVal': predVal[-1],       # surrogate fitness + penalty at xNew
         'predSolu': predSolu,             # surrogate fitness at solu (only diagnostics).
         'predSoluPenal': predSoluPenal,   # surrogate fitness + penalty at solu (only diagnostics).
         'sigmaD': s_opts.sigmaD[1],    # the 1st of the three elements is the currently active sigmaD
         'penaF': s_opts.penaF[1],      # the 1st of the three elements is the currently active penaF
         'XI': p2.gama,
         'fBest': cobra.df.Best[last],
         'EPS': EPS,
         'muVec': p2.currentEps,        # this is df2$currentEps in R
         'PLOG': p2.PLOG[-1],
         'pshift': p2.pshift[-1],
         'pEffect': p2.pEffect,
         'err1': p2.err1[-1],
         'err2': p2.err2[-1],
         'nv_cB': p2.ev1.nv_conB,  # diagnostics for refine mechanism
         'nv_cA': p2.ev1.nv_conA,
         'nv_tB': p2.ev1.nv_trueB,  # diagnostics for refine mechanism
         'nv_tA': p2.ev1.nv_trueA,
         'state': p2.ev1.state
    }, index=[0])
    cobra.df2 = pd.concat([cobra.df2, new_row_df2], axis=0)

    # TODO (later, when TR is ready):
    # cobra$dftr<-rbind(cobra$dftr,data.frame(
    #     TRiter=cobra$TRiter,
    #     TRapprox=cobra$TRapprox,
    #     TFR=cobra$TFR,
    #     TSR=cobra$TSR,
    #     Fratio=cobra$Fratio,
    #     TRpop=cobra$TRpop,
    #     TRdelta=cobra$TRdelta
    # ))

    # consistency check for data frames df and df2:
    msg = "[updateSaveCobra] wrong nrow for df and df2"
    if s_opts.phase1DesignPoints is None:
        assert cobra.df.shape[0] == cobra.df2.shape[0] + s_opts.ID.initDesPoints, msg
    else:
        assert cobra.df.shape[0] == cobra.df2.shape[0] + s_opts.phase1DesignPoints, msg

    # TODO: saveIntermediate
    # if (cobra$saveIntermediate) {
    #     # save intermediate results
    #     # cobraResult = list(cobra=cobra, df=df, constraintSurrogates=cobra$constraintSurrogates, fn=fn)
    #     cobraResult = cobra
    #     if (is.na(file.info("results")$isdir)) dir.create("results")    # if directory "results" does not exist, create it
    #     save(cobraResult, file=sprintf("results/cobra-%s-%s-%i.RData",cobra$fName,cobra$seqOptimizer,cobra$cobraSeed))
    # }
