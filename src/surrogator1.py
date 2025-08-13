import time
import numpy as np
from typing import Union
# need to specify SACOBRA_Py.src as source folder in File - Settings - Project Structure,
# then the following import statements will work:
from cobraInit import CobraInitializer
from cobraPhaseII import Phase2Vars
from innerFuncs import verboseprint, plog, plogReverse
from opt.isaOptions import O_LOGIC
from rbfModel import RBFmodel


# Note: this class with three static elements is here just to bundle these elements for the documentation
class Surrogator1:
    """
        This class calculates the surrogate models in case ``ISA.onlinePLOG =`` **O_LOGIC.XNEW**.
    """

    @staticmethod
    class AdFitter:
        """
            Adjust fitness values of ``Fres``, depending on :ref:`p-effect <pEffect-label>` (if ``onlinePLOG != O_LOGIC.NONE``)
            or depending on ``FRange``, ``ISA.TFRange`` (if ``onlinePLOG == O_LOGIC.NONE``)
        """
        def __init__(self, cobra: CobraInitializer, p2: Phase2Vars, Fres):
            """
            The values provided in ``Fres`` are conditionally plog-transformed and the results
            are available from ``self.surrogateInput``.

            :param cobra:   AdFitter needs ``sac_opts`` and ``sac_res``
            :param p2:      AdFitter needs ``p2.pEffect`` and changes ``p2.PLOG``, ``p2.pshift``
            :param Fres:    the ``Fres`` input, may be either cobra.sac_res['Fres'] or cobra.for_rbf['Fres']
            """
            s_opts = cobra.get_sac_opts()
            s_res = cobra.get_sac_res()
            idp = s_opts.ID.initDesPoints
            # self.FRange = (max(Fres) - min(Fres))                 # 2025/08/07: bug fix
            self.FRange = (max(Fres[0:idp]) - min(Fres[0:idp]))     # make FRange really constant (over iterations)
            self.pshift = 0
            self.PLOG = False

            if s_opts.ISA.aFF:
                # print("adjusting fitness function")
                if s_opts.ISA.onlinePLOG is not O_LOGIC.NONE:
                    assert p2.pEffect is not None
                    # if p2.pEffect > 1:  # /WK/2025/08/04: bug fix, pEffect=log10(errRatio) has to be compared with 0:
                    if p2.pEffect > 0:
                        self.PLOG = True
                        Fres = plog(Fres, pShift=self.pshift)
                    # else: leave Fres at its input value

                else:  # i.e. if ISA.onlinePLOG == O_LOGIC.NONE
                    if self.FRange > s_opts.ISA.TFRange:
                        if s_opts.ISA.adaptivePLOG:
                            self.pshift = np.mean((s_res['fbest'], 0))
                        # else: leave self.pshift=0

                        self.PLOG = True
                        Fres = plog(Fres, pShift=self.pshift)
                    # else: leave Fres at its input value

            p2.PLOG = np.concat((p2.PLOG, [self.PLOG]))
            p2.pshift = np.concat((p2.pshift, [self.pshift]))
            self.surrogateInput = Fres
            self.FRange_after = (max(Fres) - min(Fres))
            verboseprint(s_opts.verbose, False,
                         f"[adFit] FRange = {self.FRange}, PLOG={self.PLOG}, new FRange = {self.FRange_after}")

        def __call__(self):
            """
            :return: ``self.surrogateInput``, a potentially plog-transformed ``Fres``
            :rtype: np.ndarray
            """
            return self.surrogateInput

        def get_PLOG(self):
            return self.PLOG

        def get_pshift(self):
            return self.pshift

        def get_FRange_before(self):
            return self.FRange

        def get_FRange_after(self):
            return self.FRange_after

    @staticmethod
    def calcPEffect(p2: Phase2Vars, xNew: np.ndarray, xNewEval: np.ndarray):
        """
            Calculates the :ref:`p-effect <pEffect-label>` in variable ``p2.pEffect`` with method described in
            :ref:`Details for onlinePLOG <detail_onlinePLOG-label>`, case **O_LOGIC.XNEW**.

            Let ``opl = s_opts.ISA.onlinePLOG``.

            In case ``opl != NONE``, class :class:`.AdFitter` will
            apply plog to ``Fres`` if ``p2.pEffect`` > 0, else ``Fres`` is used directly.

            In case ``opl == NONE``, ``p2.pEffect`` is irrelevant.

            :param p2:      needs  ``p2.fitnessSurrogate1`` and  ``p2.fitnessSurrogate2`` on input and
                            changes ``p2.err1``, ``p2.err2``, ``p2.errRatio`` and ``p2.pEffect`` on output
            :param xNew:    the new infill point
            :param xNewEval: ``fn(xNew)[0]``
            :return: None
        """
        assert p2.fitnessSurrogate1.__class__.__name__ == 'RBFmodel', \
               "[calcPEffect] p2.fitnessSurrogate1 is not RBFmodel"
        assert p2.fitnessSurrogate2.__class__.__name__ == 'RBFmodel', \
               "[calcPEffect] p2.fitnessSurrogate2 is not RBFmodel"
        newPredY1 = p2.fitnessSurrogate1(xNew)
        newPredY2 = p2.fitnessSurrogate2(xNew)
        newErr1 = abs(newPredY1 - xNewEval[0])
        newErr2 = abs(plogReverse(newPredY2) - xNewEval[0])
        # newErr2 = abs(newPredY2-xNewEval[0])
        p2.err1 = np.concat((p2.err1, newErr1))
        p2.err2 = np.concat((p2.err2, newErr2))
        nu = 1e-20   # regularizing constant to avoid 0/0-situation in p2.errRatio
        p2.errRatio = (p2.err1 + nu) / (p2.err2 + nu)

        # due to regularization with nu, the following three assertions should never fire. They are in here only
        # as sanity check:
        assert not np.isinf(newErr2), f"[calcPEffect] newErr2={newErr2} is infinity"
        assert not np.isinf(newErr1), f"[calcPEffect] newErr1={newErr1} is infinity"
        # if np.isinf(newErr2):
        #     print(f"*** Warning ***: [calcPEffect] newErr2={newErr2} is infinity")
        #     p2.errRatio[-1] = 0
        # elif np.isinf(newErr1):
        #     print(f"*** Warning ***: [calcPEffect] newErr1={newErr1} is infinity")
        #     p2.errRatio[-1] = np.inf

        z = np.nanmedian(p2.errRatio)       # nanmedian: compute median while ignoring NaNs
        assert z > 0, f"[calcPEffect] z={z} is <= 0"
        p2.pEffect = np.log10(z)
        # if z <= 0: print(f"*** Warning ***: [calcPEffect] z={z} is <= 0")
        # p2.pEffect = np.log10(z) if z > 0 else 0

    @staticmethod
    def trainSurrogates(cobra: CobraInitializer, p2: Phase2Vars) -> Phase2Vars:
        """
            Train surrogate models  ``p2.fitnessSurrogate``, ``p2.constraintSurrogates``, ``p2.fitnessSurrogate1``, ``p2.fitnessSurrogate2``.

            :param cobra:
            :param p2:
            :return: p2
            :rtype: Phase2Vars
        """
        s_opts = cobra.get_sac_opts()
        s_res = cobra.get_sac_res()
        CONSTRAINED = s_res['nConstraints'] > 0
        verboseprint(s_opts.verbose, False, f"[trainSurrogates] Training {s_opts.RBF.model} surrogates ...")
        start = time.perf_counter()

        # A = s_res['A']
        A = cobra.for_rbf['A']
        # important: use here the A from cobra.for_rbf (!), this avoids the LinAlgError ("Singular Matrix") that
        # would otherwise occur in calls to RBFInterpolator (bug fix 2025/06/03)

        # important: use here the Fres from cobra.for_rbf (!), will be also used for Fres1, Fres2 below.
        p2.adFit = Surrogator1.AdFitter(cobra, p2, cobra.for_rbf['Fres'].copy())
        Fres = p2.adFit()   # the __call__ method returns p2.adfit.surrogateInput, a potentially plog-transformed Fres

        #   if(cobra$TFlag){
        #   Fres<-cobra$TFres
        #   A<-cobra$TA
        #   }

        if s_opts.ISA.isa_ver > 0:
            if p2.PLOG[-1] and p2.printP:
                verboseprint(s_opts.verbose, True, f"PLOG transformation is done ( iter={A.shape[0]} )")
        p2.printP = False

        if CONSTRAINED:
            # Gres=s_res['Gres']
            Gres = cobra.for_rbf['Gres']
            # important: use here the Gres from cobra.for_rbf (!), avoids LinAlgError
            if not s_opts.MS.apply or not s_opts.MS.active:
                p2.fitnessSurrogate = RBFmodel(A, Fres, interpolator=s_opts.RBF.interpolator,
                                               kernel=s_opts.RBF.model,
                                               degree=s_opts.RBF.degree, rho=s_opts.RBF.rho)
                p2.constraintSurrogates = RBFmodel(A, Gres, interpolator=s_opts.RBF.interpolator,
                                                   kernel=s_opts.RBF.model,
                                                   degree=s_opts.RBF.degree, rho=s_opts.RBF.rho)
            else:
                # TODO: the model selection (MS) part
                raise NotImplementedError("[trainSurrogates] MS-part in branch 'if CONSTRAINED' not yet implemented! ")

        else:  # i.e. if not CONSTRAINED

            if not s_opts.MS.apply or not s_opts.MS.active:
                kernel = s_opts.RBF.model
            else:
                # TODO: the model selection (MS) part
                raise NotImplementedError("[trainSurrogates] MS-part in branch 'if not CONSTRAINED' not yet implemented! ")

            p2.fitnessSurrogate = RBFmodel(A, Fres, interpolator=s_opts.RBF.interpolator,
                                           kernel=kernel, degree=s_opts.RBF.degree, rho=s_opts.RBF.rho)

            # if (cobra$DEBUG_RBF$active){
            # print(nrow(A))
            # cobra < - debugVisualizeRBF(cobra, cobra$fitnessSurrogate, A, Fres)  # see defaultDebugRBF.R
            # }

        # build  every onlineFreqPLOG iterations the models to measure p-effect anew:
        recalc_fit12 = (s_opts.ISA.onlinePLOG is not O_LOGIC.NONE and (A.shape[0] % s_opts.ISA.onlineFreqPLOG == 0)) \
                       or (p2.fitnessSurrogate1 is None)
        recalc_fit12 = True
        if recalc_fit12:
            # two models are built after every onlineFreqPLOG iterations:
            Fres1 = cobra.for_rbf['Fres']
            Fres2 = plog(cobra.for_rbf['Fres'])
            p2.fitnessSurrogate1 = RBFmodel(A, Fres1, interpolator=s_opts.RBF.interpolator,
                                            kernel=s_opts.RBF.model,
                                            degree=s_opts.RBF.degree, rho=s_opts.RBF.rho)
            p2.fitnessSurrogate2 = RBFmodel(A, Fres2, interpolator=s_opts.RBF.interpolator,
                                            kernel=s_opts.RBF.model,
                                            degree=s_opts.RBF.degree, rho=s_opts.RBF.rho)

        DO_ASSERT = False
        if DO_ASSERT:
            # test that at the observation points (rows of A), all three models,
            # fn(A)[:,1:], s_res['Gres'], p2.constraintSurrogates(A), have the same values
            #
            # might need adjustment due to rescale /WK/
            fnEval = np.apply_along_axis(s_res['fn'], axis=1, arr=A)    # fnEval.shape = (initDesPoints, nConstraints+1)
            Gres = fnEval[:, 1:]
            assert np.allclose(Gres, s_res['Gres']), "Gres-assertion failed"

            Gres = p2.constraintSurrogates(A)
            # Gres = np.apply_along_axis(s_res['constraintSurrogates'], axis=1, arr=A)
            for i in range(Gres.shape[1]):
                gi = Gres[:, i]
                z = (gi - s_res['Gres'][:, i]) / (np.max(gi) - np.min(gi))
                # TODO (in such a way that assertion does not fire w/o reason):
                # if max(abs(z)) > 5e-6:
                #     dummy = 0
                # assert max(abs(z)) <= 5e-6, f"s_res['constraintSurrogates'](A)-assertion failed for constraint {i}"
            verboseprint(s_opts.verbose, False, "[trainSurrogates] All assertions passed")

        verboseprint(s_opts.verbose, False,
                     f"[trainSurrogates] ... finished ({(time.perf_counter() - start)*1000} msec)")
        return p2
