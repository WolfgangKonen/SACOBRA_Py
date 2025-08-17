import time
import numpy as np
from typing import Union
# need to specify SACOBRA_Py.src as source folder in File - Settings - Project Structure,
# then the following import statements will work:
from surrogator1 import Surrogator1   # for AdFitter
from cobraInit import CobraInitializer
from cobraPhaseII import Phase2Vars
from innerFuncs import verboseprint, plog, plogReverse
from rbfModel import RBFmodel


# Note: this class with three static elements is here just to bundle these elements for the documentation
class Surrogator2:
    """
        This class calculates the surrogate models in case ``ISA.onlinePLOG =`` **O_LOGIC.MIDPTS**.
    """

    @staticmethod
    def comp_midpoints(A: np.ndarray, pts: Union[int, None]) -> np.ndarray:
        """
            Take the first ``pts`` points from initial design matrix ``A`` and calculate the midpoints of all pairs

            :param A: initial design matrix
            :param pts: will be clipped  if larger than ``ID.initDesPoints``. If ``None`` use this value as well.
            :return: the midpoint matrix ``midp``
        """
        assert A.ndim == 2
        idp = A.shape[0]       # ID.initDesPoints
        if pts is None:
            pts = idp
        pts = min(pts, idp)    # clip pts if it demands more points than ID.initDesPoints
        # a sanity check, should not happen:
        assert pts > 1, f"[comp_midpoints] pts={pts} is not > 1 --> cannot form any pair"
        kmax = pts*(pts-1)//2
        midp = np.zeros((kmax, A.shape[1]))
        k = 0
        for i in range(pts):
            for j in range(i+1,pts):
                assert k < kmax
                midp[k, :] = (A[i, :] + A[j, :])/2.0
                k = k + 1
        return midp

    @staticmethod
    def comp_midp_eval(cobra: CobraInitializer, midp: np.ndarray) -> np.ndarray:
        nrow = midp.shape[0]
        midpEval = np.zeros(nrow)
        for k in range(nrow):
            midpEval[k] = cobra.sac_res['fn'](midp[k, :])[0]
        return midpEval

    @staticmethod
    def calcPEffect(p2: Phase2Vars, midp: np.ndarray, midpEval: np.ndarray, verbose=False):
        """
            Calculates the :ref:`p-effect <pEffect-label>` in variable ``p2.pEffect`` with method described in
            :ref:`Details for onlinePLOG <detail_onlinePLOG-label>`, case **O_LOGIC.MIDPTS**.

            Let ``opl = s_opts.ISA.onlinePLOG``.

            In case ``opl != NONE``, class :class:`.AdFitter` will
            apply plog to ``Fres`` if ``p2.pEffect`` > 0, else ``Fres`` is used directly.

            In case ``opl == NONE``, ``p2.pEffect`` is irrelevant.

            :param p2:      needs  ``p2.fitnessSurrogate1`` and  ``p2.fitnessSurrogate2`` on input and
                            changes ``p2.err1``, ``p2.err2``, ``p2.errRatio`` and ``p2.pEffect`` on output
            :param midp:    the midpoints
            :param midpEval: ``fn(midp)[0]``
            :param verbose: if True, print a warning message if any clipping occurs
            :return: None
        """
        assert p2.fitnessSurrogate1.__class__.__name__ == 'RBFmodel', \
               "[calcPEffectNew] p2.fitnessSurrogate1 is not RBFmodel"
        assert p2.fitnessSurrogate2.__class__.__name__ == 'RBFmodel', \
               "[calcPEffectNew] p2.fitnessSurrogate2 is not RBFmodel"
        nrow = midp.shape[0]
        assert midpEval.size == nrow
        newErr1 = newErr2 = 0
        clip_done = False
        for k in range(nrow):
            x = midp[k, :]
            newPredY1 = p2.fitnessSurrogate1(x)
            newPredY2 = p2.fitnessSurrogate2(x)
            if np.abs(newPredY2) > 705: clip_done = True
            newErr1 += abs(newPredY1 - midpEval[k])
            newErr2 += abs(plogReverse(newPredY2, verbose=verbose) - midpEval[k])
        p2.err1 = np.concat((p2.err1, newErr1/nrow))
        p2.err2 = np.concat((p2.err2, newErr2/nrow))
        nu = 1e-20   # regularizing constant to avoid 0/0-situation in p2.errRatio
        p2.errRatio = (p2.err1 + nu) / (p2.err2 + nu)
        if verbose and clip_done > 705:
            print(f"[calcPEffect] Warning: clipping done --> last p2.errRatio = {p2.errRatio[-1]}")

        # Thanks to the regularization with nu (and thanks to the clipping on large values in plogReverse), the
        # following three assertions should never fire. They are in here only as sanity check:
        assert not np.isinf(newErr1), f"[calcPEffect] newErr1={newErr1} is infinity"
        assert not np.isinf(newErr2), f"[calcPEffect] newErr2={newErr2} is infinity"

        z = np.nanmedian(p2.errRatio)       # nanmedian: compute median while ignoring NaNs
        assert z > 0, f"[calcPEffect] z={z} is <= 0"
        p2.pEffect = np.log10(z)

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
        verboseprint(s_opts.verbose, False, f"[trainSurrogates] Training {s_opts.RBF.kernel} surrogates ...")
        start = time.perf_counter()

        # A = s_res['A']
        A = cobra.for_rbf['A']
        # important: use here the A from cobra.for_rbf (!), this avoids the LinAlgError ("Singular Matrix") that
        # would otherwise occur in calls to RBFInterpolator (bug fix 2025/06/03)

        recalc_fit12 = True
        if recalc_fit12:
            # two models are built in every iteration:
            Fres1 = cobra.for_rbf['Fres']
            Fres2 = plog(cobra.for_rbf['Fres'])
            p2.fitnessSurrogate1 = RBFmodel(A, Fres1, s_opts.RBF)
            p2.fitnessSurrogate2 = RBFmodel(A, Fres2, s_opts.RBF)

        if p2.midpts is None:   # i.e. on first pass through cobraPhaseII while loop:
            # calculate the midpoints for pEffect
            p2.midpts = Surrogator2.comp_midpoints(cobra.for_rbf['A'],
                                                   s_opts.ISA.pEff_npts)
            p2.midptsEval = Surrogator2.comp_midp_eval(cobra, p2.midpts)

        Surrogator2.calcPEffect(p2, p2.midpts, p2.midptsEval, verbose=True)    # calculates p2.pEffect

        p2.adFit = Surrogator1.AdFitter(cobra, p2, cobra.for_rbf['Fres'].copy())     # appends to p2.PLOG
        Fres = p2.adFit()   # the __call__ method returns p2.adfit.surrogateInput, a potentially plog-transformed Fres

        if s_opts.ISA.isa_ver > 0:
            if p2.PLOG[-1] and p2.printP:
                verboseprint(s_opts.verbose, True, f"PLOG transformation is done ( iter={A.shape[0]} )")
        p2.printP = False

        if CONSTRAINED:
            # Gres=s_res['Gres']
            Gres = cobra.for_rbf['Gres']
            # important: use here the Gres from cobra.for_rbf (!), avoids LinAlgError
            if not s_opts.MS.apply or not s_opts.MS.active:
                p2.fitnessSurrogate = p2.fitnessSurrogate2 if p2.PLOG[-1] else p2.fitnessSurrogate1
                p2.constraintSurrogates = RBFmodel(A, Gres, s_opts.RBF)
            else:
                # TODO: the model selection (MS) part
                raise NotImplementedError("[trainSurrogates] MS-part in branch 'if CONSTRAINED' not yet implemented! ")

        else:  # i.e. if not CONSTRAINED

            if not s_opts.MS.apply or not s_opts.MS.active:
                kernel = s_opts.RBF.kernel
            else:
                # TODO: the model selection (MS) part
                raise NotImplementedError("[trainSurrogates] MS-part in branch 'if not CONSTRAINED' not yet implemented! ")

            p2.fitnessSurrogate = p2.fitnessSurrogate2 if p2.PLOG[-1] else p2.fitnessSurrogate1

            # if (cobra$DEBUG_RBF$active){
            # print(nrow(A))
            # cobra < - debugVisualizeRBF(cobra, cobra$fitnessSurrogate, A, Fres)  # see defaultDebugRBF.R
            # }

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
