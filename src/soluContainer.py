import numpy as np
from typing import Tuple
from cobraInit import CobraInitializer
from evaluatorReal import getPredY0
from innerFuncs import distLine
from phase2Vars import Phase2Vars


class SoluContainer():
    """
    This class contains the true solution (if it is provided, it is optional and may be also None) ``solu`` in
    rescaled space and ``originalSolu`` in original space together with methods to act on them.

    ``solu`` may be a 1-dim vector (if there is only one solution) or a 2-dim matrix (if there are multiple equivalent
    solutions, 1 solution vector per matrix row).

    Used by ``updateSaveCobra``.
    """
    def __init__(self, solu, cobra: CobraInitializer):
        self.originalSolu = solu

        if cobra.sac_opts.ID.rescale:
            if solu is not None:
                if solu.ndim == 1:
                    solu = cobra.rw.forward(solu)
                elif solu.ndim == 2:
                    solu = np.apply_along_axis(cobra.rw.forward, axis=1, arr=solu)
                else:
                    raise ValueError(f"solu.ndim = {solu.ndim} is not allowed!")

        self.solu = solu

    def get_solu(self):
        return self.solu

    def get_original_solu(self):
        return self.originalSolu

    def predict_at_solu(self, p2: Phase2Vars, fitnessSurrogate, fitFuncPenalRBF) -> Tuple[float, float]:
        """
        Predict the value of the objective surrogate model at the true solution point (only for diagnostics).
        Return ``np.nan`` if the solution is None.
        Return the minimum, if multiple equivalent solutions exist (``solu.ndim==2``)

        :param p2: needed for prediction
        :param fitnessSurrogate: the objective function surrogate model
        :param fitFuncPenalRBF:
        :return: tuple (predSolu, predSoluPenal) = (predicted objective function value by model fitnessSurrogate,
                                                    predicted value by fitFuncPenalRBF)
        """
        if self.solu is None:
            predSolu = np.nan
            predSoluPenal = np.nan

        else:
            predSoluFunc = lambda x: getPredY0(x, fitnessSurrogate, p2)
            if self.solu.ndim == 1:
                predSolu = predSoluFunc(self.solu)
                predSoluPenal = fitFuncPenalRBF(self.solu)
            elif self.solu.ndim == 2:  # in case of multiple global optima in solu:
                predSolu = np.apply_along_axis(predSoluFunc, axis=1, arr=self.solu)
                predSoluPenal = np.apply_along_axis(fitFuncPenalRBF, axis=1, arr=self.solu)
            else:
                raise ValueError(f"solu.ndim = {self.solu.ndim} is not allowed!")

            predSolu = min(predSolu)                # Why min? - In case of multiple global optima: predSolu is the
            predSoluPenal = min(predSoluPenal)      # value of predSoluFunc at the best solution solu
                                                    # (same for predSoluPenal)
        return predSolu, predSoluPenal

    def distance_to_solu(self, cobra: CobraInitializer) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the distance of all infill points to the true solution. Each row of matrix ``A = cobra.sac_res['A']``
        contains an infill point.

        Return a vector of ``A.shape[0]`` ``np.nan``'s if ``solu`` is None.

        Return the minimum over solutions, if multiple equivalent solutions exist (``solu.ndim==2``)

        :param cobra: needed for infill points ``A`` and rescale wrapper
        :return: tuple (distA, distOrig) = (vector of distances in rescaled space, ... of distances in original space)
        """
        A = cobra.sac_res['A']
        if self.solu is None:
            distA = np.repeat(np.nan, A.shape[0])
            distOrig = np.repeat(np.nan, A.shape[0])
        else:
            origA = np.apply_along_axis(lambda x: cobra.rw.inverse(x), axis=1, arr=A)
            if self.solu.ndim == 1:
                distA = distLine(self.solu, A)
                distOrig = distLine(self.originalSolu, origA)
            elif self.solu.ndim == 2:
                da = np.apply_along_axis(lambda x: distLine(x, A), axis=1, arr=self.solu)
                do = np.apply_along_axis(lambda x: distLine(x, origA), axis=1, arr=self.originalSolu)
                # da and do have solu.shape[0] rows and A.shape[0] columns. Select in each column the minimum element:
                distA = np.min(da, axis=0)
                distOrig = np.min(do, axis=0)
            else:
                raise ValueError(f"solu.ndim = {self.solu.ndim} is not allowed!")

        return distA, distOrig
