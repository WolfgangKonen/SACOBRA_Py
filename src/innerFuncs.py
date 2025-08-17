from typing import Union

import numpy as np

def verboseprint(verbose:int, important:bool, message:str):
    """
    :param verbose:     0: print nothing. 1: print only important messages. 2: print everything
    :param important:   True for important messages
    :param message:     the message string
    """
    if verbose != 0:
        if verbose == 2 or (verbose == 1 and important == True):
            print(message)


def distLine(x,xp):
    """
    Euclidean distance of ``x`` to a line of points ``xp``.

    :param x:   vector of dimension d
    :param xp:  n points x_i of dimension d are arranged in (n x d) matrix xp.
                If xp is a vector, it is interpreted as (n x 1) matrix, i.e. d=1.
    :return:    vector of length n, the Euclidean distances
    """
    xp = xp.reshape(xp.shape[0], x.shape[0])
    z = np.tile(x,(xp.shape[0], 1)) - xp
    z = np.sqrt(np.sum(z*z, axis=1))
    return z


def plog(f, pShift=0.0):
    """
    Monotonic transform. This  function is introduced in [Regis 2014] and extended here by a parameter
    ``pShift``. It is used to squash a function with a large range into a smaller range.

    Let :math:`f' =  f - p_{shift}`. Then:

    :math:`plog(f) = +\\ln(1 + f'),  \\quad\\mbox{if}\\quad f' \\ge 0`
    and

    :math:`plog(f) = -\\ln(1 - f'), \\quad\\mbox{if}\\quad f'  <   0`


    :param f:   function value(s), number or np.array
    :param pShift:  optional shift
    :return:    np.sign(f') * ln(1 + |f'|)
    """
    return np.sign(f - pShift) * np.log(1 + np.abs(f - pShift))


def plogReverse(y: Union[float, np.ndarray], pShift=0.0, verbose=False) -> Union[float, np.ndarray]:
    """
    Inverse of ``plog(f, pShift)``.

    Too large values of ``np.abs(y)`` are clipped to avoid ``RuntimeError: overflow in exp``.

    :param y:       function argument, number or array
    :param pShift:  optional shift
    :param verbose: if True, print a warning message if any clipping occurs
    :return:    np.sign(y) * (np.exp(|y|) - 1) + pShift
    """
    # if True:    # isinstance(y, float):
    #                                     # bug fix 2025/06/24:
    #     if np.abs(y) > 709:             # avoid 'RuntimeWarning: overflow encountered in exp' (G02 with dim=10)
    #         # print(f"[plogReverse] Warning: clipping too large y={y} to np.sign(y)*(np.exp(700) - 1)")
    #         return np.sign(y) * (np.exp(700) - 1) + pShift
    #         # return np.sign(y) * np.inf + pShift
    # return np.sign(y) * (np.exp(np.abs(y)) - 1) + pShift
    # # /WK/2025/03/06: bug fix for negative y

    # /WK/2025/08/13:  better version to avoid 'RuntimeWarning: overflow': The problem of the above (commented-out)
    # approach is that 'if np.abs(y) > 709' does not work for arrays y with size > 1. Now we clip only those entries
    # of np.abs(y) that are > 705 to 705, and this works for arrays of any size as well:
    abs_y_clip = np.minimum(np.abs(y), 705)
    if verbose:
        if any(abs_y_clip != np.abs(y)):
            print(f"[plogReverse] Warning: clipping too large |y|={np.abs(y)} to np.sign(y)*(np.exp(705) - 1)")
            dummy = 0
    return np.sign(y) * (np.exp(abs_y_clip) - 1) + pShift

# # -----------------------------------------------------------------------------------------------
# # ----------------  helper functions     subprob*, gCOBRA*  -------------------------------------
# # -----------------------------------------------------------------------------------------------
#
# calcConstrPred < - function(x, cobra)   is now in seqOptimizer.py
#
# subProb2 and gCOBRA are as member functions of SeqFuncFactory in seqOptimizer.py
#
# subProbPhase1 is in seqOptimizer.py
# subProb, gCOBRA_cobyla, isresCobyla are in seqOptimizer.py (currently commented out)
# subProb2Phase1, gCOBRAPhase1, isresCobyla are in seqOptimizer.py (currently commented out)
#
# subProbConstraintHandling < - function(x, cobra, penalty1, penalty2, maxViolation, sigmaD, penaF)
# is now also in seqOptimizer.py (currently commented out)
#
