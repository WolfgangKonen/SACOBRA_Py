import numpy as np

from innerFuncs import verboseprint
from opt.sacOptions import SACoptions
from cobraInit import CobraInitializer
from phase2Vars import Phase2Vars


class RandomStarter:
    """
        Decide which start point ``xStart`` to take for the next sequential optimizer step

        Either cobra.sac_res['xbest'] or a random point in search space
    """
    def __init__(self, s_opts: SACoptions):
        """
            Initialize RandomStarter RNGs with seed ``s_opts.cobraSeed``.

            Case ``ISA.RS_rep==False`` is the normal case: Use ``np.random.default_rng`` as RNG.

            Case ``ISA.RS_rep==True`` is only for comparing with SACOBRA R: We use ``self.my_rng2`` as RNG
            which produces with a given seed the same random numbers as ``my_rng2`` in R.
        """
        self.rng = np.random.default_rng(s_opts.cobraSeed)
        self.val = s_opts.cobraSeed         # for other RNG self.my_rng2

    def random_start(self, cobra: CobraInitializer, p2: Phase2Vars) -> np.ndarray:
        """
        This method decides whether ``xStart`` is ``cobra.sac_res['xbest']`` or a random start point. In the latter
        case we set ``p2.rs_done == True``.

        If ``ISA.RS==False`` the probability for random start is 0.

        If ``ISA.RS==True`` the probability depends on the feasibility rate, ``ISA.RStype``, ``ISA.RSauto``,
        ``ISA.RSmin``, ``ISA.RSmax``.

        :param cobra: we need elements ``sac_opts``, ``sac_res['xbest', 'dimension', 'numViol']``
        :param p2: we write on ``p2.rs_done``
        :return: xStart
        :rtype: np.ndarray
        """
        if cobra.sac_opts.ISA.RS:
            xStart = self.decide_about_random_start(cobra, p2)
        else:
            xStart = cobra.sac_res['xbest']     # bug fix
            xStart = xStart.reshape(cobra.sac_res['dimension'])   # if xbest has shape (1,dim) --> reshape to (dim,)
            p2.rs_done = False

        cobra.sac_res['xStart'] = xStart
        return xStart

    def decide_about_random_start(self, cobra: CobraInitializer, p2: Phase2Vars) -> np.ndarray:
        s_opts = cobra.sac_opts

        if s_opts.ISA.RS_rep:
            anewrand = self.my_rng2(1,1)[0,0]
        else:
            anewrand = self.rng.random()

        numViol = cobra.sac_res['numViol']
        feasibleRate = np.flatnonzero(numViol == 0).size / numViol.size  # fraction of feasible points in the population
        diff = s_opts.ISA.RSmax - s_opts.ISA.RSmin

        if s_opts.ISA.RSauto and feasibleRate < 0.05:
            p_const = 0.4
        else:
            p_const = (s_opts.ISA.RSmax+s_opts.ISA.RSmin)/2  # default: 0.175

        if s_opts.ISA.RStype == "SIGMOID":
            x = - (cobra.sac_res['A'].shape[0] - (s_opts.ID.initDesPoints + 15))
            p_restart = (diff / 2) * np.tanh(x) + (diff / 2) + s_opts.ISA.RSmin
            # Explanation of the SIGMOID case:
            #   - if the argument of tanh is negative and big, then p_restart approx. RSmin
            #   - if the argument of tanh is positive and big, then p_restart approx. RSmax
            # The argument of tanh is positive in the early iterations (<= 15 iterations after cobraInit)
            # and negative in the later iterations (> 15 iterations after cobraInit)
        elif s_opts.ISA.RStype == "CONSTANT":
            p_restart = p_const
        else:
            raise RuntimeError(f"[randomStart] s_opts.ISA.RStype = {s_opts.ISA.RStype} is not valid")

        # TODO: decide whether to use/increment p2.noProgressCount (currently we don't)
        if anewrand < p_restart or p2.noProgressCount >= s_opts.ISA.RS_Cs:
            d = cobra.sac_res['dimension']
            lower = cobra.sac_res['lower']
            upper = cobra.sac_res['upper']
            if s_opts.ISA.RS_rep:
                xStart = self.my_rng2(1, d)[0]  # uniform random in [0,1)
            else:
                xStart = self.rng.random(d)  # uniform random in [0,1)
            xStart = xStart @ np.diag(upper - lower) + lower
            # this is just for debug logging:
            cobra.sac_res['rs'] = np.concatenate((cobra.sac_res['rs'], p2.num, xStart), axis=None)
            verboseprint(s_opts.verbose, s_opts.ISA.RS_verb,
                         f"[random_start] random xStart = {xStart} at iteration {p2.num}"
                         f" (anewrand = {anewrand:.4f}, p_restart = {p_restart:.4f})")
            p2.rs_done = True
            # cobra$noProgressCount = 0
        else:
            xStart = cobra.sac_res['xbest']
            xStart = xStart.reshape(cobra.sac_res['dimension'])   # if xbest has shape (1,dim) --> reshape to (dim,)
            verboseprint(s_opts.verbose, s_opts.ISA.RS_verb,
                         f"[random_start] NO random start at iteration {p2.num}"
                         f" (anewrand = {anewrand:.4f}, p_restart = {p_restart:.4f})")
            p2.rs_done = False

        return xStart

    def my_rng2(self, n: int, d: int) -> np.ndarray:
        """ A reproducible RNG, only called from random_start, which has same results on the R side

        :param n:
        :param d:
        :return: an (n,d)-array with random numbers uniformly distributed in [0,1[
        """
        MOD = 10 ** 5 + 7
        OFS = 10 ** 5 - 7
        x = np.zeros((n, d), dtype=np.float32)
        for n_ in range(n):
            for d_ in range(d):
                self.val = (self.val*self.val*np.sqrt(self.val)+OFS) % MOD    # avoid cycles (!)
                x[n_, d_] = self.val / MOD   # map val to range [0,1[
        return x


