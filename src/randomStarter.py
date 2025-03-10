import numpy as np

from innerFuncs import verboseprint
from opt.sacOptions import SACoptions
from cobraInit import CobraInitializer
from phase2Vars import Phase2Vars


class RandomStarter:
    """
        Decide which start point xStart to take for the next sequential optimizer step

        Either cobra.sac_res['xbest'] or a random point in search space
    """
    def __init__(self, s_opts: SACoptions):
        """
            Initialize RandomStarter RNGs with seed ``s_opts.cobraSeed``
        """
        self.rng = np.random.default_rng(s_opts.cobraSeed)
        self.val = s_opts.cobraSeed         # for other RNG self.my_rng2

    def random_start(self, cobra: CobraInitializer, p2: Phase2Vars) -> np.ndarray:
        if cobra.sac_opts.ISA.RS:
            xStart = self.decide_about_random_start(cobra, p2)
        else:
            xStart = cobra.sac_res['xbest']     # bug fix

        cobra.sac_res['xStart'] = xStart
        return xStart

    def decide_about_random_start(self, cobra: CobraInitializer, p2: Phase2Vars) -> np.ndarray:
        s_opts = cobra.sac_opts

        anewrand = self.my_rng2(1,1)[0,0] if s_opts.ISA.RS_rep else self.rng.random()

        numViol = cobra.sac_res['numViol']
        feasibleRate = np.flatnonzero(numViol == 0).size / numViol.size  # fraction of feasible points in the population
        diff = s_opts.ISA.RSmax - s_opts.ISA.RSmin

        if s_opts.ISA.RSauto and feasibleRate < 0.05:
            p_const = 0.4
        else:
            p_const = (s_opts.ISA.RSmax+s_opts.ISA.RSmin)/2  # default: 0.175

        if s_opts.ISA.RStype == "SIGMOID":
            x = - (cobra.sac_res['A'].shape[0] - (s_opts.ID.initDesPoints+15))
            p_restart = (diff / 2) * np.tanh(x)+(diff / 2) + s_opts.ISA.RSmin
            # Explanation of the SIGMOID case:
            #   - if the argument of tanh is negative and big, then p_restart -> RSmin
            #   - if the argument of tanh is positive and big, then p_restart -> RSmax
            # The argument of tanh is positive in the early iterations and negative in the later iterations
        elif s_opts.ISA.RStype == "CONSTANT":
            p_restart = p_const
        else:
            raise RuntimeError(f"[randomStart] s_opts.ISA.RStype = {s_opts.ISA.RStype} is not valid")

        # TODO: decide whether to use/increment p2.noProgressCount
        if anewrand < p_restart or p2.noProgressCount >= s_opts.ISA.RS_Cs:
            # cat(sprintf("RS: iter=%03d, noProgressCount=%03d\n",nrow(cobra$A)+1,cobra$noProgressCount))
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
            verboseprint(s_opts.verbose, True,
                         f"[randomStart] random xStart = {xStart} at iteration {p2.num}")
            p2.RS_done = True
            # cobra$noProgressCount<-0
        else:
            xStart = cobra.sac_res['xbest']
            xStart = xStart.reshape(cobra.sac_res['dimension'])   # if xbest has shape (1,dim) --> reshape to (dim,)
            p2.RS_done = False
        #print(p2.num, anewrand, p_restart, p2.RS_done)

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


