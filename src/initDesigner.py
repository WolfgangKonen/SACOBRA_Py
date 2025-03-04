import numpy as np
from opt.sacOptions import SACoptions

class InitDesigner:
    def __init__(self, xStart, fn, lower, upper, s_opts: SACoptions):
        """
        Create matrix ``self.A`` with shape ``(P, d)`` of sample points in input space ``[lower, upper]``
        :math:` \in \mathbb{R}^d`. ``P = initDesPoints``.

        Apply ``fn`` to these points and divide the result in objective
        function values ``self.Fres`` (shape ``(P,)``) and constraint function values ``self.Gres`` (shape
        ``(P,nC)`` with ``nC`` = number of constraints).

        :param xStart:  the last point ``self.A[-1,:]`` is ``xStart``
        :param fn:
        :param lower:
        :param upper:
        :param s_opts: object of class :class:`SACoptions`. Here we use from element ID the
                       elements ``initDesign`` and ``initDesPoints``.
        """
        d = xStart.size
        if s_opts.ID.initDesign == "RANDOM":
            # Create self.A with shape (npts,d) where the first npts-1 points in R^d are uniform random from
            # [lower, upper] and the last point is xStart.
            npts = s_opts.ID.initDesPoints
            self.A = np.random.rand(npts-1,d)      # uniform random in [0,1)
            self.A = self.A @ np.diag(upper-lower) + np.tile(lower, (npts-1,1))
            self.A = np.vstack((self.A, xStart))
        elif s_opts.ID.initDesign == "RAND_R":
            # same as "RANDOM", but with reproducible random numbers (reproducible also on the R side)
            npts, seed = s_opts.ID.initDesPoints, 42
            self.A = self.my_rng(npts - 1, d, seed)  # uniform random in [0,1)
            self.A = self.A @ np.diag(upper-lower) + np.tile(lower, (npts - 1, 1))
            self.A = np.vstack((self.A, xStart))
        else:
            raise RuntimeError(f"[InitDesigner] Invalid value s_opts.initDesign = '{s_opts.ID.initDesign}' ")

        # TODO: other initial designs (LHS, ...)

        # Apply fn to all points (rows) in matrix self.A. A point is a row in matrix self.A (axis=1).
        fnEval = np.apply_along_axis(fn, axis=1, arr=self.A)    # fnEval.shape = (initDesPoints, nConstraints+1)
        # nConstraints = fnEval.shape[0] - 1
        self.Fres = fnEval[:, 0]
        self.Gres = fnEval[:, 1:]

    def __call__(self, *args, **kwargs)  -> tuple[np.ndarray,np.ndarray,np.ndarray]:
        return self.A, self.Fres, self.Gres

    def my_rng(self, n, d, seed):
        MOD = 10 ** 5 + 7
        val = seed
        x = np.zeros((n, d), dtype=np.float32)
        for n_ in range(n):
            for d_ in range(d):
                val = (val * val) % MOD
                x[n_, d_] = val / MOD   # map val to range [0,1[
        return x


