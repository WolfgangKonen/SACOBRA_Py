import numpy as np
from cobraInit import CobraInitializer
from cobraPhaseII import CobraPhaseII
from gCOP import COP, show_error_plot


class MyCOP(COP):
    def __init__(self):
        super().__init__()
        self.name = 'first example'
        self.fn = lambda x: np.array([3 * np.sum(x ** 2), 1 - np.sum(x)])
        self.fbest = 1.5
        self.lower = np.array([-5,-5])
        self.upper = np.array([+5,+5])
        self.is_equ = np.array([False])
cop = MyCOP()

cobra = CobraInitializer(None, cop.fn, cop.name, cop.lower, cop.upper, cop.is_equ)

c2 = CobraPhaseII(cobra).start()

print(f"xbest: {cobra.get_xbest()}")
print(f"fbest: {cobra.get_fbest()}")
print(f"final error: {cobra.get_fbest() - cop.fbest}")

show_error_plot(cobra, cop)
