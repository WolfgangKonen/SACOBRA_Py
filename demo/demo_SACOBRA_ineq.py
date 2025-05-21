import numpy as np
from cobraInit import CobraInitializer
from gCOP import GCOP
from cobraPhaseII import CobraPhaseII
from opt.sacOptions import SACoptions
from opt.idOptions import IDoptions
from opt.rbfOptions import RBFoptions
from opt.seqOptions import SEQoptions
from show_error_plot import show_error_plot

G06 = GCOP("G06")


cobra = CobraInitializer(G06.x0, G06.fn, G06.name, G06.lower, G06.upper, G06.is_equ,
                         solu=G06.solu,
                         s_opts=SACoptions(verbose=1, feval=40, cobraSeed=42,
                                           ID=IDoptions(initDesign="LHS", initDesPoints=6),
                                           RBF=RBFoptions(degree=2),
                                           SEQ=SEQoptions(conTol=1e-7)))

c2 = CobraPhaseII(cobra).start()

show_error_plot(cobra, G06, file="../error_plot.png")

fin_err = np.array(cobra.get_fbest() - G06.fbest)
print(f"final error: {fin_err}")
