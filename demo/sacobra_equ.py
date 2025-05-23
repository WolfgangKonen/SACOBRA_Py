import numpy as np
from cobraInit import CobraInitializer
from gCOP import GCOP
from cobraPhaseII import CobraPhaseII
from opt.equOptions import EQUoptions
from opt.sacOptions import SACoptions
from opt.idOptions import IDoptions
from opt.rbfOptions import RBFoptions
from opt.seqOptions import SEQoptions
from show_error_plot import show_error_plot

G13 = GCOP("G13")

cobra = CobraInitializer(G13.x0, G13.fn, G13.name, G13.lower, G13.upper, G13.is_equ, solu=G13.solu,
                         s_opts=SACoptions(verbose=1, feval=300, cobraSeed=42,
                                           ID=IDoptions(initDesign="LHS", initDesPoints=6*7//2),
                                           RBF=RBFoptions(degree=2, rho=2.5, rhoDec=2.0),
                                           EQU=EQUoptions(muGrow=100, dec=1.6, equEpsFinal=1e-7, refineAlgo="COBYLA"),
                                           SEQ=SEQoptions(conTol=1e-7)))

c2 = CobraPhaseII(cobra).start()

show_error_plot(cobra, G13, file="../error_plot_G13.png")

fin_err = np.array(cobra.get_fbest() - G13.fbest)
print(f"final error: {fin_err}")

