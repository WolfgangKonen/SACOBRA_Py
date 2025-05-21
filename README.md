# SACOBRA_Py
SACOBRA Python port.

SACOBRA is a package for constrained optimization with relatively few function evaluations.

SACOBRA stands for **Self-Adjusting Constraint Optimiization By Radial basis function Optimization**. It is used for numerical optimization and can handle an arbitrary number of inequality and/or equality constraints.

SACOBRA was originally developed in R. This repository **SACOBRA_Py** contains the beta version of a Python port, which is simplified in code and up to 4 times faster than the R version. (The R-version of SACOBRA is available from [this GitHub repository](https://github.com/WolfgangKonen/SACOBRA).)

## How to install

You just need to install the few packages listed in [requirements.txt](./requirements.txt) (if they are not already present in your Python environment). Then, clone the contents of this repository and proceed with the examples.

## How to use
Below is an example from [demo_SACOBRA_ineq.py](./demo/demo_SACOBRA_ineq.py): 

```Python
G06 = GCOP("G06")

cobra = CobraInitializer(G06.x0, G06.fn, G06.name, G06.lower, G06.upper, G06.is_equ, solu=G06.solu,
                         s_opts=SACoptions(verbose=1, feval=40, cobraSeed=42,
                                           ID=IDoptions(initDesign="LHS", initDesPoints=6),
                                           RBF=RBFoptions(degree=2),
                                           SEQ=SEQoptions(conTol=1e-7)))
c2 = CobraPhaseII(cobra).start()

show_error_plot(cobra, G06)

fin_err = np.array(cobra.get_fbest() - G06.fbest)
print(f"final error: {fin_err}")
```

First we construct the constraint optimization problem (COP) ``G06`` from the G-problem benchmark suite.  SACOBRA contains several such benchmark problems in [gCOP.py](./src/gCOP.py). Next we construct with ``CobraInitializer`` the object ``cobra`` with all optimization settings. The optimization is then started with ``CobraPhaseII`` where 40 iterations on surrogate models are carried out. (There is also a ``CobraPhaseI``, but it is optional and can be left out.) We show with ``show_error_plot`` the error on a logarithmic scale, i.e. the distance between the objective found by the solver in each iteration and the true objective ``G06.fbest``.

<img src="error_plot.png" alt="Error Plot G06" title="Error curve obtained by SACOBRA" width=600 />

## Publications
You can read more about SACOBRA in the following scientific publications:

< ... >

