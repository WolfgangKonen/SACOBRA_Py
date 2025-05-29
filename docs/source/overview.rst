--------
Overview
--------

What is **SACOBRA_Py** and what is contained in this documentation?


SACOBRA_Py
-----------------

`**SACOBRA_Py** <https://github.com/WolfgangKonen/SACOBRA_Py>`_ is the SACOBRA Python port.

SACOBRA is a package for constrained optimization with relatively few function evaluations.

SACOBRA stands for **Self-Adjusting Constraint Optimization By Radial basis function Approximation**. It is used for numerical optimization and can handle an arbitrary number of inequality and/or equality constraints.

SACOBRA was originally developed in R. This repository **SACOBRA_Py** contains the beta version of a Python port, which is simplified in code and up to 4 times faster than the R version. (The R-version of SACOBRA is available from `this GitHub repository <https://github.com/WolfgangKonen/SACOBRA>`_.)

The **SACOBRA_Py** realization relies on these other Python packages:

- ``lhsmdu`` for the latin hypercube sampling (LHS) initial design
- ``scipy.RBFInterpolator`` for the RBF surrogate models
- ``nlopt`` for the nonlinear optimization in the sequential optimization step


Documentation
-----------------

This documentation contains:

    - a brief introduction to COPs (constrained optimization problems) and the G-problem benchmark
    - how initialization of SACOBRA_Py works
    - how optimization with SACOBRA_Py works
    - usage examples
    - an appendix with further details (data frames ``cobra.df``, ``cobra.df2``)