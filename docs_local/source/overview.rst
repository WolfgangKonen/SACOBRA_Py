--------
Overview
--------

What is **SACOBRA_Py** and what is contained in this documentation?


SACOBRA_Py
-----------------

**SACOBRA_Py**, available from `<https://github.com/WolfgangKonen/SACOBRA_Py>`_, is the SACOBRA Python port.

.. image:: ../../demo/sacobra-logo.png
   :height: 153px
   :width: 576px
   :align: center

SACOBRA is a package for constrained optimization with relatively few function evaluations.

SACOBRA stands for **Self-Adjusting Constraint Optimization By Radial basis function Approximation**. It is used for numerical optimization and can handle an arbitrary number of inequality and/or equality constraints.

SACOBRA was originally developed in R. This repository **SACOBRA_Py** contains the beta version of a Python port, which is simplified in code and faster than the R version by a factor of 4 - 40. (The R-version of SACOBRA is available from `this GitHub repository <https://github.com/WolfgangKonen/SACOBRA>`_.)



Documentation
-----------------

This documentation contains:

    - a brief introduction to COPs (constrained optimization problems) and to the G-problem benchmark
    - how initialization of SACOBRA_Py works
    - how optimization in SACOBRA_Py is done
    - usage examples
    - an appendix with further details (dict ``cobra.sac_res`` and data frames ``cobra.df``, ``cobra.df2``)


Authors and Credits
-------------------

The **SACOBRA_Py** Python port is developed by

- Wolfgang Konen, TH Köln

It is based on the earlier R package SACOBRA which was authored by

- Samineh Bagheri, formerly TH Köln, now inovex GmbH
- Wolfgang Konen, TH Köln
- Thomas Baeck, Univ. Leiden

SACOBRA uses many ideas from and extends COBRA, which was developed by R. G. Regis [Regis14]_.

The **SACOBRA_Py** realization relies on these other Python packages and software tools:

- ``scipy.RBFInterpolator`` for building the RBF surrogate models
- ``scipy.stats.qmc.LatinHypercube`` for latin hypercube sampling (LHS) in the initial design phase
- ``nlopt`` for nonlinear optimization algorithms in the sequential optimization step
- ``Sphinx`` for building the package documentation from inline docstrings and .rst files
- ``readthedocs.io`` for deploying and hosting the documentation pages
.. ``lhsmdu`` for latin hypercube sampling (LHS) in the initial design phase

We acknowledge and are grateful for all the work that goes into these great open source software tools!

.. [Regis14] R. G. Regis. Constrained optimization by radial basis function interpolation for high-dimensional expensive black-box problems with infeasible initial points. Engineering Optimization, 46(2):218-243, 2014.