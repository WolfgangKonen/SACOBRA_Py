--------
Overview
--------

What is **SACOBRA_Py** and what is contained in this documentation?


SACOBRA_Py
----------------

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
    - how to conduct reproducible experiments
    - an appendix with further details (dict ``cobra.sac_res`` and data frames ``cobra.df``, ``cobra.df2``)


Publications
-------------
You can read more about SACOBRA in the following scientific publications:

- [Bagh16a]_
- [Bagh16b]_
- [Bagh17a]_
- [Bagh17b]_
- [Bagh18]_

Authors and Credits
-------------------

The **SACOBRA_Py** Python port is developed by

- Wolfgang Konen, TH Köln

It is based on the earlier R package SACOBRA which was authored by

- Samineh Bagheri, formerly TH Köln, now inovex GmbH
- Wolfgang Konen, TH Köln
- Thomas Baeck, Univ. Leiden

SACOBRA uses many ideas from and extends COBRA [Regis14]_, which was developed by

- Rommel G. Regis, SJU, Philadelphia.

The **SACOBRA_Py** realization relies on these other Python packages and software tools:

- ``scipy.RBFInterpolator`` for building the RBF surrogate models
- ``scipy.stats.qmc.LatinHypercube`` for latin hypercube sampling (LHS) in the initial design phase
- ``nlopt`` for nonlinear optimization algorithms in the sequential optimization step
- ``Sphinx`` for building the package documentation from inline docstrings and .rst files
- ``readthedocs.io`` for deploying and hosting the documentation pages
- ``lhsmdu`` for latin hypercube sampling (LHS) in the initial design phase

We acknowledge and are grateful for all the work that goes into these great open source software tools!


.. [Regis14] Regis, Rommel G. Constrained optimization by radial basis function interpolation for high-dimensional expensive black-box problems with infeasible initial points. Engineering Optimization, 46(2):218-243, 2014.

.. [Bagh16a] Bagheri, S., Konen, W., Bäck, T. **Equality constraint handling for surrogate-assisted constrained optimization.** In K. C. Tan, editor, Proc. World Congress on Computational Intelligence (WCCI), Vancouver, p. 1924-1931. IEEE, 2016. [URL](http://www.gm.fh-koeln.de/~konen/Publikationen/Bagh16-WCCI.pdf)

.. [Bagh16b] Bagheri, S., Konen, W., Bäck, T. **Online Selection of Surrogate Models for Constrained Black-Box Optimization.** In: Jin, Yaochu (Hrsg.): SSCI'2016, Athens, S. 1, IEEE, 2016. (**Best Student Paper Award**) [URL](http://www.gm.fh-koeln.de/~konen/Publikationen/Bagh16-SSCI.pdf)

.. [Bagh17a] Bagheri, S., Konen, W., Emmerich, M., Bäck, T. **Self-adjusting parameter control for surrogate-assisted constrained optimization under limited budgets.** In: Applied Soft Computing, vol. 61, p. 377-393, ISSN: 1568-4946, 2017. [URL](http://www.gm.fh-koeln.de/ciopwebpub/Bagh17b/ASOC-SACOBRA17.pdf)

.. [Bagh17b] Bagheri, S., Konen, W., Bäck, T. **Comparing Kriging and Radial Basis Function Surrogates.** In: Hoffmann, Frank; Hüllermeier, Eyke (Hrsg.): Proceedings 27. Workshop Computational Intelligence, S. 243-259, Universitätsverlag Karlsruhe, 2017. [URL](https://publikationen.bibliothek.kit.edu/1000074341)

.. [Bagh18] Bagheri, S., Konen, W., Bäck, T. **How to Solve the Dilemma of Margin-Based Equality Handling Methods.** In: Hoffmann, Frank; Hüllermeier, Eyke; Mikut, Ralf (Hrsg.): Proceedings 27. Workshop Computational Intelligence, S. 257-270, Universitätsverlag Karlsruhe, 2017. (**Young Author Award**) [URL](https://blogs.gm.fh-koeln.de/ciop/files/2018/12/GMA2018.pdf)