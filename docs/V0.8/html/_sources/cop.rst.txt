--------------------------------
Constraint Optimization Problems
--------------------------------

This document defines COPs (Constraint Optimization Problems) and introduces the G-problem benchmark.


COPs
-----------------

A constrained optimization problem (COP) for numerical and continuous quantities in :math:`\mathbb{R}^d` is defined as:


.. raw:: latex html

	\[ Min    \quad  f(\vec{x}), \quad \vec{x} \in [\vec{a},\vec{b}] \subset \mathbb{R}^d \]
.. raw:: latex html

	\[ \text{subject to}  \quad  g_{i}(\vec{x}) \leq 0,  \quad i=1,2,\ldots,m \]
.. raw:: latex html

	$$ 	\quad\qquad\qquad   h_{j}(\vec{x})   =  0,  \quad j=1,2,\ldots,r $$



G-problem benchmark
--------------------------------

The G-problem benchmark suite originates from a CEC 2006 competition [LiangRunar]_. It is a set of 24 constrained optimization problems (COPs, G-problems) G01, ..., G24 with various properties like dimension, number of equality / inequality constraints, feasibility ratio, etc. Eight of the 24 COPs have equality constraints. Although these problems were introduced as a suite in the technical report [LiangRunar]_ at CEC 2006, many of them have been used by different authors earlier.

The G-problems are available in **SACOBRA_Py** as objects of class ``GCOP``:

.. autoclass:: gCOP.GCOP


.. [LiangRunar] J. Liang, T. P. Runarsson, E. Mezura-Montes, M. Clerc, P. Suganthan, C. C. Coello, and K. Deb, “Problem definitions and evaluation criteria for the CEC 2006 special session on constrained real-parameter optimization,” Journal of Applied Mechanics, vol. 41, p. 8, 2006. `http://www.lania.mx/~emezura/util/files/tr_cec06.pdf <http://www.lania.mx/~emezura/util/files/tr_cec06.pdf>`_

