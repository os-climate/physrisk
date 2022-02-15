Onboarding a new model/data
===========================

Exceedance curves 
-----------------

A curve of hazard event intensities is obtained at the location of one partcular asset. The
probability of occurrence is either a 'return period' or an 'exceedance probability' (the reciprocal
of the return period).

.. image:: onboarding/return_periods.png
  :width: 600
  
.. image:: onboarding/exceedance_curve.png
  :width: 600

Probability bins can then be obtained from the exceedance curve, by subtracting one cumulative probability from another.

On-boarding a model based on a damage/disruption curve
------------------------------------------------------

.. image:: onboarding/disruption_curve.png
  :width: 600

.. image:: onboarding/vulnerability_curve.png
  :width: 600

On-boarding a model based on a damage/disruption curve with uncertainty
-----------------------------------------------------------------------

.. image:: onboarding/damage_with_uncertainty.png
  :width: 600
  
Include some code
::
  import math
