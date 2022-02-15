Onboarding a new model/data
===========================

Exceedance curves 
-----------------

A curve of hazard event intensities is obtained at the location of one particular asset. The
probability of occurrence is either a 'return period' or an 'exceedance probability' (the reciprocal
of the return period).

.. image:: onboarding/return_periods.png
  :width: 600
  
.. image:: onboarding/exceedance_curve.png
  :width: 600

Probability bins can then be obtained from the exceedance curve, by subtracting one cumulative probability from another.

.. image:: onboarding/histo_from_exceedance.png
  :width: 600

In code this can be done by
::
  exceedance_curve = ExceedanceCurve(1.0 / return_periods, event_intensities)
  intensity_bins, probs = exceedance_curve.get_probability_bins()
  
Vulnerability/Event Model 
-------------------------
In general a Vulnerability/Event Model is responsible for obtaining for a particular asset: 
#. ::HazardEventDistrib::: provides probabilities of hazard event intensentities for the asset
#. VulnerabilityDistrib: provides conditional probabilities that given a hazard event of given intensity has occurred, a loss wwill occur of a given amount

A loss is either a damage or a disruption.

The current implementation is non-parametric and based on discete bins, although a continuous HazardEventDistrib/VulnerabilityDistrib can certainly be added if desired. 
 
HazardEventDistrib is in this non-parametric version a histogram of hazard event intensities: defines a set of intensity bins and the annual probability of occurrence.

VulnerabilityDistrib is a matix that provides the probability that is an event occurs for a particular intensity bin, wew see an impact in a particular impact bin.

* A type of hazard event (Inundation, Wildfire, Drought etc)
*


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
