User guide
===============

If you are looking for the methodology of Physrisk or information about the different hazard indicators and vulnerability models, please check the `methodology document <methodology.html>`_. If it is not library-related, it should be there. Please see also `GitHub Contributing <https://github.com/os-climate/physrisk/blob/main/CONTRIBUTING.md>`_ for how to get set up.

This section documents the structures and conventions of Physrisk and assumes some familiarity with the calculation of Physical Climate Risk (see `methodology document <methodology.html>`_ introduction).

Introduction to Physrisk
------------------------
Physic comprises:

* A :code:`HazardModel` that retrieves *hazard indicators* for different locations. 
* :code:`VulnerabilityModels` that assess the vulnerability of assets to different climate hazards. :code:`VulnerabilityModels` use hazard indicators requested from the :code:`HazardModel` to calculate the *impact* of a hazard on a collection of assets.
* Financial models that use the impacts calculated by the :code:`VulnerabilityModels` to calculate risk measures and scores.

:code:`VulnerabilityModels` request hazard indicators using an :code:`indicator_id` (e.g. 'flood_depth' for inundation, 'max_speed' for wind). It is the responsibility of the :code:`HazardModel` to select the source of the hazard indicator data.

