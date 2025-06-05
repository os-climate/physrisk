User Guide
==========

If you are looking for the methodology of Physrisk, please check the `methodology document <methodology.html>`_. If you are looking for library-related information (e.g. conventions, formats, structure) this is in the the User Guide. For information about the different hazard indicator models and vulnerability models, this is split:

* the methodology document contains a summary of the different sources of hazard and vulnerability models and references; however
* details about how data sets are transformed and used in Physrisk are in the User Guide.

As an example, consider a hazard indicator data set of wind speed return periods. Documentation detailing how the data set is sourced and transformed for use in Physrisk would be in the User Guide, including links, code snippets, graphs and equations (perhaps the data set gives changes in return periods which need to be converted into wind speed). By convention, each hazard indicator model or vulnerability model has its own Jupyter notebook for documentation that is found in the User Guide.

For developing Physrisk please see also `GitHub Contributing <https://github.com/os-climate/physrisk/blob/main/CONTRIBUTING.md>`_ for how to get set up.

The following sections document the structures and conventions of Physrisk and assumes some familiarity with the calculation of Physical Climate Risk (see `methodology document <methodology.html>`_ introduction).


.. toctree::
   :maxdepth: 2

   introduction
   hazard_indicators/hazard_indicators
   vulnerability/vulnerability
   hosting/hosting
