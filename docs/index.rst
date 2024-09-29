Documentation for physrisk
========================================

This website contains the documentation for physrisk, a calculation engine for physical climate risk.


.. _cards-clickable:

..
  list with all the possible icons for the grid
  https://sphinx-design.readthedocs.io/en/latest/badges_buttons.html

.. raw:: html

   <style>
       .grid {
           display: grid;
           grid-template-columns: repeat(2, 1fr);
           grid-template-rows: repeat(4, 1fr);
           grid-gap: 10px; /* Ajusta el valor según el espaciado deseado */
           gap: 100px 100px;
       }
   </style>


.. grid:: 2
    :gutter: 1

    .. grid-item-card::  Getting started
      :link: getting-started.html
      :text-align: center

      :octicon:`location;5em;sd-text-info`
      ^^^
      Tutorials and walk-throughs.

    .. grid-item-card::  Methodology
      :link: methodology.html
      :text-align: center

      :octicon:`book;5em;sd-text-info`
      ^^^
      Main methodology document.

.. grid:: 2
    :gutter: 1

    .. grid-item-card::  User guide
      :link: user-guide.html
      :text-align: center

      :octicon:`upload;5em;sd-text-info`
      ^^^
      Technical notes for users/contributors.


    .. grid-item-card::  API reference
      :link: api/physrisk.html
      :text-align: center

      :octicon:`code;5em;sd-text-info`
      ^^^
      API reference derived from code.

Physical risk and resilience
=============================

Physrisk is a library for assessing the physical effects of climate change and thereby the potential benefit of measures to improve resilience.
Physrisk is primarily designed to run 'bottom-up' calculations that model the impact of climate hazards on large numbers of individual assets
(including natural). These calculations can be used to assess financial risks or socio-economic impacts. To do this physrisk collects:

- hazard indicators and
- models of the vulnerability of assets to hazards.

Hazard indicators, that is quantities that quantify a hazard, are on-boarded from public data sets or inferred from climate projections, e.g. from CMIP or CORDEX data sets. In both cases, indicator data is created from code in the
`hazard repo <https://github.com/os-climate/hazard>`_, open-source to make the data lineage as transparent as possible.

The `physrisk repo <https://github.com/os-climate/physrisk>`_ contains the main calculation engine. 

Physrisk is also designed to be a hosted, e.g. to provide on-demand calculations; the project is a co-operative of members some of whom need to integrate physical risk calculation into other systems. `physrisk-api <https://github.com/os-climate/physrisk-api>`_ and `physrisk-ui <https://github.com/os-climate/physrisk-ui>`_ provide an example API and user interface.

A `development or 'sandbox' version of the UI <https://physrisk-ui-sandbox.apps.odh-cl1.apps.os-climate.org>`_ is hosted by OS-Climate. Although used for demonstration and test, this is a useful way to explore the available hazard data.

Also see the `OSC webpage <https://os-climate.org/physical-risk-resilience/>`_

The getting-started section contains a number of examples that provide a walk-through of physrisk's functionality, via API then running locally. The methodology document gives a more detailed introduction to the subject and describes the models in more formal detail. The user-guide describes the design and conventions of physrisk and implementation detail for sourcing hazard indicator and vulnerability data; it is there, and in the code, that the details of data sources can be found.

.. image:: images/PRR-5.jpg
  :width: 800

|

.. image:: images/PRR-6.jpg
  :width: 800



Contents
==========

.. toctree::
  :maxdepth: 2

  getting_started/getting_started.rst
  methodology
  user_guide/user_guide.rst
  api/physrisk

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
