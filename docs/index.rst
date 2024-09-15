Physrisk documentation
========================================

This website contains the documentation for Physrisk, a calculation engine for Physical Climate Risk.


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

Physical Risk and Resilience
=============================

Physrisk is a library for assessing the physical effects of climate change and thereby the potential benefit of measures to improve resilience.
Physrisk is primarily designed to run 'bottom-up' calculations that model the impact of climate hazards on large numbers of individual assets
(including natural) and operations. These calculations can be used to assess financial risks or socio-economic impacts. To do this physrisk collects:

- hazard indicators and
- models of vulnerability of assets/operations to hazards.

Hazard indicators are on-boarded from public resources or inferred from climate projections, e.g. from CMIP or CORDEX data sets. Indicators are created from code in the
[hazard repo](https://github.com/os-climate/hazard) to make calculations as transparent as possible.

Physrisk is also designed to be a hosted, e.g. to provide on-demand calculations. [physrisk-api](https://github.com/os-climate/physrisk-api) and [physrisk-ui](https://github.com/os-climate/physrisk-ui) provide an example API and user interface.
A [development version of the UI](https://physrisk-ui-sandbox.apps.odh-cl1.apps.os-climate.org) is hosted by OS-Climate.

Please also see the `OSC webpage <https://os-climate.org/physical-risk-resilience/>`_


.. image:: images/PRR-5.jpg
  :width: 800

|

.. image:: images/PRR-6.jpg
  :width: 800



Contents
==========

.. toctree::
  :maxdepth: 2

  getting-started
  methodology
  user-guide
  api/physrisk

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
