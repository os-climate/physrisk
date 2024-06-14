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

Note that units of the quantity are provided to the :code:`VulnerabilityModel` by the :code:`HazardModel`.

Hazard indicator data sets
-------------------------
The :code:`HazardModel` retrieves hazard indicators in a number of ways and can be made composite in order to combine different ways of accessing the data. At time of writing the common cases are that:

1. Hazard indicator data is stored in `Zarr <https://zarr.readthedocs.io/en/stable/>`_ format (in an arbitrary Zarr store, although S3 is a popular choice).
2. Hazard indicator data is retrieved via call to an external API. This is mainly used when combining commercial data to the public-domain.

In case 1, hazard indicators are stored as three dimensional arrays. The array is ordered :math:`(z, y, x)` where :math:`y` is the spatial :math:`y` coordinate, :math:`x` is the spatial :math:`x` coordinate and :math:`z` is an *index* coordinate. The *index* takes on different meanings according to the type of data being stored.

Indicators can be either:

* Acute (A): the data comprises a set of hazard intensities for different return periods. In this case *index* refers to the different return periods.
* Parametric (P): the data comprises a set of parameters. Here *index* refers to the different parameters. The parameters may be single values, or *index* might refer to a set of thresholds. Parametric indicators are used for chronic hazards.

As mentioned above, :code:`VulnerabilityModels` only specify the identifier of the hazard indicator that is required, as well as the climate scenario ID and the year of the future projection. This means that hazard indicator ID uniquely defines the data. For example, a vulnerability model requesting 'flood depth' could have data returned from a variety of data sets, depending on how the :code:`HazardModel` is configured. But

+-----------------------+-------------------------------+---------------------------------------+
| Hazard class          | Indicator ID (type)           | Description                           |
+=======================+===============================+=======================================+
| CoastalInundation,    | flood_depth (A)               | Flood depth (m) for available         |
| PluvialInundation,    |                               | return periods. This is unprotected   |
| RiverineInundation    |                               | depth.                                |
|                       +-------------------------------+---------------------------------------+
|                       | sop (P)                       | Standard of protection                |
|                       |                               | (as return period in years).          |
+-----------------------+-------------------------------+---------------------------------------+
| Fire                  | fire_probability (P)          | Annual probability that location      |
|                       |                               | is in a wildfire zone.                |
+-----------------------+-------------------------------+---------------------------------------+
| Heat                  | mean_degree_days/above/index  | Mean mean-temperature degree days per |
|                       | (P)                           | year above a set of temperature       |
|                       |                               | threshold indices.                    |
+-----------------------+-------------------------------+---------------------------------------+
| Drought               | months/spei/12m/below/index   | Mean months per year where the 12     |
|                       | (P)                           | month SPEI index is below a set of    |
|                       |                               | indices.                              |
+-----------------------+-------------------------------+---------------------------------------+
| Wind                  | max_speed                     | Maximum 1 minute sustained wind speed |
|                       | (A)                           | for available return periods.         |
+-----------------------+-------------------------------+---------------------------------------+
