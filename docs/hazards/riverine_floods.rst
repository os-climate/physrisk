
Introduction
-------------------------

Globally, floods are the most damaging type of hazard, accounting for
44% of all disaster events from 2000 to 2019 and affecting 1.6 billion
people worldwide (UNDRR & CRED 2020). Recent studies indicate that river
flooding in the European Union and UK leads to annual damages worth €7.6
billion and exposes approximately 160,000 people to inundation each
year. Furthermore, climate change is exacerbating the severity and
frequency of riverine flooding, with annual flooding more than doubling
in the past four decades. In a 3\ :math:`^\circ` C global warming
scenario, without climate change adaptation, flood damage in Europe
would reach €44 billion per year, posing a risk to nearly half a million
Europeans annually until the end of the century. Given these figures,
scientists emphasize the urgent need for Europe to implement adaptation
measures to mitigate the projected increase in flood risk.
:cite:`JRC_facing` :cite:`nature_summary`

Physical climate risk models typically consist of three main components:
a hazard module, an exposure module, and a vulnerability module. This
model structure is widely accepted in the literature on this topic, and
flood events are no exception.

The hazard module includes information about the specific hazards that
we want to consider in our model, along with their fundamental
characteristics. In our case, it will contain specific information about
flood events that we will discuss further on. Let us stress that in the
hazard module, we do not refer to any group of assets yet. Instead, we
focus on the hazard events themselves. The exposure component includes
information about the assets, including their descriptions and specific
locations. Finally, the vulnerability component serves as a connection
between hazard, exposure, and loss, allowing for the estimation of the
relative damage to an asset based on a specific hazard level. The core
of the vulnerability model is usually given by the so-called damage
functions, which translate the flood intensity into an estimated damage
as a ratio of the total value of the asset.

In the following, we will analyze each of the above components of the
physical risk model in the context of river floods and demonstrate how
to conduct a risk assessment for flood events.

Hazard module
----------------

The aim of the hazard module is to provide relevant information about
the flood events themselves. The main information we are interested in
is related to two issues: potential intensity and frequency of flood
events. Let us stress that "intensity" can be defined in many different
ways, which leads to the concept of the hazard indicator. By hazard
indicator, we mean a quantity that provides relevant information about
the hazard related to its intensity or frequency. The selection of a
hazard metric is a crucial aspect of the hazard model, and it typically
follows a widely accepted approach. However, it is important to note
that the chosen metric may not fully capture all the factors
contributing to damage. For instance, in the context of flood damage,
the primary metric is typically the flood depth, however, factors such
as the duration of inundation, flow velocity, and water pollution may
also have a significant impact :cite:`Mitchel`.

The relevant information related to the selected flood indicator is
presented in the form of a suitable hazard dataset. Below, we will
briefly discuss the most common datasets utilized in flood risk
assessment, with a specific emphasis on return period maps.

Flood datasets
=====================

There are two main approaches to the risk assessment of catastrophe
events, particularly for floods. The first approach, sometimes called
the probabilistic one, involves estimating the probability of flood
events of different intensities and then translating this intensity into
potential damage using the vulnerability component. The second approach,
known as the "event-based" or deterministic approach, simulates
thousands of potential flood events and estimates the flood peril based
on this collection of events. Creating this collection of events can be
achieved using climate models or stochastic analysis based on past
historical events. In our approach, we will rely on the probabilistic
approach, and the remainder of this document will focus entirely on this
methodology.

The choice of the risk assessment method dictates the datasets we rely
on. In the event-based approach, the hazard dataset consists of a
collection of thousands of simulated events that can be used in flood
risk analysis. These datasets are often presented in the form of
event-loss tables. On the other hand, the probabilistic approach mainly
involves working with return period maps of flood events. A return
period map provides information about the likelihood of a flood event of
a given intensity occurring at various locations on a map. An example of
the return period map for a 100-year return period is shown in Figure
`1 <fig:hazard_int_>`_.

.. _fig:hazard_int:

.. figure:: ../images/river_flood/intensity2_riverflood.png
   :align: center
   :width: 80.0%

   Return period map for river floods in various locations in Spain. The
   intensity is measured in terms of flood depth.

The interpretation of the map is as follows: for each point
:math:`(x,y)` on the map, there is a unique value :math:`I` representing
the intensity in that location. The intensity value :math:`I` indicates
that the return period for intensity :math:`I` at point :math:`(x,y)` is
exactly 100 years. In simpler terms, if the intensity at a given point
is :math:`I`, it means that, statistically, at least one event of
intensity equal to or greater than :math:`I` will occur within a
100-year period. It’s important to note that the intensity indicator in
this case is flood depth.

The area that is affected by a flood event is called a flood footprint.
The map displaying the flood footprint is provided with a specific
resolution, which is a crucial measure of the dataset’s accuracy. Return
period maps, used to indicate the intensity and frequency of flood
events, can be created based on historical climate parameters or in
scenario versions that consider various possibilities of climate change.

In practice, we typically require a collection of return period maps for
different return periods to perform a comprehensive risk estimation.

Exposure module
-------------------

Exposure refers to the collection of assets that are susceptible to
potential hazards. The exposure model encompasses data regarding the
assets, properties, and infrastructure, along with their vulnerability
to potential risks. This information serves as a vital input for the
catastrophe model. In practical applications, an exposure database
typically includes the following information:

-  Type of assets (e.g., buildings, infrastructure, agriculture,
   machines, etc.)

-  Location of assets (usually specified in terms of latitude and
   longitude)

-  Value of the assets

The complexity of the exposure component varies depending on the
specific use case. When conducting risk assessment on a macroeconomic
level, such as for a country or region, estimating exposure can be
challenging due to the need for a comprehensive information about
economic properties and services. However, when assessing a portfolio of
assets for a company or bank, the exposure part typically presents fewer
difficulties. It comes from the fact that companies generally possess
detailed information about their assets, which serves as a primary input
for our climate risk model.

It should be stressed that flood events pose a significant risk to a
substantial number of people and assets worldwide. Globally,
approximately 2 billion individuals reside in freshwater flood
inundation zones, accounting for around 25% of the global population.
The level of exposure to river flooding varies across regions, with
Europe, South Asia, and Southeast Asia facing the highest levels of
risk. Just as an example, approximately 23% of the world’s croplands are
situated within inundation areas, including key agricultural nations
such as India (45%), China (31%), and the United States of America (23%)
:cite:`nature_summary`. Information about global exposure to
floods, divided by countries, can be found for instance in
`https://www.marshmclennan.com <https://www.marshmclennan.com/insights/publications/2021/september/marsh-mclennan-flood-risk-index.html>`__.

Vulnerability module
-------------------------

While in the hazard module we are interested in hazard events
themselves, the aim of the vulnerability module is to translate the
intensity of a hazard to the damage incurred by the assets exposed to
it. This damage is usually measured by various metrics, such as the
damage ratio. Since damage will strongly depend on the hazard and
exposure characteristics, it is naturally built on the foundation of the
hazard and exposure modules. More precisely, the output of the hazard
and exposure modules is usually used as the input for the vulnerability
module. The main concept in the vulnerability module is the so-called
damage functions that quantify the impact of a hazard intensity on a
given asset :cite:`Mitchel`.

In particular the above framework may be applied to flood events.
Assessing the potential damage caused by flood events is an essential
component of effective flood risk management. To estimate direct flood
damage, depth-damage curves are commonly employed. These curves provide
information on the expected flood damage for specific water depths,
categorized by assets or land-use classes. Figure 1.2 illustrates a
damage function for residential, commercial and industrial buildings
exposed to floods, sourced from the Joint Research Centre under the
European Commission. The dataset with this damage function can be
downloaded from the Joint Research Centre repository
:cite:`Houz`. The plots illustrate how flood intensity
(flood depth) is transformed into potential damage for different types
of buildings. As the flood depth increases, the damage also rises,
reaching 100% for all building types when the flood depth reaches
approximately 6 meters.

Though several countries have created flood damage models using
historical flood data and expert input, the lack of comprehensive
depth-damage curves across all regions poses a challenge. Additionally,
variations in methodologies employed by different countries for damage
modeling make direct comparisons difficult and limit the feasibility of
supra-national flood damage assessments :cite:`Houz`.

.. _fig:damage1_riv:

.. figure:: ../images/river_flood/damage_funs.png
   :align: center
   :width: 80.0%

   The plots show the relationship between flood depth and the
   corresponding damage factor, ranging from 0% to 100%, for three types
   of assets: residential buildings, commercial buildings, and
   industrial buildings. In all cases, the damage reaches 100% when the
   flood depth approaches approximately 6 meters.

Impact assessment
-------------------------------

After collecting all the necessary components of hazard, exposure, and
vulnerability, we proceed with the most important part, which is risk
assessment. We usually follow these steps: First, we use the return
period maps to determine the flood intensity associated with each
location of the area of interest. Next, we map the flood intensities
data onto the exposure map to identify the specific flood hazard level
that each asset faces. Then, we estimate the potential damage to each
asset by applying the appropriate damage function, which quantifies the
relationship between flood intensity and asset vulnerability. By
utilizing these functions, we can calculate the expected level of damage
or loss for each asset based on the corresponding flood intensity.

Once the asset damage estimates are obtained, we aggregate and analyze
the results to gain an overall assessment of the risk. This involves
summarizing the estimated damages for all exposed assets, calculating
the total expected losses, and identifying areas or assets that are at
higher risk. The final output of the risk assessment is usually provided
in a form of risk metrics that provide information about the risk
related to the portfolio of assets. Common metrics include
:cite:`Mitchel`:

-  Annual Expected Loss (AEL).

-  | Standard deviation (SD) around the AAL
   | SD is a measure of the volatility of loss around the AAL.

-  | Occurrence Exceedance Probability (OEP).
   | OEP is the probability that the maximum event loss in a year
     exceeds a given level.

-  | Aggregate Exceedance Probability (AEP).
   | AEP is the probability that the sum of event losses in a year
     exceeds a given level.

-  | Value at risk (VaR).
   | VaR is the loss value at a specific quantile of the relevant loss
     distribution.

Additionally, by considering factors such as asset valuation,
replacement costs, business interruption losses, and indirect expenses,
a more comprehensive estimation of the financial impact can be achieved.

Example - Flood risk assessment for powerplants in Spain
-------------------------------------------------------------------

| In this section, we will briefly demonstrate how to perform a risk
  assessment for flood events using the example of power plants in
  Spain. The entire process will be executed using the open-source
  platform CLIMADA, but one can also utilize other similar open-source
  or commercial tools of this kind (see for instance OS-climate
  platform). The documentation related to the CLIMADA platform can be
  found here:
| `https://climada-python.readthedocs <https://climada-python.readthedocs.io/en/stable/>`__.

CLIMADA stands for CLIMate ADAptation and is a probabilistic natural
catastrophe impact model developed and maintained mainly by the Weather
and Climate Risks Group at ETH Zürich. It provides a software tool
designed to assess and analyze climate-related risks and impacts for
various hazards, such as floods, storms, heatwaves, and droughts, and
their potential consequences on different sectors, including
infrastructure, agriculture, and human populations. The CLIMADA platform
integrates advanced climate models, geospatial data, and statistical
methods to simulate and visualize the potential impacts of climate
events.

Hazard
===================

For our example we have used river flood hazard maps prepared spanish
Ministerio para la Transición Ecologica y el Reto Demografico. The
dataset consist of return period maps in a historic scenario for three
different return periods: 10, 100, 500 years. The maps cover the region
of entire Spain and its resolution is equal to 1m. The data can be
downloaded from the following link:
`https://www.miteco.gob.es <https://www.miteco.gob.es/es/cartografia-y-sig/ide/descargas/agua/Mapas-peligrosidad-por-inundacion-fluvial.aspx>`__

Using the CLIMADA platform’s Riverflood python class, we can visualize
the datasets as a map. The figure `3 <intensity_climada_>`_ displays
the return period map representing the intensity of the river flood for
a 10-year return period from the dataset we used.

.. _intensity_climada:

.. figure:: ../images/river_flood/intensity2.png
   :align: center
   :width: 100.0%

   The flood intensity in Spain represented in terms of a flood depth
   for a 10-year return period.

Exposure
==============

As an example of the asset portfolio in the exposure part, we utilized
the dataset from the Global Power Plant Database, a global and
open-source database of power plants. The dataset includes a set of
power plants in Spain and is accessible at
`https://datasets.wri.org <https://datasets.wri.org/dataset/globalpowerplantdatabase>`__.
We used the electrical generating capacity in megawatts as a proxy for
the power plant’s value. CLIMADA provides a tool to create a map
representation of the exposure dataset, and its effect can be seen in
Figure `4 <powerplants_exp_riv_>`_. The geographical longitude and latitude
provide the location of the power plants. It is important to note that
the value in USD does not correspond to the actual energy production
value but is solely used to illustrate the differences in energy
production between the power plants in the dataset.

.. _powerplants_exp_riv:

.. figure:: ../images/river_flood/exposure2.png
   :align: center
   :width: 100.0%

   Power plants in Spain, with the energy production serving as a proxy
   for the power plant’s value. The value in USD does not correspond to
   the actual energy production value but is used to illustrate the
   differences in energy production between the power plants in the
   dataset.

Vulnerability
================

Next, we proceeded to the vulnerability module, aiming to convert the
intensity of the river flood into the damage incurred on the power
plants. The damage function utilized in this step was obtained from
Huizinga et al. and can be downloaded from the following link:
`https://publications.jrc.ec.europa.eu <https://publications.jrc.ec.europa.eu/repository/handle/JRC105688>`__.

This paper includes damage functions for six different types of assets.
For the sake of simplicity, we have selected the damage function for
residential buildings. The plot of this function is shown in Figure
`5 <fig:damage2_>`_.

.. _fig:damage2:

.. figure:: ../images/river_flood/damage_function_spain_riverflood.png
   :align: center
   :width: 80.0%

   The plots demonstrate the conversion of flood depth into a damage
   factor ranging from 0% to 100%. Here, MDD represents the mean damage
   (impact) degree, PAA denotes the percentage of affected assets, and
   MDR is the mean damage ratio calculated as MDR =
   MDD\ :math:`\cdot`\ PAA.

The plots demonstrate how flood intensity (flood depth) is translated
into potential damage. The blue curve represents the mean damage ratio
(MDR), which shows the fraction (0%-100%) of the total asset value lost
due to the flood event. For example, from the graph, we can see that a
flood intensity of 2m results in approximately 50% damage to residential
assets. The red line indicates the percentage of affected assets (PAA),
and it is an internal parameter of CLIMADA that is not relevant for us
in this example.

To calculate the damage value to a set of assets, we multiply the value
of each exposed asset in a grid cell by the Mean Damage Ratio
corresponding to the flood intensity in that grid cell. The figure
`6 <impact3_>`_ shows the annual expected impact map.

.. _impact3:

.. figure:: ../images/river_flood/impact3.png
   :align: center
   :width: 100.0%

   Map illustrating the annual expected impact on assets in different
   locations.


Bibliography
---------------------------------

.. bibliography:: ../references.bib
