ssp126 = """The SSP126 scenario represents a strong mitigation path under the Paris Agreement to limit warming to
1.5–2.0 °C relative to the pre-industrial level, leading to a 2.6 W/m<sup>2</sup> of additional radiative forcing by 2100.
It can be understood as the RCP2.6 scenario combined with additional socio-economic assumptions."""

ssp245 = """The SSP245 scenario is based on ‘middle-of-the-road’ projections of development (SSP2),
resulting in an additional radiative forcing of 4.5 W/m<sup>2</sup> by 2100.
It can be understood as the RCP4.5 scenario combined with additional socio-economic assumptions."""

ssp585 = """The SSP585 scenario is based on ‘Fossil Fuelled Development’ projections, leading to an additional
radiative forcing of 8.5 W/m<sup>2</sup> and 5°C of projected warming by 2100. It can be understood as the
RCP8.5 scenario combined with additional socio-economic assumptions."""

historical = """The historical scenario is based on the current climate."""


def description():
    return {
        "ssp126": ssp126,
        "ssp245": ssp245,
        "ssp585": ssp585,
        "historical": historical,
    }
