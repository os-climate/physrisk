<!-- markdownlint-disable -->
<!-- prettier-ignore-start -->
> [!IMPORTANT]
> On June 26 2024, Linux Foundation announced the merger of its financial services umbrella, the Fintech Open Source Foundation ([FINOS](https://finos.org)), with OS-Climate, an open source community dedicated to building data technologies, modeling, and analytic tools that will drive global capital flows into climate change mitigation and resilience; OS-Climate projects are in the process of transitioning to the [FINOS governance framework](https://community.finos.org/docs/governance); read more on [finos.org/press/finos-join-forces-os-open-source-climate-sustainability-esg](https://finos.org/press/finos-join-forces-os-open-source-climate-sustainability-esg)
<!-- prettier-ignore-end -->
<!-- markdownlint-enable -->

<!-- prettier-ignore-start -->
<!-- markdownlint-disable-next-line MD013 -->
[![OS-Climate](https://img.shields.io/badge/OS-Climate-blue)](https://os-climate.org/) [![Slack](https://img.shields.io/badge/slack-osclimate-blue.svg?logo=slack)](https://os-climate.slack.com) [![Source Code](https://img.shields.io/badge/GitHub-100000?logo=github&logoColor=white&color=blue)](https://github.com/os-climate/physrisk) [![PyPI](https://img.shields.io/pypi/v/physrisk-lib?logo=python&logoColor=white&color=blue)](https://pypi.org/project/physrisk-lib) [![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

<!-- markdownlint-disable-next-line MD013 -->
 [![pre-commit.ci status badge]][pre-commit.ci results page] [![TestPyPI](https://img.shields.io/pypi/v/physrisk-lib?logo=python&label=TestPyPi&logoColor=white&color=32C955&pypiBaseUrl=https://test.pypi.org)](https://test.pypi.org/project/physrisk-lib) [![Python Build/Test](https://github.com/os-climate/physrisk/actions/workflows/build-test.yaml/badge.svg?branch=main)](https://github.com/os-climate/physrisk/actions/workflows/build-test.yaml) [![🔐 CodeQL](https://github.com/os-climate/physrisk/actions/workflows/codeql.yml/badge.svg)](https://github.com/os-climate/physrisk/actions/workflows/codeql.yml) [![OpenSSF Scorecard](https://api.scorecard.dev/projects/github.com/os-climate/physrisk/badge)](https://scorecard.dev/viewer/?uri=github.com/os-climate/physrisk)
<!-- prettier-ignore-end -->

# Physrisk

Physical climate risk calculation engine.

<img src="docs/images/OS-Climate-Logo.png" alt="drawing" width="150"/>

## About physrisk

An [OS-Climate](https://os-climate.org) project, physrisk is a library for
assessing the physical effects of climate change and thereby the potential
benefit of measures to improve resilience.

An introduction and methodology can be found in the
[online documentation](https://physrisk.readthedocs.io/en/latest/).

Physrisk is primarily designed to run 'bottom-up' calculations that model
the impact of climate hazards on large numbers of individual assets
(including natural) and operations. These calculations can be used to assess
financial risks or socio-economic impacts. To do this physrisk collects:

- hazard indicators and
- models of vulnerability of assets/operations to hazards.

Hazard indicators are on-boarded from public resources or inferred from
climate projections, e.g. from CMIP or CORDEX data sets. Indicators are
created from code in the
[hazard repository](https://github.com/os-climate/hazard) to make
calculations as transparent as possible.

Physrisk is also designed to be a hosted, e.g. to provide on-demand
calculations.
[physrisk-api](https://github.com/os-climate/physrisk-api) and
[physrisk-ui](https://github.com/os-climate/physrisk-ui) provide an example
API and user interface. A
[development version of the UI](https://physrisk-ui-physrisk.apps.osc-cl1.apps.os-climate.org)
is hosted by OS-Climate.

## Using the library

The library can be run locally. The library is installed via:

```bash
pip install physrisk-lib
```

Hazard indicator data is freely available via the [Amazon Sustainability Data Initiative, here](https://registry.opendata.aws/os-climate-physrisk/).
Information about the project is available via the
[community-hub](https://github.com/os-climate/OS-Climate-Community-Hub).

An inventory of the hazard data is maintained in the
[hazard inventory](https://github.com/os-climate/hazard/blob/main/src/inventories/hazard/inventory.json)
(this is used by the physrisk library itself). The
[UI hazard viewer](https://physrisk-ui-physrisk.apps.osc-cl1.apps.os-climate.org)
is a convenient way to browse data sets.

A good place to start is the Getting Started section in the documentation site which has a number of walk-throughs.

[pre-commit.ci results page]: https://results.pre-commit.ci/latest/github/os-climate/physrisk/main
[pre-commit.ci status badge]: https://results.pre-commit.ci/badge/github/os-climate/physrisk/main.svg
