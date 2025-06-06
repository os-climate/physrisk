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
 [![pre-commit.ci status badge]][pre-commit.ci results page] [![TestPyPI](https://img.shields.io/pypi/v/physrisk-lib?logo=python&label=TestPyPi&logoColor=white&color=32C955&pypiBaseUrl=https://test.pypi.org)](https://test.pypi.org/project/physrisk-lib) [![🧪 GitHub Actions CI/CD workflow tests badge]][GHA workflow runs list] [![🔐 CodeQL](https://github.com/os-climate/physrisk/actions/workflows/codeql.yml/badge.svg)](https://github.com/os-climate/physrisk/actions/workflows/codeql.yml) [![OpenSSF Scorecard](https://api.scorecard.dev/projects/github.com/os-climate/physrisk/badge)](https://scorecard.dev/viewer/?uri=github.com/os-climate/physrisk)
<!-- prettier-ignore-end -->

# Physrisk

Physical climate risk calculation engine.

![OS-Climate Logo](docs/images/OS-Climate-Logo.png)

## About physrisk

An [OS-Climate](https://os-climate.org) project, physrisk is a library for
assessing the physical effects of climate change and thereby the potential
benefit of measures to improve resilience.

An introduction and methodology is available in the
[Physical Risk Methodology document](https://github.com/os-climate/physrisk/blob/main/methodology/PhysicalRiskMethodology.pdf).

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
[development version of the UI](https://physrisk-ui-physrisk.apps.odh-cl2.apps.os-climate.org)
is hosted by OS-Climate.

## Using the library

The library can be run locally, although access to the hazard indicator data
is needed. The library is installed via:

```bash
pip install physrisk-lib
```

Hazard indicator data is freely available. Members of the project are able to
access OS-Climate S3 buckets. Credentials are available from the
[OS-Climate S3 keys](https://console-openshift-console.apps.odh-cl2.apps.os-climate.org/k8s/ns/physrisk/secrets/physrisk-s3-keys).
Information about the project is available via the
[community-hub](https://github.com/os-climate/OS-Climate-Community-Hub).
Non-members are able to download or copy hazard indicator data.

Hazard indicator data can be downloaded or copied from the
'os-climate-public-data' bucket. A list of the keys to copy is available from
<https://os-climate-public-data.s3.amazonaws.com/hazard/keys.txt>

An inventory of the hazard data is maintained in the
[hazard inventory](https://github.com/os-climate/hazard/blob/main/src/inventories/hazard/inventory.json)
(this is used by the physrisk library itself). The
[UI hazard viewer](https://physrisk-ui-physrisk.apps.odh-cl2.apps.os-climate.org)
is a convenient way to browse data sets.

Access to hazard event data requires setting of environment variables
specifying the S3 Bucket, for example:

```bash
OSC_S3_BUCKET=physrisk-hazard-indicators
OSC_S3_ACCESS_KEY=**********
OSC_S3_SECRET_KEY=**********
```

For use in a Jupyter environment, it is recommended to put the environment
variables in a credentials.env file and do, for example:

```python
from dotenv import load_dotenv
load_dotenv(dotenv_path=dotenv_path, override=True)
```

[🧪 GitHub Actions CI/CD workflow tests badge]: https://github.com/os-climate/physrisk/actions/workflows/build-test.yaml/badge.svg
[GHA workflow runs list]: https://github.com/os-climate/physrisk/actions/workflows/build-test.yaml
[pre-commit.ci results page]: https://results.pre-commit.ci/latest/github/os-climate/physrisk/main
[pre-commit.ci status badge]: https://results.pre-commit.ci/badge/github/os-climate/physrisk/main.svg
