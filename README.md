Physrisk
==============================
Physical climate risk calculation engine.

<img src="docs/images/OS-Climate-Logo.png" alt="drawing" width="150"/>


## About physrisk

An [OS-Climate](https://os-climate.org) project, physrisk is a library for assessing the physical effects of climate change and thereby the potential benefit of measures to improve resilience.

An introduction and methodology is available [here](https://github.com/os-climate/physrisk/blob/main/methodology/PhysicalRiskMethodology.pdf).

Physrisk is primarily designed to run 'bottom-up' calculations that model the impact of climate hazards on large numbers of individual assets (including natural) and operations. These calculations can be used to assess financial risks or socio-economic impacts. To do this physrisk collects:

- hazard indicators and
- models of vulnerability of assets/operations to hazards.

Hazard indicators are on-boarded from public resources or inferred from climate projections, e.g. from CMIP or CORDEX data sets. Indicators are created from code in the
[hazard repo](https://github.com/os-climate/hazard) to make calculations as transparent as possible.

Physrisk is also designed to be a hosted, e.g. to provide on-demand calculations. [physrisk-api](https://github.com/os-climate/physrisk-api) and [physrisk-ui](https://github.com/os-climate/physrisk-ui) provide an example API and user interface. A [development version of the UI](https://physrisk-ui-sandbox.apps.odh-cl1.apps.os-climate.org) is hosted by OS-Climate.

## Using the library
The library can be run locally, although access to the hazard indicator data is needed. The library is installed via:
```
pip install physrisk-lib
```

Hazard indicator data is freely available. Members of the project are able to access OS-Climate S3 buckets. Credentials are available [here](https://console-openshift-console.apps.odh-cl1.apps.os-climate.org/k8s/ns/sandbox/secrets/physrisk-s3-keys). Information about the project is available via the [community-hub](https://github.com/os-climate/OS-Climate-Community-Hub). Non-members are able to download or copy hazard indicator data.

Hazard indicator data can be downloaded or copied from the 'os-climate-public-data' bucket. A list of the keys to copy is available from
https://os-climate-public-data.s3.amazonaws.com/hazard/keys.txt

An inventory of the hazard data is maintained [here](https://github.com/os-climate/hazard/blob/main/src/inventories/hazard/inventory.json) (this is used by the physrisk library itself). The [UI hazard viewer](https://physrisk-ui-sandbox.apps.odh-cl1.apps.os-climate.org) is a convenient way to browse data sets.

Access to hazard event data requires setting of environment variables specifying the S3 Bucket, for example:

```
OSC_S3_BUCKET=physrisk-hazard-indicators
OSC_S3_ACCESS_KEY=**********
OSC_S3_SECRET_KEY=**********
```

For use in a Jupyter environment, it is recommended to put the environment variables in a credentials.env file and do, for example:
```
from dotenv import load_dotenv
load_dotenv(dotenv_path=dotenv_path, override=True)
```
