Physrisk
==============================
Physical risk calculation engine.

```
pip install physrisk-lib
```

Access to hazard event data requires setting of environment variables specifying the S3 Bucket, for example:

```
OSC_S3_BUCKET=physrisk-hazard-indicators
OSC_S3_ACCESS_KEY=**********6I
OSC_S3_SECRET_KEY=**********mS
```

For use in a Jupyter environment, it is recommended to put the environment variables in a credentials.env file and do, for example:
```
from dotenv import load_dotenv
load_dotenv(dotenv_path=dotenv_path, override=True)
```
