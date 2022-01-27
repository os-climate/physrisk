Physrisk
==============================
Physical risk calculation engine.

```
pip install pyrisk
```

Access to hazard event data requires setting of environment variables specifying the S3 Bucket, for example:

```
S3_BUCKET=redhat-osc-physical-landing-647521352890
S3_ACCESS_KEY=**********6I
S3_SECRET_KEY=**********mS
```

For use in a Jupyter environment, it is recommended to put the environment variables in a credentials.env file and do, for example:
```
from dotenv import load_dotenv
load_dotenv(dotenv_path=dotenv_path,override=True)
```
