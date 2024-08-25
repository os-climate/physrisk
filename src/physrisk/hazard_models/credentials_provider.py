from os import getenv
from typing import Dict, Protocol


class CredentialsProvider(Protocol):
    def jba_access_key(self) -> str: ...

    def jba_api_disabled(self) -> bool: ...

    def jupiter_client_id(self) -> str: ...

    def jupiter_client_secret(self) -> str: ...

    def jupiter_refresh_token(self) -> str: ...

    def proxies(self) -> Dict[str, str]: ...


class EnvCredentialsProvider(CredentialsProvider):
    def __init__(self, disable_api_calls=True):
        self._disable_api_calls = disable_api_calls

    def jba_access_key(self) -> str:
        return getenv(
            "JBA_TOKEN", getenv("JBA_PROD_TOKEN", "")
        )  # for back compatibility, alllow JBA_PROD_TOKEN also

    def jba_api_disabled(self) -> bool:
        return self._disable_api_calls

    def jupiter_client_id(self) -> str:
        return getenv("JUPITER_CLIENT_ID", "")

    def jupiter_client_secret(self) -> str:
        return getenv("JUPITER_CLIENT_SECRET", "")

    def jupiter_refresh_token(self) -> str:
        return getenv("JUPITER_REFRESH_TOKEN", "")

    def proxies(self) -> Dict[str, str]:
        return {"https": getenv("PROXY_HTTPS", ""), "http": getenv("PROXY_HTTP", "")}
