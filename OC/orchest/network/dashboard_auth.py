"""
Access control for the dashboard HTTP API.

The dashboard must listen on the LAN: a worker's dashboard relays chat and generation to the
host's dashboard over HTTP. The API can delete models, clear caches, re-point a worker to
another host and start generation, so it is not left open to the whole network:

  - requests from this machine (loopback) are allowed, so the local browser UI keeps working;
  - requests from other machines to /api/* must carry the shared token in the
    X-Orchest-Token header (ORCHEST_DASHBOARD_TOKEN, set to the same value on every machine);
  - with no token configured, remote /api/* access is refused (fail closed).

Static UI files are not protected; they hold no data and change nothing.
"""

import hmac
import ipaddress
import os
from typing import Dict, Optional

TOKEN_ENV = "ORCHEST_DASHBOARD_TOKEN"
TOKEN_HEADER = "X-Orchest-Token"
MIN_TOKEN_LEN = 16


def configured_token() -> Optional[str]:
    token = os.environ.get(TOKEN_ENV, "").strip()
    return token or None


def is_loopback(host: Optional[str]) -> bool:
    if not host:
        return False
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host.split("%")[0]).is_loopback
    except ValueError:
        return False


def check_request(client_host: Optional[str], path: str, presented_token: Optional[str]) -> Optional[str]:
    """Returns None when the request may proceed, else the reason it is refused."""
    if not path.startswith("/api/"):
        return None
    if is_loopback(client_host):
        return None
    token = configured_token()
    if token is None:
        return (
            f"Remote dashboard API access is disabled. Set {TOKEN_ENV} (at least {MIN_TOKEN_LEN} "
            f"characters, the same on every machine) to allow it."
        )
    if len(token) < MIN_TOKEN_LEN:
        return f"{TOKEN_ENV} is too short (minimum {MIN_TOKEN_LEN} characters); remote API access refused."
    if not presented_token or not hmac.compare_digest(presented_token.encode(), token.encode()):
        return f"Missing or invalid {TOKEN_HEADER} header."
    return None


def auth_headers() -> Dict[str, str]:
    """Headers a dashboard adds when it calls another machine's dashboard API."""
    token = configured_token()
    return {TOKEN_HEADER: token} if token else {}
