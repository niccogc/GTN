"""
AIM Authentication Patch

Adds support for custom HTTP headers (like Authorization tokens) to AIM's remote tracking.

Usage:
    import os
    from aim_auth import Run  # Use this instead of: from aim import Run

    # Set your auth token
    os.environ['AIM_AUTH_TOKEN'] = 'your-secret-token-here'

    # Connect to remote server (token will be automatically added)
    run = Run(repo='aim://your-server.com:53800')
    run.track(0.5, name='loss', step=1)

Cloudflare Access tokens:
    CF_ACCESS_CLIENT_ID: Cloudflare Access Client ID
    CF_ACCESS_CLIENT_SECRET: Cloudflare Access Client Secret
"""

import os
import json
import requests
from aim import Run as OriginalRun
from aim.ext.transport.client import Client

DEBUG = os.getenv("AIM_AUTH_DEBUG", "").lower() in ("1", "true", "yes")

# Store original methods
_original_client_init = Client.__init__
_original_protocol_probe = Client.protocol_probe
_original_connect = Client.connect
_original_refresh_ws = Client.refresh_ws


def _get_auth_headers():
    """Helper function to get authentication headers from environment variables"""
    headers = {}

    auth_token = os.getenv("AIM_AUTH_TOKEN")
    if auth_token:
        headers["Authorization"] = f"Bearer {auth_token}"
        if DEBUG:
            print(f"[AIM_AUTH] Using Bearer token (length={len(auth_token)})")

    cf_client_id = os.getenv("CF_ACCESS_CLIENT_ID")
    cf_client_secret = os.getenv("CF_ACCESS_CLIENT_SECRET")
    if cf_client_id and cf_client_secret:
        headers["CF-Access-Client-Id"] = cf_client_id
        headers["CF-Access-Client-Secret"] = cf_client_secret
        if DEBUG:
            print(f"[AIM_AUTH] Using CF Access (client_id={cf_client_id[:8]}...)")

    custom_headers = os.getenv("AIM_CUSTOM_HEADERS")
    if custom_headers:
        try:
            headers_dict = json.loads(custom_headers)
            headers.update(headers_dict)
            if DEBUG:
                print(f"[AIM_AUTH] Using custom headers: {list(headers_dict.keys())}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse AIM_CUSTOM_HEADERS as JSON: {e}")

    if DEBUG and not headers:
        print("[AIM_AUTH] WARNING: No authentication headers configured!")

    return headers


def _patched_client_init(self, remote_path: str):
    """Patched Client.__init__ that adds authentication headers"""
    # Call original initialization first
    _original_client_init(self, remote_path)

    # Add authentication headers AFTER original initialization
    # This ensures headers are available for all subsequent requests
    try:
        auth_headers = _get_auth_headers()
        if not hasattr(self, "request_headers"):
            self.request_headers = {}
        self.request_headers.update(auth_headers)
    except ValueError as e:
        raise RuntimeError(f"AIM Auth: {e}")


def _patched_protocol_probe(self):
    """Patched protocol_probe that uses authentication headers"""
    try:
        auth_headers = _get_auth_headers()
    except ValueError as e:
        raise RuntimeError(f"AIM Auth: {e}")

    headers = (
        {**self.request_headers, **auth_headers}
        if hasattr(self, "request_headers")
        else auth_headers
    )

    endpoint = f"http://{self.remote_path}/status/"
    try:
        if DEBUG:
            print(f"[AIM_AUTH] Probing HTTP: {endpoint}")
        response = requests.get(endpoint, headers=headers, timeout=10)
        if DEBUG:
            print(f"[AIM_AUTH] HTTP probe: status={response.status_code}, url={response.url}")
        if response.status_code == 200:
            if response.url.startswith("https://"):
                self._http_protocol = "https://"
                self._ws_protocol = "wss://"
                return
    except Exception as e:
        if DEBUG:
            print(f"[AIM_AUTH] HTTP probe failed: {e}")

    endpoint = f"https://{self.remote_path}/status/"
    try:
        if DEBUG:
            print(f"[AIM_AUTH] Probing HTTPS: {endpoint}")
        response = requests.get(endpoint, headers=headers, timeout=10, verify=self.ssl_certfile)
        if DEBUG:
            print(f"[AIM_AUTH] HTTPS probe: status={response.status_code}")
        if response.status_code == 200:
            self._http_protocol = "https://"
            self._ws_protocol = "wss://"
    except Exception as e:
        if DEBUG:
            print(f"[AIM_AUTH] HTTPS probe failed: {e}")

    endpoint = f"https://{self.remote_path}/status/"
    try:
        response = requests.get(endpoint, headers=headers, timeout=10, verify=self.ssl_certfile)
        if response.status_code == 200:
            self._http_protocol = "https://"
            self._ws_protocol = "wss://"
    except Exception:
        pass


def _patched_connect(self):
    """Patched connect method that ensures authentication headers are used"""
    if not hasattr(self, "request_headers") or not self.request_headers:
        self.request_headers = {}

    try:
        auth_headers = _get_auth_headers()
        for key, value in auth_headers.items():
            self.request_headers[key] = value
        if DEBUG:
            print(f"[AIM_AUTH] Connect: headers set = {list(self.request_headers.keys())}")
    except ValueError as e:
        raise RuntimeError(f"AIM Auth: {e}")

    try:
        return _original_connect(self)
    except requests.exceptions.JSONDecodeError as e:
        print(f"[AIM_AUTH] ERROR: Server returned non-JSON response during connect")
        print(f"[AIM_AUTH] This usually means authentication failed or server is unreachable")
        print(f"[AIM_AUTH] Check CF_ACCESS_CLIENT_ID and CF_ACCESS_CLIENT_SECRET are set correctly")
        raise
    except Exception as e:
        if DEBUG:
            print(f"[AIM_AUTH] Connect failed: {type(e).__name__}: {e}")
        raise


def _patched_refresh_ws(self):
    """Patched refresh_ws that includes authentication headers for WebSocket reconnection"""
    from websockets.sync.client import connect as ws_connect

    if not hasattr(self, "request_headers") or not self.request_headers:
        self.request_headers = {}

    try:
        auth_headers = _get_auth_headers()
        for key, value in auth_headers.items():
            self.request_headers[key] = value
    except ValueError as e:
        raise RuntimeError(f"AIM Auth: {e}")

    ws_url = f"{self._ws_protocol}{self._tracking_endpoint}/{self.uri}/write-instruction/"

    if DEBUG:
        print(f"[AIM_AUTH] Refreshing WebSocket: {ws_url}")
        print(f"[AIM_AUTH] WebSocket headers: {list(self.request_headers.keys())}")

    self._ws = ws_connect(
        ws_url,
        additional_headers=self.request_headers,
        max_size=None,
        ssl_context=self.ssl_context,
    )


# Apply the monkey patches
Client.__init__ = _patched_client_init
Client.protocol_probe = _patched_protocol_probe
Client.connect = _patched_connect
Client.refresh_ws = _patched_refresh_ws

# Re-export Run so users can import from this module
Run = OriginalRun

# Export everything AIM normally exports
__all__ = ["Run", "Client"]
