"""SSL context factory.

We rely on the stdlib `ssl` module's `create_default_context`, which on
Python 3.10+ loads the system trust store correctly on Linux/macOS/Windows.
No certifi bundle is shipped — set SSL_CERT_FILE if your system store is
broken, or pass an explicit ca_bundle= to the transport.
"""
from __future__ import annotations

import ssl


def make_ssl_context(
    *, verify: bool = True, ca_bundle: str | None = None
) -> ssl.SSLContext:
    if not verify:
        ctx = ssl._create_unverified_context()
        return ctx
    ctx = ssl.create_default_context()
    if ca_bundle:
        ctx.load_verify_locations(cafile=ca_bundle)
    return ctx
