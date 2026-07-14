"""Shared URL-safety helpers for media types that download from URLs.

`Image` and `Audio` can fetch media from a URL. Two things must be true for that
to be safe against SSRF and hanging-host DoS:

1. The fetch must be gated behind an explicit opt-in, so untrusted content
   (tool output, a retrieved document, a model output being cast to a media
   type) can never *trigger* an outbound request on its own. That gating lives
   in the media types themselves.

2. Even an opted-in fetch must validate its destination, because the URL itself
   may have come from untrusted input. `assert_public_url` rejects URLs that
   resolve to loopback, private, link-local, or otherwise non-public addresses
   (e.g. the cloud metadata endpoint ``169.254.169.254``).

These are complementary: (1) controls the trigger, (2) validates the target.
"""
from __future__ import annotations

import ipaddress
import socket
from urllib.parse import urlparse

DEFAULT_DOWNLOAD_TIMEOUT = 30.0


def assert_public_url(url: str) -> None:
    """Raise ``ValueError`` unless every address ``url``'s host resolves to is
    a public, routable IP address.

    Guards against the common SSRF targets: loopback (``127.0.0.1``,
    ``localhost``), private ranges (``10/8``, ``192.168/16``, ...), link-local
    (``169.254/16``, including the cloud metadata endpoint), and other reserved
    or non-global ranges.

    Note: this validates at resolution time and does not defend against DNS
    rebinding (a host that resolves to a public IP here but a private one when
    the request is actually made). That is a deeper mitigation than a media
    loader should own; this closes the direct SSRF surface.
    """
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError(
            f"Refusing to download from URL with scheme {parsed.scheme!r}; "
            "only http and https are allowed."
        )
    host = parsed.hostname
    if not host:
        raise ValueError(f"Could not parse a host from URL: {url!r}")

    try:
        addrinfos = socket.getaddrinfo(host, parsed.port or (443 if parsed.scheme == "https" else 80))
    except socket.gaierror as e:
        raise ValueError(f"Could not resolve host {host!r}: {e}") from e

    for info in addrinfos:
        ip = ipaddress.ip_address(info[4][0])
        if not ip.is_global or ip.is_multicast or ip.is_reserved:
            raise ValueError(
                f"Refusing to download from {url!r}: host {host!r} resolves to "
                f"non-public address {ip} (SSRF guard). If this is intentional "
                "(e.g. a trusted internal host), fetch the bytes yourself and "
                "pass them directly."
            )
