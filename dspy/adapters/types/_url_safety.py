"""Shared URL-safety helpers for media types that download from URLs.

`Image` and `Audio` can fetch media from a URL. Three things must be true for
that to be safe against SSRF and hanging-host DoS:

1. The fetch must be gated behind an explicit opt-in, so untrusted content
   (tool output, a retrieved document, a model output being cast to a media
   type) can never *trigger* an outbound request on its own. That gating lives
   in the media types themselves.

2. Every request the fetch actually makes must go to a public destination, at
   every hop. `safe_get` validates the initial URL *and every redirect target*
   with `assert_public_url` before connecting, so a public URL that redirects
   to an internal address (e.g. the cloud metadata endpoint) is refused.

3. Neither the DNS lookup nor the HTTP request may hang indefinitely. The HTTP
   request is bounded by `timeout`; name resolution is bounded separately (see
   `_resolve`), since `socket.getaddrinfo` does not honor the socket timeout on
   most platforms.

Known limitation: `assert_public_url` validates at resolution time and does not
defend against DNS rebinding (a host that resolves public here and private when
the socket actually connects). Closing that requires resolving once and
connecting to the pinned IP with the original host in the SNI/Host header, which
is a deeper change than a media loader should own.
"""
from __future__ import annotations

import ipaddress
import socket
import threading
from urllib.parse import urljoin, urlparse

import requests

DEFAULT_DOWNLOAD_TIMEOUT = 30.0
MAX_REDIRECTS = 5


def _resolve(host: str, port: int, timeout: float):
    """`socket.getaddrinfo` with a hard wall-clock timeout.

    `getaddrinfo` ignores `socket.setdefaulttimeout` on most platforms, so a
    host pointed at a dead DNS server can hang the caller indefinitely. Run it
    on a daemon thread and give up after `timeout` seconds; the caller never
    blocks longer than that even if the lookup thread is still stuck.
    """
    box: dict = {}

    def work():
        try:
            box["ok"] = socket.getaddrinfo(host, port)
        except Exception as e:  # noqa: BLE001 - surfaced below as ValueError
            box["err"] = e

    th = threading.Thread(target=work, daemon=True)
    th.start()
    th.join(timeout)
    if th.is_alive():
        raise ValueError(f"DNS resolution for {host!r} timed out after {timeout}s (SSRF check).")
    if "err" in box:
        raise ValueError(f"Could not resolve host {host!r}: {box['err']}")
    return box["ok"]


def assert_public_url(url: str, *, dns_timeout: float = DEFAULT_DOWNLOAD_TIMEOUT) -> None:
    """Raise ``ValueError`` unless every address ``url``'s host resolves to is
    a public, routable IP address.

    Guards against the common SSRF targets: loopback (``127.0.0.1``,
    ``localhost``), private ranges (``10/8``, ``192.168/16``, ...), link-local
    (``169.254/16``, including the cloud metadata endpoint), and other reserved
    or non-global ranges. Name resolution is bounded by ``dns_timeout``.
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

    addrinfos = _resolve(host, parsed.port or (443 if parsed.scheme == "https" else 80), dns_timeout)
    for info in addrinfos:
        ip = ipaddress.ip_address(info[4][0])
        if not ip.is_global or ip.is_multicast or ip.is_reserved:
            raise ValueError(
                f"Refusing to download from {url!r}: host {host!r} resolves to "
                f"non-public address {ip} (SSRF guard). If this is intentional "
                "(e.g. a trusted internal host), fetch the bytes yourself and "
                "pass them directly."
            )


def safe_get(
    url: str,
    *,
    verify: bool = True,
    timeout: float = DEFAULT_DOWNLOAD_TIMEOUT,
    max_redirects: int = MAX_REDIRECTS,
) -> requests.Response:
    """GET ``url``, validating the destination at every hop.

    Redirects are followed manually so that each ``Location`` is checked with
    ``assert_public_url`` before it is fetched. This closes the redirect-based
    SSRF bypass: a public URL that 3xx-redirects to an internal address is
    refused instead of followed. Returns the final (non-redirect) response;
    the caller is responsible for ``raise_for_status()``.
    """
    current = url
    for _ in range(max_redirects + 1):
        assert_public_url(current, dns_timeout=timeout)
        resp = requests.get(current, verify=verify, timeout=timeout, allow_redirects=False)
        if resp.is_redirect or resp.is_permanent_redirect:
            location = resp.headers.get("Location")
            if not location:
                return resp
            current = urljoin(resp.url, location)
            continue
        return resp
    raise ValueError(f"Too many redirects (> {max_redirects}) while downloading {url!r}.")
