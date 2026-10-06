"""
Proxy route resolution.

An explicit ``proxy=`` on the transport always wins; otherwise, with
``trust_env=True`` (the default), the standard environment variables
(``HTTP_PROXY`` / ``HTTPS_PROXY`` / ``ALL_PROXY``, minus ``NO_PROXY``)
are consulted per request — the platform contract every HTTP client
honors, and the mechanism egress brokers and corporate proxies rely on.
``trust_env=False`` makes the transport hermetic.

Only plain-HTTP proxies are supported (``http://`` proxy URLs). Plain
targets are forwarded with an absolute-URI request line; TLS targets are
tunneled with CONNECT and the TLS handshake runs end-to-end to the
origin — the proxy never sees inside the tunnel.
"""
from __future__ import annotations

import base64
from dataclasses import dataclass
from urllib.parse import unquote
from urllib.request import getproxies, proxy_bypass

from ._url import ParsedURL, parse_url


@dataclass(frozen=True, slots=True)
class ProxyRoute:
    """Where to connect, and what Proxy-Authorization to present (if any)."""

    host: str
    port: int
    basic_auth: str | None = None  # full header value, e.g. "Basic dXNlcjpwdw=="

    def authority(self, parsed: ParsedURL) -> str:
        """The target's host:port authority-form, as CONNECT wants it."""
        host = f"[{parsed.host}]" if ":" in parsed.host else parsed.host
        return f"{host}:{parsed.port}"


def parse_proxy_url(url: str) -> ProxyRoute:
    """Parse a proxy URL into a ProxyRoute. Raises ValueError on anything unsupported."""
    if "://" not in url:
        url = "http://" + url
    scheme, _, rest = url.partition("://")
    if scheme.lower() != "http":
        raise ValueError(
            f"unsupported proxy scheme {scheme!r} in {url!r}: lm15 speaks to plain-HTTP proxies only"
        )
    auth: str | None = None
    authority = rest.split("/", 1)[0]
    if "@" in authority:
        userinfo, _, hostpart = authority.rpartition("@")
        rest = hostpart + rest[len(authority):]
        auth = "Basic " + base64.b64encode(unquote(userinfo).encode("utf-8")).decode("ascii")
    parsed = parse_url("http://" + rest)
    return ProxyRoute(host=parsed.host, port=parsed.port, basic_auth=auth)


def proxy_route_for(parsed: ParsedURL, *, proxy: str | None, trust_env: bool) -> ProxyRoute | None:
    """Resolve the proxy for one target URL, or None for a direct connection."""
    if proxy is not None:
        return parse_proxy_url(proxy)
    if not trust_env:
        return None
    proxies = getproxies()
    url = proxies.get(parsed.scheme) or proxies.get("all")
    if not url:
        return None
    try:
        if proxy_bypass(parsed.host):  # honors NO_PROXY, including "*"
            return None
    except Exception:  # pragma: no cover — platform lookup quirks bypass nothing
        pass
    return parse_proxy_url(url)


def route_origin(parsed: ParsedURL, proxy: ProxyRoute | None) -> tuple[str, str, int]:
    """The connection-pool key for this (target, proxy) pair.

    Direct: the target origin, as before. Plain HTTP via proxy: one pool
    per proxy — the proxy multiplexes targets on a single connection.
    TLS via proxy: one pool per (proxy, target) — a CONNECT tunnel is
    welded to its target and cannot be reused for another host.
    """
    if proxy is None:
        return parsed.origin()
    if parsed.is_tls:
        return (f"https@{proxy.host}:{proxy.port}", parsed.host, parsed.port)
    return ("http-proxy", proxy.host, proxy.port)


def connect_payload(parsed: ParsedURL, proxy: ProxyRoute) -> bytes:
    """Serialize the CONNECT request that opens a tunnel to the target."""
    authority = proxy.authority(parsed)
    lines = [f"CONNECT {authority} HTTP/1.1", f"Host: {authority}"]
    if proxy.basic_auth is not None:
        lines.append(f"Proxy-Authorization: {proxy.basic_auth}")
    return ("\r\n".join(lines) + "\r\n\r\n").encode("ascii")
