"""
Minimal URL parser.

We only need to parse `http://` and `https://` URLs into the four bits a
transport actually uses: scheme, host, port, and request-target.  Full RFC 3986
parsing is unnecessary — provider URLs are well-formed.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ParsedURL:
    scheme: str       # "http" or "https"
    host: str         # hostname or IP literal (no brackets for IPv6)
    port: int         # always set (default 80 or 443)
    target: str       # request-target: "/path?query"
    is_tls: bool

    def origin(self) -> tuple[str, str, int]:
        """Pool key — two URLs with the same origin share connections."""
        return (self.scheme, self.host, self.port)

    def host_header(self) -> str:
        """The value to put in the Host: header."""
        is_default = (self.is_tls and self.port == 443) or (not self.is_tls and self.port == 80)
        if is_default:
            return _bracketed_if_v6(self.host)
        return f"{_bracketed_if_v6(self.host)}:{self.port}"


def _bracketed_if_v6(host: str) -> str:
    return f"[{host}]" if ":" in host else host


def parse_url(url: str) -> ParsedURL:
    """Parse an http(s) URL. Raises ValueError on anything unsupported."""
    # Scheme
    if "://" not in url:
        raise ValueError(f"URL missing scheme: {url!r}")
    scheme, _, rest = url.partition("://")
    scheme = scheme.lower()
    if scheme not in ("http", "https"):
        raise ValueError(f"unsupported scheme {scheme!r} in {url!r}")
    is_tls = scheme == "https"

    # Strip fragment (never sent to server)
    rest, _, _fragment = rest.partition("#")

    # Split authority from path
    slash = rest.find("/")
    qmark = rest.find("?")
    # First separator, if any
    cut = min((i for i in (slash, qmark) if i != -1), default=-1)
    if cut == -1:
        authority = rest
        target = "/"
    else:
        authority = rest[:cut]
        target = rest[cut:]
        # If the first char of target is '?', prepend '/'
        if target.startswith("?"):
            target = "/" + target

    if not authority:
        raise ValueError(f"URL missing host: {url!r}")

    host, port = _split_authority(authority, is_tls)

    return ParsedURL(
        scheme=scheme,
        host=host,
        port=port,
        target=target,
        is_tls=is_tls,
    )


def _split_authority(authority: str, is_tls: bool) -> tuple[str, int]:
    # Handle IPv6 literal: [::1]:8080  or  [::1]
    if authority.startswith("["):
        end = authority.find("]")
        if end == -1:
            raise ValueError(f"malformed IPv6 literal in authority: {authority!r}")
        host = authority[1:end]
        rest = authority[end + 1:]
        if rest.startswith(":"):
            port_s = rest[1:]
            if not port_s.isdigit():
                raise ValueError(f"bad port in authority: {authority!r}")
            return host, int(port_s)
        if rest == "":
            return host, 443 if is_tls else 80
        raise ValueError(f"unexpected trailing chars in authority: {authority!r}")

    # Hostname or IPv4
    if ":" in authority:
        host, _, port_s = authority.rpartition(":")
        if not port_s.isdigit():
            raise ValueError(f"bad port in authority: {authority!r}")
        return host, int(port_s)
    return authority, 443 if is_tls else 80
