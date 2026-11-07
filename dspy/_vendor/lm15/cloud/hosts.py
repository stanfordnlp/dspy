"""
lm15.cloud.hosts — a dialect reaches a cloud door through a host (AUTH-10).

Three pure functions, in the order an adapter calls them:

1. ``resolve_settings(host, given, env)`` — the host's settings from the
   caller's values, then the environment (router/doctor only; a bare
   adapter is given its settings explicitly, like its credential), then
   defaults.  A required setting with no value raises ``NotConfiguredError``
   naming the variable: ``region`` and ``resource`` have no default on
   purpose (a wrong-region default is a residency bug).  With an
   ``endpoint`` (a full URL root) the settings the URL alone needed are
   not required (``HostSpec.url_only_settings``).
2. ``render_base_url(host, settings, endpoint=None)`` — the base URL for
   the settings; ``{location_host}`` is derived from ``location``.  An
   endpoint replaces the template's root; the door's path is appended
   unless the endpoint already ends with it or with a leading part of it
   (``join_endpoint``; spec/auth.md AUTH-10, amended 2026-09-19).
3. ``finish_request(...)`` — the dialect built its request against that
   base URL; this applies the host's closed set of rewrites (endpoint path
   override, model into the path, ``anthropic_version`` into the body,
   required headers, ``query-key``) and then signs (``sigv4``).  Ports
   implement exactly this function; the harness pins its output.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import re
from urllib.parse import quote, urlsplit
from typing import Any, Callable, Mapping

from ..credentials import ApiKey, AwsCredentials, CredentialValue
from ..errors import NotConfiguredError, UnsupportedFeatureError
from ..features import AccessPolicy, HostSpec
from . import sigv4

__all__ = ["resolve_settings", "render_base_url", "join_endpoint", "endpoint_from_env", "location_host",
           "finish_request", "Clock", "utc_now"]

Clock = Callable[[], datetime]


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def resolve_settings(
    host: HostSpec | None,
    given: Mapping[str, str] | None,
    env: Mapping[str, str] | None = None,
    *,
    provider: str = "",
    profile: Callable[[str], Any] | None = None,
    endpoint: str | None = None,
    sources: dict[str, str] | None = None,
    unprobed_ok: bool = False,
    problems: list[NotConfiguredError] | None = None,
) -> dict[str, str]:
    """Explicit values, then ``env`` (when given), then the cloud's own
    configuration (``profile(name)`` → ``(value, from)``: the AWS profile's
    ``region``; the Google project from the credential file, gcloud's
    active configuration, the ADC file, the metadata server — AUTH-10),
    then defaults.  With ``endpoint`` the settings only the URL root needed
    are optional.

    ``sources``, when given, receives each setting's origin in the AUTH-10
    ``from`` vocabulary (``explicit``, ``env:<VAR>``, ``adc-env``,
    ``gcloud-config``, ``adc-file``, ``metadata``, ``aws-profile``,
    ``default``).  ``unprobed_ok`` (the offline doctor): a setting only a
    network source could supply is left out and recorded as
    ``unprobed:<from>`` instead of raising.  ``problems``, when given
    (the doctor), receives the missing-setting errors instead of a raise,
    and the settings that did resolve are returned."""
    out: dict[str, str] = {}
    if host is None:
        return dict(given or {})
    given = dict(given or {})
    relaxed = host.url_only_settings if endpoint else frozenset()
    record = sources if sources is not None else {}
    missing: NotConfiguredError | None = None
    for setting in host.settings:
        value = given.pop(setting.name, None)
        origin = "explicit" if value else ""
        if not value and env is not None:
            for var in setting.env:
                candidate = env.get(var)
                if candidate:
                    value, origin = candidate, f"env:{var}"
                    break
        unprobed = ""
        if not value and profile is not None:
            found = profile(setting.name)
            if isinstance(found, tuple):
                if found[0]:
                    value, origin = found[0], found[1]
                else:
                    unprobed = found[1]
            elif found:
                value, origin = found, "profile"
        if not value and setting.default:
            value, origin = setting.default, "default"
        if not value:
            if setting.name in relaxed:
                continue
            if unprobed and unprobed_ok:
                record[setting.name] = f"unprobed:{unprobed}"
                continue
            hint = f"set {' or '.join(setting.env)}" if setting.env else f"pass settings={{'{setting.name}': ...}}"
            if setting.name == "project":
                # The Google project also comes from gcloud and the credential
                # file; those were read and said nothing (AUTH-10).
                hint += ", run `gcloud config set project <id>`, or pass settings={'project': ...}"
            if setting.name in host.url_only_settings and host.endpoint_env:
                hint += f", or the endpoint: {' or '.join(host.endpoint_env)}"
            record[setting.name] = "missing"
            # Every setting is still resolved, so the doctor reports them all;
            # the first missing one is the error.
            missing = missing or NotConfiguredError(
                f"{provider or 'host'}: setting {setting.name!r} is required and has no default; {hint}",
                provider=provider or None,
                credential_hint=hint,
            )
            continue
        out[setting.name] = value
        record[setting.name] = origin
    unknown = sorted(given)
    if unknown:
        raise ValueError(f"{provider or 'host'}: unknown host setting(s) {unknown}; known: {list(host.setting_names)}")
    if missing is not None:
        if problems is None:
            raise missing
        problems.append(missing)
    return out


def location_host(location: str) -> str:
    """Vertex host for a location (vertex-locations.md:40-63, :91)."""
    if location == "global":
        return "aiplatform.googleapis.com"
    if location in ("us", "eu"):
        return f"aiplatform.{location}.rep.googleapis.com"
    return f"{location}-aiplatform.googleapis.com"


def endpoint_from_env(host: HostSpec | None, env: Mapping[str, str] | None) -> str | None:
    """The first non-empty vendor endpoint variable this door honours."""
    if host is None or env is None:
        return None
    for var in host.endpoint_env:
        value = (env.get(var) or "").strip()
        if value:
            return value
    return None


def join_endpoint(endpoint: str, path: str, *, provider: str = "") -> str:
    """``endpoint`` (a URL root the caller or the vendor's variable named)
    joined with the door's ``path``.

    The door's path is appended unless the endpoint already ends with it,
    or with a leading part of it: the console shows an account root
    (``https://acct.services.ai.azure.com``), Microsoft's own examples
    show ``…/anthropic`` and ``…/openai/v1``, and all three must mean the
    same door.  Stated trade-off: a gateway whose own path happens to end
    with a leading part of the door's path (``…/openai`` meaning
    ``…/openai/openai/v1``) cannot be spelled; no such gateway is known.
    """
    parts = urlsplit(endpoint.strip())
    if parts.scheme not in ("http", "https") or not parts.netloc:
        raise NotConfiguredError(f"{provider or 'host'}: endpoint must be an http(s) URL with a host, got {endpoint!r}",
                                 provider=provider or None)
    if parts.query or parts.fragment or "@" in parts.netloc:
        raise NotConfiguredError(f"{provider or 'host'}: endpoint must not carry a query, fragment or userinfo",
                                 provider=provider or None)
    given = [segment for segment in parts.path.split("/") if segment]
    door = [segment for segment in path.split("/") if segment]
    base = given
    for k in range(min(len(given), len(door)), 0, -1):
        if given[-k:] == door[:k]:
            base = given[:-k]
            break
    joined = "/".join(base + door)
    return f"{parts.scheme}://{parts.netloc}" + (f"/{joined}" if joined else "")


def render_base_url(host: HostSpec, settings: Mapping[str, str], endpoint: str | None = None, *, provider: str = "") -> str:
    values = dict(settings)
    for name in ("region", "resource", "location"):
        if name in values and not re.fullmatch(r"[A-Za-z0-9-]+", values[name]):
            raise NotConfiguredError(f"host setting {name!r} must be a DNS label")
    if "project" in values:
        values["project"] = quote(values["project"], safe="")
    if "location" in values and "location_host" not in values:
        values["location_host"] = location_host(values["location"])
    template = host.base_url if endpoint is None else host.path_template
    try:
        rendered = template.format(**values)
    except KeyError as exc:
        raise NotConfiguredError(f"host base URL needs setting {exc.args[0]!r}") from None
    if endpoint is None:
        return rendered
    return join_endpoint(endpoint, rendered, provider=provider)


@dataclass(frozen=True, slots=True, repr=False)
class FinishedRequest:
    url: str
    headers: dict[str, str]
    payload: Any
    params: dict[str, str]


def finish_request(
    policy: AccessPolicy,
    settings: Mapping[str, str],
    *,
    base_url: str,
    url: str,
    headers: Mapping[str, str],
    payload: Any,
    params: Mapping[str, Any] | None,
    endpoint: str | None,
    stream: bool,
    model: str | None,
    credential: CredentialValue | None,
) -> FinishedRequest:
    """Apply the host's rewrites before serialization.  Signing happens
    after serialization in ``sign_request``."""
    host = policy.host
    out_headers = dict(headers)
    out_params = {str(k): str(v) for k, v in (params or {}).items()}
    if host is None:
        return FinishedRequest(url=url, headers=out_headers, payload=payload, params=out_params)

    if host.stream_framing != "sse" and stream:
        raise UnsupportedFeatureError(
            f"{policy.provider}: {host.stream_framing} stream framing is not implemented yet (phase 2)",
            provider=policy.provider,
        )

    key = f"{endpoint}/stream" if (endpoint and stream and f"{endpoint}/stream" in host.paths) else endpoint
    if key is not None and key in host.paths:
        if "{model}" in host.paths[key] and not model:
            raise ValueError(f"{policy.provider}: endpoint {endpoint!r} needs the model in the path")
        path_model = model or ""
        if endpoint == "generateContent":
            path_model = path_model.removeprefix("models/")
        url = base_url.rstrip("/") + host.paths[key].format(model=quote(path_model, safe=":@"))

    if isinstance(payload, dict):
        payload = dict(payload)
        if host.model_in == "path":
            payload.pop("model", None)
        if host.anthropic_version_in.startswith("body:"):
            payload["anthropic_version"] = host.anthropic_version_in[len("body:"):]
            out_headers = {k: v for k, v in out_headers.items() if k.lower() != "anthropic-version"}

    for name, setting in host.required_headers:
        value = settings.get(setting)
        if not value:
            raise NotConfiguredError(f"{policy.provider}: header {name} needs setting {setting!r}", provider=policy.provider)
        out_headers[name] = value

    if credential is not None and isinstance(credential, ApiKey) and "query-key" in policy.auth_scheme:
        from ..access import select_scheme

        if select_scheme(policy, credential) == "query-key":
            out_params["key"] = credential.value

    return FinishedRequest(url=url, headers=out_headers, payload=payload, params=out_params)


def sign_request(
    policy: AccessPolicy,
    settings: Mapping[str, str],
    *,
    method: str,
    url: str,
    headers: list[tuple[str, str]],
    body: bytes,
    credential: CredentialValue | None,
    now: datetime,
) -> list[tuple[str, str]]:
    """The headers to send.  ``sigv4`` replaces them with the signed set;
    every other scheme was already applied by the dialect's auth header."""
    host = policy.host
    if not isinstance(credential, AwsCredentials):
        return headers
    if host is None or host.sigv4_service is None:
        raise NotConfiguredError(f"{policy.provider}: AWS credentials need a sigv4 host", provider=policy.provider)
    region = settings.get("region")
    if not region:
        raise NotConfiguredError(f"{policy.provider}: sigv4 needs the region setting", provider=policy.provider)
    signature = sigv4.sign(
        method=method,
        url=url,
        headers={k: v for k, v in headers if k.lower() not in ("authorization", "x-api-key")},
        payload=body,
        credentials=credential,
        region=region,
        service=host.sigv4_service,
        now=now,
    )
    return list(signature.headers.items())
