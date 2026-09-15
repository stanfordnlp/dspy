"""
lm15.cloud.hosts — a dialect reaches a cloud door through a host (AUTH-10).

Three pure functions, in the order an adapter calls them:

1. ``resolve_settings(host, given, env)`` — the host's settings from the
   caller's values, then the environment (router/doctor only; a bare
   adapter is given its settings explicitly, like its credential), then
   defaults.  A required setting with no value raises ``NotConfiguredError``
   naming the variable: ``region`` and ``resource`` have no default on
   purpose (a wrong-region default is a residency bug).
2. ``render_base_url(host, settings)`` — the base URL for the settings;
   ``{location_host}`` is derived from ``location``.
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
from urllib.parse import quote
from typing import Any, Callable, Mapping

from ..credentials import ApiKey, AwsCredentials, CredentialValue
from ..errors import NotConfiguredError, UnsupportedFeatureError
from ..features import AccessPolicy, HostSpec
from . import sigv4

__all__ = ["resolve_settings", "render_base_url", "location_host", "finish_request", "Clock", "utc_now"]

Clock = Callable[[], datetime]


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def resolve_settings(
    host: HostSpec | None,
    given: Mapping[str, str] | None,
    env: Mapping[str, str] | None = None,
    *,
    provider: str = "",
    profile: Callable[[str], str | None] | None = None,
) -> dict[str, str]:
    """Explicit values, then ``env`` (when given), then the cloud profile
    (``profile(name)``: the AWS shared config's ``region`` for the active
    profile, the ADC file's ``quota_project_id`` — AUTH-10), then defaults."""
    out: dict[str, str] = {}
    if host is None:
        return dict(given or {})
    given = dict(given or {})
    for setting in host.settings:
        value = given.pop(setting.name, None)
        if not value and env is not None:
            for var in setting.env:
                candidate = env.get(var)
                if candidate:
                    value = candidate
                    break
        if not value and profile is not None:
            value = profile(setting.name)
        if not value:
            value = setting.default
        if not value:
            hint = f"set {' or '.join(setting.env)}" if setting.env else f"pass settings={{'{setting.name}': ...}}"
            raise NotConfiguredError(
                f"{provider or 'host'}: setting {setting.name!r} is required and has no default; {hint}",
                provider=provider or None,
                credential_hint=hint,
            )
        out[setting.name] = value
    unknown = sorted(given)
    if unknown:
        raise ValueError(f"{provider or 'host'}: unknown host setting(s) {unknown}; known: {list(host.setting_names)}")
    return out


def location_host(location: str) -> str:
    """Vertex host for a location (vertex-locations.md:40-63, :91)."""
    if location == "global":
        return "aiplatform.googleapis.com"
    if location in ("us", "eu"):
        return f"aiplatform.{location}.rep.googleapis.com"
    return f"{location}-aiplatform.googleapis.com"


def render_base_url(host: HostSpec, settings: Mapping[str, str]) -> str:
    values = dict(settings)
    for name in ("region", "resource", "location"):
        if name in values and not re.fullmatch(r"[A-Za-z0-9-]+", values[name]):
            raise NotConfiguredError(f"host setting {name!r} must be a DNS label")
    if "project" in values:
        values["project"] = quote(values["project"], safe="")
    if "location" in values and "location_host" not in values:
        values["location_host"] = location_host(values["location"])
    try:
        return host.base_url.format(**values)
    except KeyError as exc:
        raise NotConfiguredError(f"host base URL needs setting {exc.args[0]!r}") from None


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
