"""
lm15.cloud.chains — the three cloud credential chains as data over ten rung kinds.

spec/auth.md AUTH-1 (``aws-chain``, ``azure-chain``, ``gcp-chain``) and
AUTH-11 (rung kinds).  Every order, variable, path and endpoint here is
cited in lm15-contract/research/cloud-hosts/10-facts-*.md from the cloud
SDKs' own resolver sources.

Two entry points:

- ``explain(policy, ctx)`` — the offline doctor walk (AUTH-7): every rung
  reports ``selected`` / ``shadowed`` / ``absent`` / ``unprobed``.  No
  network, no subprocess: a rung that needs either is ``unprobed`` when
  its configuration is present.  The first offline-usable rung is
  ``selected``; an ``unprobed`` rung that precedes it may still win at
  request time, and the report says so.
- ``credential_provider(policy, ctx)`` — the AUTH-2 provider callable:
  resolves once, caches until the AUTH-3 skew window, re-resolves after.
  Cache key = provider id + the identity-selecting settings.

Rungs that are declared but not implemented raise ``NotConfiguredError``
naming the gap and the fix; they never fall through silently
(``aws login`` refresh with a DPoP key; Azure Service Fabric managed
identity; GCP ``external_account`` with an AWS source).
"""

from __future__ import annotations

import base64
import configparser
import hashlib
import http.client
import json
import os
import subprocess
import threading
import urllib.parse
import urllib.request
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Mapping
from xml.etree import ElementTree

from ..credentials import ApiKey, AwsCredentials, BearerToken, CredentialValue, parse_rfc3339
from ..errors import AuthError, NotConfiguredError
from ..features import RUNG_KINDS, AccessPolicy
from . import rs256, sigv4

__all__ = ["ChainContext", "Rung", "Step", "chain_for", "explain", "credential_provider",
           "token_exchange_build", "token_exchange_parse", "RUNG_KINDS"]


_SKEW = timedelta(seconds=300)  # AUTH-3
_GCP_SCOPE = "https://www.googleapis.com/auth/cloud-platform"
_GCP_TOKEN_URL = "https://oauth2.googleapis.com/token"
_GCP_STS_URL = "https://sts.googleapis.com/v1/token"
_JWT_BEARER = "urn:ietf:params:oauth:grant-type:jwt-bearer"
_CLIENT_ASSERTION_TYPE = "urn:ietf:params:oauth:client-assertion-type:jwt-bearer"


# ─── Context: everything a chain touches, injectable ─────────────────

HttpFn = Callable[[str, str, Mapping[str, str], bytes | None, float], tuple[int, dict[str, str], bytes]]


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        # Never forward credential headers or token bodies to another URL.
        return None


def _default_http(method: str, url: str, headers: Mapping[str, str], body: bytes | None, timeout: float):
    try:
        req = urllib.request.Request(url, data=body, method=method, headers=dict(headers))
        with urllib.request.build_opener(_NoRedirect()).open(req, timeout=timeout) as r:
            return r.status, {k.lower(): v for k, v in r.headers.items()}, r.read()
    except urllib.error.HTTPError as e:
        with e:
            return e.code, {k.lower(): v for k, v in e.headers.items()}, e.read()
    except (OSError, ValueError, http.client.HTTPException):
        raise AuthError("credential HTTP request failed") from None


def _default_run(argv: list[str], timeout: float, *, env: Mapping[str, str] | None = None) -> str:
    try:
        out = subprocess.run(argv, capture_output=True, text=True, timeout=timeout, check=False, env=env)
    except (OSError, subprocess.SubprocessError):
        raise AuthError("credential command failed") from None
    if out.returncode != 0:
        raise AuthError(f"credential command exited {out.returncode}")
    return out.stdout


@dataclass(repr=False)
class ChainContext:
    """What a chain reads.  ``http`` / ``run`` = None means offline (the
    doctor).  ``files`` overrides the filesystem for the harness."""

    env: Mapping[str, str]
    home: Path = field(default_factory=Path.home)
    files: Mapping[str, str] | None = None
    http: HttpFn | None = None
    run: Callable[[list[str], float], str] | None = None
    now: Callable[[], datetime] = lambda: datetime.now(timezone.utc)
    settings: Mapping[str, str] = field(default_factory=dict)

    @classmethod
    def online(cls, env: Mapping[str, str] | None = None, **kw: Any) -> "ChainContext":
        values = dict(env if env is not None else os.environ)
        if "home" not in kw:
            kw["home"] = Path(values["HOME"]) if values.get("HOME") else Path.home()
        return cls(env=values, http=_default_http,
                   run=lambda argv, timeout: _default_run(argv, timeout, env=values), **kw)

    def path(self, text: str) -> Path:
        if text.startswith("~"):
            return self.home / text[1:].lstrip("/\\")
        return Path(text)

    def read(self, text: str) -> str | None:
        if self.files is not None:
            for key, content in self.files.items():
                if str(self.path(key)) == str(self.path(text)):
                    return content
            return None
        try:
            return self.path(text).read_text(encoding="utf-8")
        except OSError:
            return None

    def exists(self, text: str) -> bool:
        return self.read(text) is not None

    def on_path(self, command: str) -> str | None:
        """Where ``command`` would run from, from the context's PATH only —
        an offline file check, so the doctor can say "not installed"."""
        if "/" in command or "\\" in command:
            return command if self.exists(command) else None
        path = self.env.get("PATH") or ""
        if self.files is not None:
            for directory in path.split(os.pathsep):
                if directory and self.exists(f"{directory.rstrip('/')}/{command}"):
                    return f"{directory.rstrip('/')}/{command}"
            return None
        import shutil

        return shutil.which(command, path=path) if path else None

    @property
    def offline(self) -> bool:
        return self.http is None


# ─── Rungs and steps ─────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class Rung:
    name: str        # the fixture kind: "env:AWS_REGION", "assume-role", "imds", …
    kind: str        # RUNG_KINDS
    source: str      # human label
    needs: str       # "" (offline-usable) | "network" | "subprocess"
    probe: Callable[["ChainContext"], tuple[str, str]]          # → ("usable"|"configured"|"absent", detail)
    acquire: Callable[["ChainContext"], CredentialValue | None]  # online resolution

    def __post_init__(self) -> None:
        if self.kind not in RUNG_KINDS:
            raise ValueError(f"unknown rung kind {self.kind!r}")


@dataclass(frozen=True, slots=True)
class Step:
    kind: str
    source: str
    detail: str
    state: str  # selected | shadowed | absent | unprobed


# ─── Helpers ─────────────────────────────────────────────────────────


def _expires_from(now: datetime, seconds: Any) -> datetime | None:
    try:
        return now + timedelta(seconds=int(float(seconds)))
    except (TypeError, ValueError, OverflowError):
        return None


def _json_body(body: bytes) -> dict:
    try:
        data = json.loads(body.decode("utf-8", "replace"))
    except ValueError:
        return {}
    return data if isinstance(data, dict) else {}


def _form(pairs: list[tuple[str, str]]) -> bytes:
    return urllib.parse.urlencode(pairs).encode("ascii")


def _exchange(ctx: ChainContext, method: str, url: str, headers: Mapping[str, str], body: bytes | None, what: str) -> dict:
    assert ctx.http is not None
    status, _, raw = ctx.http(method, url, headers, body, 30.0)
    data = _json_body(raw)
    if not 200 <= status < 300:
        raise AuthError(f"{what}: HTTP {status}")
    return data


def _bearer_from_oauth(data: dict, now: datetime, what: str) -> BearerToken:
    token = data.get("access_token")
    if not isinstance(token, str) or not token:
        raise AuthError(f"{what}: no valid access_token in response")
    expires = None
    if data.get("expires_on") not in (None, ""):
        try:
            expires = datetime.fromtimestamp(int(float(data["expires_on"])), tz=timezone.utc)
        except (TypeError, ValueError, OverflowError, OSError):
            expires = None
    if expires is None and data.get("expires_in") not in (None, ""):
        expires = _expires_from(now, data["expires_in"])
    return BearerToken(token, expires_at=expires)


# ─── AWS ─────────────────────────────────────────────────────────────


def _aws_config(ctx: ChainContext) -> tuple[configparser.ConfigParser, configparser.ConfigParser, str]:
    profile = ctx.env.get("AWS_PROFILE") or "default"
    creds = configparser.ConfigParser(interpolation=None)
    conf = configparser.ConfigParser(interpolation=None)
    text = ctx.read(ctx.env.get("AWS_SHARED_CREDENTIALS_FILE") or "~/.aws/credentials")
    try:
        if text:
            creds.read_string(text)
        text = ctx.read(ctx.env.get("AWS_CONFIG_FILE") or "~/.aws/config")
        if text:
            conf.read_string(text)
    except configparser.Error:
        raise NotConfiguredError("malformed AWS profile configuration; check the AWS config and credentials files") from None
    return creds, conf, profile


def _aws_profile_section(conf: configparser.ConfigParser, profile: str) -> Mapping[str, str]:
    name = profile if profile == "default" else f"profile {profile}"
    if conf.has_section(name):
        return conf[name]
    if conf.has_section(profile):
        return conf[profile]
    return {}


def _aws_static(section: Mapping[str, str], expiry: str | None = None) -> AwsCredentials | None:
    key, secret = section.get("aws_access_key_id"), section.get("aws_secret_access_key")
    if not key or not secret:
        return None
    return AwsCredentials(key, secret, session_token=section.get("aws_session_token") or None)


def _aws_from_response(d: Mapping[str, Any]) -> AwsCredentials:
    expires = None
    raw = d.get("Expiration") or d.get("expiration")
    if isinstance(raw, str):
        expires = parse_rfc3339(raw)
    elif isinstance(raw, (int, float)):
        expires = datetime.fromtimestamp(raw / (1000 if raw > 10**11 else 1), tz=timezone.utc)
    key = d.get("AccessKeyId") or d.get("accessKeyId")
    secret = d.get("SecretAccessKey") or d.get("secretAccessKey")
    if not isinstance(key, str) or not key or not isinstance(secret, str) or not secret:
        raise AuthError("AWS credential response lacks access key id or secret access key")
    return AwsCredentials(
        key,
        secret,
        session_token=(d.get("SessionToken") or d.get("Token") or d.get("sessionToken")) or None,
        expires_at=expires,
    )


def _sts_xml_credentials(raw: bytes) -> AwsCredentials:
    root = ElementTree.fromstring(raw)
    ns = {"sts": "https://sts.amazonaws.com/doc/2011-06-15/"}
    node = root.find(".//sts:Credentials", ns)
    if node is None:
        raise AuthError("STS: no Credentials in response")
    get = lambda tag: (node.findtext(f"sts:{tag}", namespaces=ns) or "")  # noqa: E731
    return AwsCredentials(get("AccessKeyId"), get("SecretAccessKey"), session_token=get("SessionToken") or None,
                          expires_at=parse_rfc3339(get("Expiration")) if get("Expiration") else None)


def _aws_source_credentials(ctx: ChainContext, section: Mapping[str, str], depth: int = 0) -> AwsCredentials:
    """Source credentials for assume-role: source_profile (recursive) or credential_source."""
    if depth > 5:
        raise AuthError("assume-role: source_profile chain too deep")
    source_profile = section.get("source_profile")
    if source_profile:
        creds, conf, _ = _aws_config(ctx)
        sub = dict(_aws_profile_section(conf, source_profile))
        if creds.has_section(source_profile):
            sub.update(creds[source_profile])
        if sub.get("role_arn"):
            return _assume_role(ctx, sub, depth + 1)
        static = _aws_static(sub)
        if static is None:
            raise NotConfiguredError(f"assume-role: source_profile {source_profile!r} has no keys")
        return static
    source = section.get("credential_source")
    if source == "Environment":
        static = _env_aws(ctx)
        if static is None:
            raise NotConfiguredError("assume-role: credential_source=Environment but AWS_ACCESS_KEY_ID is not set")
        return static
    if source == "EcsContainer":
        got = _container_acquire(ctx)
        if got is None:
            raise NotConfiguredError("assume-role: credential_source=EcsContainer but no container endpoint is configured")
        return got
    if source == "Ec2InstanceMetadata":
        got = _imds_acquire(ctx)
        if got is None:
            raise NotConfiguredError("assume-role: credential_source=Ec2InstanceMetadata but IMDS answered nothing")
        return got
    raise NotConfiguredError("assume-role: profile needs source_profile or credential_source")


def _assume_role(ctx: ChainContext, section: Mapping[str, str], depth: int = 0) -> AwsCredentials:
    source = _aws_source_credentials(ctx, section, depth)
    region = ctx.settings.get("region") or ctx.env.get("AWS_REGION") or ctx.env.get("AWS_DEFAULT_REGION") or section.get("region") or "us-east-1"
    url = f"https://sts.{region}.amazonaws.com/"
    pairs = [("Action", "AssumeRole"), ("Version", "2011-06-15"), ("RoleArn", section["role_arn"]),
             ("RoleSessionName", section.get("role_session_name") or f"lm15-{uuid.uuid4().hex[:12]}")]
    if section.get("external_id"):
        pairs.append(("ExternalId", section["external_id"]))
    if section.get("duration_seconds"):
        pairs.append(("DurationSeconds", section["duration_seconds"]))
    body = _form(pairs)
    headers = {"content-type": "application/x-www-form-urlencoded"}
    signed = sigv4.sign(method="POST", url=url, headers=headers, payload=body, credentials=source, region=region, service="sts", now=ctx.now())
    assert ctx.http is not None
    status, _, raw = ctx.http("POST", url, signed.headers, body, 30.0)
    if status >= 400:
        raise AuthError(f"STS AssumeRole: HTTP {status}")
    return _sts_xml_credentials(raw)


def _env_aws(ctx: ChainContext) -> AwsCredentials | None:
    key, secret = ctx.env.get("AWS_ACCESS_KEY_ID"), ctx.env.get("AWS_SECRET_ACCESS_KEY")
    if key and secret:
        return AwsCredentials(key, secret, session_token=ctx.env.get("AWS_SESSION_TOKEN") or None)
    return None


def _web_identity_config(ctx: ChainContext) -> tuple[str, str, str] | None:
    token_file, role = ctx.env.get("AWS_WEB_IDENTITY_TOKEN_FILE"), ctx.env.get("AWS_ROLE_ARN")
    session = ctx.env.get("AWS_ROLE_SESSION_NAME") or ""
    if not (token_file and role):
        _, conf, profile = _aws_config(ctx)
        section = _aws_profile_section(conf, profile)
        token_file, role = section.get("web_identity_token_file"), section.get("role_arn")
        session = section.get("role_session_name") or ""
        if not (token_file and role and not section.get("source_profile") and not section.get("credential_source")):
            return None
    return token_file, role, session


def _web_identity_acquire(ctx: ChainContext) -> AwsCredentials | None:
    cfg = _web_identity_config(ctx)
    if cfg is None:
        return None
    token_file, role, session = cfg
    token = ctx.read(token_file)
    if token is None:
        raise NotConfiguredError(f"web identity token file {token_file} is unreadable")
    region = ctx.settings.get("region") or ctx.env.get("AWS_REGION") or ctx.env.get("AWS_DEFAULT_REGION") or "us-east-1"
    body = _form([("Action", "AssumeRoleWithWebIdentity"), ("Version", "2011-06-15"), ("RoleArn", role),
                  ("RoleSessionName", session or f"lm15-{uuid.uuid4().hex[:12]}"), ("WebIdentityToken", token.strip())])
    assert ctx.http is not None
    status, _, raw = ctx.http("POST", f"https://sts.{region}.amazonaws.com/", {"content-type": "application/x-www-form-urlencoded"}, body, 30.0)
    if status >= 400:
        raise AuthError(f"STS AssumeRoleWithWebIdentity: HTTP {status}")
    return _sts_xml_credentials(raw)


def _sso_config(ctx: ChainContext) -> dict[str, str] | None:
    _, conf, profile = _aws_config(ctx)
    section = dict(_aws_profile_section(conf, profile))
    if section.get("sso_session"):
        name = section["sso_session"]
        sess = dict(conf[f"sso-session {name}"]) if conf.has_section(f"sso-session {name}") else {}
        if not sess.get("sso_start_url"):
            return None
        cache_key = hashlib.sha1(name.encode("utf-8")).hexdigest()
        return {**sess, **section, "cache_key": cache_key, "session_name": name}
    if section.get("sso_start_url"):
        cache_key = hashlib.sha1(section["sso_start_url"].encode("utf-8")).hexdigest()
        return {**section, "cache_key": cache_key}
    return None


def _sso_acquire(ctx: ChainContext) -> AwsCredentials | None:
    cfg = _sso_config(ctx)
    if cfg is None:
        return None
    raw = ctx.read(f"~/.aws/sso/cache/{cfg['cache_key']}.json")
    if raw is None:
        raise NotConfiguredError("IAM Identity Center: no cached token; run `aws sso login`", credential_hint="aws sso login")
    token = json.loads(raw)
    now = ctx.now()
    expires = parse_rfc3339(token["expiresAt"]) if token.get("expiresAt") else None
    access = token.get("accessToken")
    sso_region = cfg.get("sso_region") or "us-east-1"
    assert ctx.http is not None
    if not access or (expires is not None and expires - now <= _SKEW):
        if not (token.get("refreshToken") and token.get("clientId") and token.get("clientSecret")):
            raise NotConfiguredError("IAM Identity Center: token expired and not refreshable; run `aws sso login`", credential_hint="aws sso login")
        data = _exchange(ctx, "POST", f"https://oidc.{sso_region}.amazonaws.com/token", {"content-type": "application/json"},
                         json.dumps({"clientId": token["clientId"], "clientSecret": token["clientSecret"],
                                     "grantType": "refresh_token", "refreshToken": token["refreshToken"]}).encode(), "sso-oidc CreateToken")
        access = data.get("accessToken")
        if not access:
            raise AuthError("sso-oidc CreateToken: no accessToken")
    account, role = cfg.get("sso_account_id"), cfg.get("sso_role_name")
    if not (account and role):
        raise NotConfiguredError("IAM Identity Center: profile needs sso_account_id and sso_role_name")
    query = urllib.parse.urlencode({"role_name": role, "account_id": account})
    status, _, raw_creds = ctx.http("GET", f"https://portal.sso.{sso_region}.amazonaws.com/federation/credentials?{query}",
                                    {"x-amz-sso_bearer_token": str(access)}, None, 30.0)
    if status >= 400:
        raise AuthError(f"sso GetRoleCredentials: HTTP {status}")
    creds = _json_body(raw_creds).get("roleCredentials") or {}
    return _aws_from_response(creds)


def _login_config(ctx: ChainContext) -> str | None:
    _, conf, profile = _aws_config(ctx)
    return _aws_profile_section(conf, profile).get("login_session") or None


def _login_cached(ctx: ChainContext) -> AwsCredentials | None:
    session = _login_config(ctx)
    if not session:
        return None
    directory = ctx.env.get("AWS_LOGIN_CACHE_DIRECTORY") or "~/.aws/login/cache"
    raw = ctx.read(f"{directory}/{hashlib.sha256(session.encode('utf-8')).hexdigest()}.json")
    if raw is None:
        return None
    token = json.loads(raw).get("accessToken") or {}
    if not token.get("accessKeyId"):
        return None
    return _aws_from_response({"AccessKeyId": token["accessKeyId"], "SecretAccessKey": token.get("secretAccessKey"),
                               "SessionToken": token.get("sessionToken"), "Expiration": token.get("expiresAt")})


def _login_acquire(ctx: ChainContext) -> AwsCredentials | None:
    if _login_config(ctx) is None:
        return None
    cached = _login_cached(ctx)
    if cached is not None and not cached.is_expired(ctx.now()):
        return cached
    # Refresh needs the signin CreateOAuth2Token call with a DPoP proof over
    # the cached EC key (botocore credentials.py LoginCredentialFetcher):
    # a stated gap; the hint names the command.
    raise NotConfiguredError("AWS login session expired; run `aws login`", credential_hint="aws login")


def _process_acquire(ctx: ChainContext) -> AwsCredentials | None:
    _, conf, profile = _aws_config(ctx)
    command = _aws_profile_section(conf, profile).get("credential_process")
    if not command:
        return None
    if ctx.run is None:
        return None
    import shlex

    out = ctx.run(shlex.split(command), 60.0)
    data = _json_body(out.encode())
    if data.get("Version") != 1:
        raise AuthError("credential_process: output Version must be 1")
    return _aws_from_response(data)


_CONTAINER_ALLOWED = {"169.254.170.2", "169.254.170.23", "fd00:ec2::23", "localhost"}


def _container_config(ctx: ChainContext) -> str | None:
    rel, full = ctx.env.get("AWS_CONTAINER_CREDENTIALS_RELATIVE_URI"), ctx.env.get("AWS_CONTAINER_CREDENTIALS_FULL_URI")
    if rel:
        if not rel.startswith("/") or rel.startswith("//") or any(c in rel for c in "\\\r\n\t#"):
            raise NotConfiguredError("container credentials relative URI must be an absolute path")
        return f"http://169.254.170.2{rel}"
    if full:
        import ipaddress

        parsed = urllib.parse.urlsplit(full)
        host = parsed.hostname or ""
        if parsed.scheme not in ("http", "https") or not host or parsed.username is not None or parsed.fragment:
            raise NotConfiguredError("container credentials URI must be HTTP(S), without userinfo or fragment")
        loopback = False
        try:
            loopback = ipaddress.ip_address(host).is_loopback
        except ValueError:
            pass
        if parsed.scheme != "https" and not loopback and host not in _CONTAINER_ALLOWED:
            raise NotConfiguredError(
                f"Unsupported host {host!r}. Can only retrieve metadata from a loopback address or one of these hosts: "
                + ", ".join(sorted(_CONTAINER_ALLOWED))
            )
        return full
    return None


def _container_acquire(ctx: ChainContext) -> AwsCredentials | None:
    url = _container_config(ctx)
    if url is None or ctx.http is None:
        return None
    headers: dict[str, str] = {}
    token = ctx.env.get("AWS_CONTAINER_AUTHORIZATION_TOKEN")
    token_file = ctx.env.get("AWS_CONTAINER_AUTHORIZATION_TOKEN_FILE")
    if token_file and not token:
        token = (ctx.read(token_file) or "").strip()
    if token:
        headers["Authorization"] = token
    status, _, raw = ctx.http("GET", url, headers, None, 5.0)
    if status >= 400:
        raise AuthError(f"container credentials: HTTP {status}")
    return _aws_from_response(_json_body(raw))


def _imds_disabled(ctx: ChainContext) -> bool:
    return (ctx.env.get("AWS_EC2_METADATA_DISABLED") or "").strip().lower() == "true"


def _imds_acquire(ctx: ChainContext) -> AwsCredentials | None:
    if _imds_disabled(ctx) or ctx.http is None:
        return None
    base = ctx.env.get("AWS_EC2_METADATA_SERVICE_ENDPOINT") or (
        "http://[fd00:ec2::254]" if (ctx.env.get("AWS_EC2_METADATA_SERVICE_ENDPOINT_MODE") or "").lower() == "ipv6" else "http://169.254.169.254"
    )
    base = base.rstrip("/")
    try:
        status, _, tok = ctx.http("PUT", f"{base}/latest/api/token", {"X-aws-ec2-metadata-token-ttl-seconds": "21600"}, None, 1.0)
    except Exception:  # noqa: BLE001 - not on EC2: the rung is absent, not an error
        return None
    if status != 200:
        return None
    headers = {"X-aws-ec2-metadata-token": tok.decode("utf-8", "replace")}
    status, _, role = ctx.http("GET", f"{base}/latest/meta-data/iam/security-credentials/", headers, None, 1.0)
    if status != 200 or not role.strip():
        return None
    status, _, raw = ctx.http("GET", f"{base}/latest/meta-data/iam/security-credentials/{role.decode().strip().splitlines()[0]}", headers, None, 1.0)
    if status != 200:
        return None
    data = _json_body(raw)
    if data.get("Code") not in (None, "Success"):
        raise AuthError("IMDS rejected the credential request")
    return _aws_from_response(data)


def _aws_chain(policy: AccessPolicy) -> list[Rung]:
    door_key = policy.env_keys[0] if policy.env_keys else None

    def env_probe(var: str):
        return lambda ctx: ("usable", "set (value never shown)") if ctx.env.get(var) else ("absent", "not set")

    def static_keys_probe(ctx: ChainContext):
        return ("usable", "set (values never shown)") if _env_aws(ctx) else ("absent", "AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY not set")

    def assume_role_probe(ctx: ChainContext):
        _, conf, profile = _aws_config(ctx)
        section = _aws_profile_section(conf, profile)
        if section.get("role_arn") and (section.get("source_profile") or section.get("credential_source")):
            return ("configured", f"profile {profile!r} assumes {section['role_arn']} (STS call at request time)")
        return ("absent", f"profile {profile!r} has no role_arn with a source")

    def assume_role_acquire(ctx: ChainContext):
        _, conf, profile = _aws_config(ctx)
        section = _aws_profile_section(conf, profile)
        if section.get("role_arn") and (section.get("source_profile") or section.get("credential_source")):
            return _assume_role(ctx, section)
        return None

    def web_identity_probe(ctx: ChainContext):
        cfg = _web_identity_config(ctx)
        return ("configured", f"token file {cfg[0]} → {cfg[1]} (STS call at request time)") if cfg else ("absent", "AWS_WEB_IDENTITY_TOKEN_FILE / AWS_ROLE_ARN not set")

    def sso_probe(ctx: ChainContext):
        cfg = _sso_config(ctx)
        if cfg is None:
            return ("absent", "no sso_session / sso_start_url in the profile")
        cached = ctx.exists(f"~/.aws/sso/cache/{cfg['cache_key']}.json")
        return ("configured", "cached token present (GetRoleCredentials at request time)" if cached else "no cached token; run `aws sso login`")

    def shared_file_probe(ctx: ChainContext):
        creds, _, profile = _aws_config(ctx)
        section = creds[profile] if creds.has_section(profile) else {}
        return ("usable", f"profile {profile!r} (values never shown)") if _aws_static(section) else ("absent", f"no keys for profile {profile!r}")

    def shared_file_acquire(ctx: ChainContext):
        creds, _, profile = _aws_config(ctx)
        return _aws_static(creds[profile]) if creds.has_section(profile) else None

    def login_probe(ctx: ChainContext):
        if _login_config(ctx) is None:
            return ("absent", "no login_session in the profile")
        cached = _login_cached(ctx)
        if cached is not None and not cached.is_expired(ctx.now()):
            return ("usable", "cached short-term credentials are fresh")
        return ("configured", "cached credentials missing or expired; refresh needs `aws login`")

    def process_probe(ctx: ChainContext):
        _, conf, profile = _aws_config(ctx)
        cmd = _aws_profile_section(conf, profile).get("credential_process")
        if not cmd:
            return ("absent", "no credential_process in the profile")
        import shlex

        executable = shlex.split(cmd)[0]
        if ctx.on_path(executable) is None:
            return ("absent", f"credential_process {executable!r} is not on PATH")
        return ("configured", "credential_process configured (run at request time)")

    def config_file_probe(ctx: ChainContext):
        _, conf, profile = _aws_config(ctx)
        return ("usable", f"static keys in config for profile {profile!r}") if _aws_static(_aws_profile_section(conf, profile)) else ("absent", "no static keys in config")

    def config_file_acquire(ctx: ChainContext):
        _, conf, profile = _aws_config(ctx)
        return _aws_static(_aws_profile_section(conf, profile))

    def container_probe(ctx: ChainContext):
        try:
            url = _container_config(ctx)
        except NotConfiguredError as exc:
            return ("absent", str(exc).splitlines()[0])
        return ("configured", "container endpoint configured (HTTP at request time)") if url else ("absent", "no AWS_CONTAINER_CREDENTIALS_* URI")

    def imds_probe(ctx: ChainContext):
        return ("absent", "AWS_EC2_METADATA_DISABLED=true") if _imds_disabled(ctx) else ("configured", "instance metadata probed at request time")

    rungs: list[Rung] = []
    if door_key:
        kind_cred = BearerToken if door_key == "AWS_BEARER_TOKEN_BEDROCK" else ApiKey
        rungs.append(Rung(f"env:{door_key}", "env", f"env ${door_key}", "", env_probe(door_key),
                          lambda ctx, k=door_key, c=kind_cred: c(ctx.env[k]) if ctx.env.get(k) else None))
    rungs += [
        Rung("env:AWS_ACCESS_KEY_ID", "env", "env $AWS_ACCESS_KEY_ID (+SECRET, +SESSION_TOKEN)", "", static_keys_probe, _env_aws),
        Rung("assume-role", "sigv4-sts", "profile assume-role via STS", "network", assume_role_probe, assume_role_acquire),
        Rung("web-identity", "unsigned-sts", "web identity via STS", "network", web_identity_probe, _web_identity_acquire),
        Rung("sso", "file-cache", "IAM Identity Center (~/.aws/sso/cache)", "network", sso_probe, _sso_acquire),
        Rung("shared-credentials-file", "ini-profile", "~/.aws/credentials", "", shared_file_probe, shared_file_acquire),
        Rung("login", "file-cache", "aws login session (~/.aws/login/cache)", "", login_probe, _login_acquire),
        Rung("credential_process", "subprocess", "profile credential_process", "subprocess", process_probe, _process_acquire),
        Rung("config-file", "ini-profile", "~/.aws/config static keys", "", config_file_probe, config_file_acquire),
        Rung("container", "http-metadata", "container credentials endpoint", "network", container_probe, _container_acquire),
        Rung("imds", "http-metadata", "EC2 instance metadata (IMDSv2)", "network", imds_probe, _imds_acquire),
    ]
    return rungs


# ─── Azure ───────────────────────────────────────────────────────────


def _azure_authority(ctx: ChainContext) -> str:
    return (ctx.settings.get("authority_host") or ctx.env.get("AZURE_AUTHORITY_HOST") or "https://login.microsoftonline.com").rstrip("/")


def _azure_scope(ctx: ChainContext) -> str:
    return ctx.settings.get("scope") or "https://ai.azure.com/.default"


def _azure_token_url(ctx: ChainContext, tenant: str) -> str:
    return f"{_azure_authority(ctx)}/{tenant}/oauth2/v2.0/token"


def azure_certificate_assertion(ctx: ChainContext, tenant: str, client_id: str, pem: str, *, jti: str | None = None,
                                send_chain: bool = False) -> str:
    """The Entra client assertion: RS256 (as azure-identity signs it), ``x5t``
    = base64url SHA-1 of the DER certificate, claims in MSAL's order
    (assertion.py:99-118), 600 s lifetime."""
    key = rs256.load_private_key(pem)
    der = rs256.certificate_der(pem)
    now = int(ctx.now().timestamp())
    header: dict[str, Any] = {"alg": "RS256", "typ": "JWT", "x5t": rs256.b64url(hashlib.sha1(der).digest())}
    if send_chain:
        header["x5c"] = [base64.b64encode(der).decode("ascii")]
    payload = {"aud": _azure_token_url(ctx, tenant), "iss": client_id, "sub": client_id,
               "exp": now + 600, "iat": now, "jti": jti or str(uuid.uuid4())}
    return rs256.jwt_encode(header, payload, key)


def _azure_environment_kind(ctx: ChainContext) -> str | None:
    e = ctx.env
    if not (e.get("AZURE_TENANT_ID") and e.get("AZURE_CLIENT_ID")):
        return None
    if e.get("AZURE_CLIENT_SECRET"):
        return "secret"
    if e.get("AZURE_CLIENT_CERTIFICATE_PATH"):
        return "certificate"
    return None


def azure_environment_request(ctx: ChainContext, *, jti: str | None = None) -> tuple[str, list[tuple[str, str]]]:
    """(token URL, form pairs) for the environment service principal."""
    e = ctx.env
    kind = _azure_environment_kind(ctx)
    tenant, client = e["AZURE_TENANT_ID"], e["AZURE_CLIENT_ID"]
    url = _azure_token_url(ctx, tenant)
    scope = _azure_scope(ctx)
    if kind == "secret":
        return url, [("client_id", client), ("scope", scope), ("client_secret", e["AZURE_CLIENT_SECRET"]), ("grant_type", "client_credentials")]
    if kind == "certificate":
        pem = ctx.read(e["AZURE_CLIENT_CERTIFICATE_PATH"])
        if pem is None:
            raise NotConfiguredError(f"AZURE_CLIENT_CERTIFICATE_PATH {e['AZURE_CLIENT_CERTIFICATE_PATH']} is unreadable")
        if e.get("AZURE_CLIENT_CERTIFICATE_PASSWORD"):
            raise NotConfiguredError("password-protected certificates are not supported; decrypt with `openssl pkey`",
                                     credential_hint="openssl pkey -in cert.pem -out cert-plain.pem")
        send_chain = (e.get("AZURE_CLIENT_SEND_CERTIFICATE_CHAIN") or "").lower() in ("1", "true")
        assertion = azure_certificate_assertion(ctx, tenant, client, pem, jti=jti, send_chain=send_chain)
        return url, [("client_id", client), ("scope", scope), ("client_assertion_type", _CLIENT_ASSERTION_TYPE),
                     ("client_assertion", assertion), ("grant_type", "client_credentials")]
    raise NotConfiguredError("Azure environment credential needs AZURE_CLIENT_SECRET or AZURE_CLIENT_CERTIFICATE_PATH")


def _azure_environment_acquire(ctx: ChainContext) -> BearerToken | None:
    if _azure_environment_kind(ctx) is None:
        return None
    url, pairs = azure_environment_request(ctx)
    data = _exchange(ctx, "POST", url, {"content-type": "application/x-www-form-urlencoded"}, _form(pairs), "Entra client credentials")
    return _bearer_from_oauth(data, ctx.now(), "Entra")


def _azure_workload_config(ctx: ChainContext) -> bool:
    e = ctx.env
    return bool(e.get("AZURE_FEDERATED_TOKEN_FILE") and e.get("AZURE_CLIENT_ID") and e.get("AZURE_TENANT_ID"))


def _azure_workload_acquire(ctx: ChainContext) -> BearerToken | None:
    if not _azure_workload_config(ctx):
        return None
    e = ctx.env
    token = ctx.read(e["AZURE_FEDERATED_TOKEN_FILE"])
    if token is None:
        raise NotConfiguredError(f"AZURE_FEDERATED_TOKEN_FILE {e['AZURE_FEDERATED_TOKEN_FILE']} is unreadable")
    pairs = [("client_id", e["AZURE_CLIENT_ID"]), ("scope", _azure_scope(ctx)), ("client_assertion_type", _CLIENT_ASSERTION_TYPE),
             ("client_assertion", token.strip()), ("grant_type", "client_credentials")]
    data = _exchange(ctx, "POST", _azure_token_url(ctx, e["AZURE_TENANT_ID"]), {"content-type": "application/x-www-form-urlencoded"}, _form(pairs), "Entra workload identity")
    return _bearer_from_oauth(data, ctx.now(), "Entra")


def _azure_msi_flavor(ctx: ChainContext) -> str:
    e = ctx.env
    if e.get("IDENTITY_ENDPOINT"):
        if e.get("IDENTITY_HEADER"):
            return "service-fabric" if e.get("IDENTITY_SERVER_THUMBPRINT") else "app-service"
        if e.get("IMDS_ENDPOINT"):
            return "azure-arc"
    if e.get("MSI_ENDPOINT"):
        return "azure-ml" if e.get("MSI_SECRET") else "cloud-shell"
    return "imds"


def _azure_msi_acquire(ctx: ChainContext) -> BearerToken | None:
    if ctx.http is None:
        return None
    e = ctx.env
    flavor = _azure_msi_flavor(ctx)
    resource = _azure_scope(ctx).removesuffix("/.default")
    client_id = e.get("AZURE_CLIENT_ID")
    now = ctx.now()
    if flavor == "imds":
        query = {"api-version": "2018-02-01", "resource": resource}
        if client_id:
            query["client_id"] = client_id
        try:
            status, _, raw = ctx.http("GET", "http://169.254.169.254/metadata/identity/oauth2/token?" + urllib.parse.urlencode(query), {"Metadata": "true"}, None, 1.0)
        except Exception:  # noqa: BLE001 - not on Azure: absent
            return None
        if status != 200:
            return None
        return _bearer_from_oauth(_json_body(raw), now, "managed identity")
    if flavor == "app-service":
        query = {"api-version": "2019-08-01", "resource": resource}
        if client_id:
            query["client_id"] = client_id
        data = _exchange(ctx, "GET", e["IDENTITY_ENDPOINT"] + "?" + urllib.parse.urlencode(query), {"X-IDENTITY-HEADER": e["IDENTITY_HEADER"]}, None, "App Service managed identity")
        return _bearer_from_oauth(data, now, "managed identity")
    if flavor == "cloud-shell":
        data = _exchange(ctx, "POST", e["MSI_ENDPOINT"], {"Metadata": "true", "content-type": "application/x-www-form-urlencoded"}, _form([("resource", resource)]), "Cloud Shell managed identity")
        return _bearer_from_oauth(data, now, "managed identity")
    if flavor == "azure-ml":
        query = {"api-version": "2017-09-01", "resource": resource}
        if client_id:
            query["clientid"] = client_id
        data = _exchange(ctx, "GET", e["MSI_ENDPOINT"] + "?" + urllib.parse.urlencode(query), {"secret": e["MSI_SECRET"]}, None, "Azure ML managed identity")
        return _bearer_from_oauth(data, now, "managed identity")
    if flavor == "azure-arc":
        url = e["IDENTITY_ENDPOINT"] + "?" + urllib.parse.urlencode({"api-version": "2019-11-01", "resource": resource})
        status, headers, _ = ctx.http("GET", url, {"Metadata": "true"}, None, 5.0)
        challenge = headers.get("www-authenticate", "")
        if status != 401 or "realm=" not in challenge:
            raise AuthError(f"Azure Arc managed identity: expected a 401 challenge, got {status}")
        key_path = challenge.split("realm=", 1)[1].strip().strip('"')
        directory = (Path(ctx.env.get("PROGRAMDATA", "C:/ProgramData")) / "AzureConnectedMachineAgent" / "Tokens"
                     if os.name == "nt" else Path("/var/opt/azcmagent/tokens"))
        path = Path(key_path)
        if path.parent != directory or path.suffix != ".key":
            raise AuthError("Azure Arc managed identity: invalid challenge file location")
        if ctx.files is None and path.resolve().parent != directory.resolve():
            raise AuthError("Azure Arc managed identity: invalid challenge file location")
        secret = ctx.read(key_path)
        if secret is None or len(secret) > 4096:
            raise AuthError("Azure Arc managed identity: challenge file missing or too large")
        data = _exchange(ctx, "GET", url, {"Metadata": "true", "Authorization": f"Basic {secret.strip()}"}, None, "Azure Arc managed identity")
        return _bearer_from_oauth(data, now, "managed identity")
    raise NotConfiguredError("Service Fabric managed identity (TLS thumbprint pinning) is not supported; use a certificate or secret")


def _az_cli_acquire(ctx: ChainContext) -> BearerToken | None:
    if ctx.run is None or ctx.on_path("az") is None:
        return None
    argv = ["az", "account", "get-access-token", "--output", "json", "--scope", _azure_scope(ctx)]
    tenant = ctx.env.get("AZURE_TENANT_ID")
    if tenant:
        argv += ["--tenant", tenant]
    data = _json_body(ctx.run(argv, 30.0).encode())
    token = data.get("accessToken")
    if token is None:
        return None
    parsed = _bearer_from_oauth({"access_token": token, "expires_on": data.get("expires_on")}, ctx.now(), "Azure CLI")
    expires = parsed.expires_at
    if expires is None and data.get("expiresOn"):
        try:
            expires = datetime.fromisoformat(str(data["expiresOn"])).astimezone(timezone.utc)
        except (ValueError, OverflowError, OSError):
            expires = None
    return BearerToken(parsed.value, expires_at=expires)


def _pwsh_acquire(ctx: ChainContext) -> BearerToken | None:
    if ctx.run is None or ctx.on_path("pwsh") is None:
        return None
    resource = _azure_scope(ctx).removesuffix("/.default").replace("'", "''")
    script = f"Get-AzAccessToken -ResourceUrl '{resource}' -AsSecureString:$false | ConvertTo-Json -Compress"
    data = _json_body(ctx.run(["pwsh", "-NoProfile", "-NonInteractive", "-Command", script], 30.0).encode())
    token = data.get("Token")
    return _bearer_from_oauth({"access_token": token}, ctx.now(), "Azure PowerShell") if token is not None else None


def _azd_acquire(ctx: ChainContext) -> BearerToken | None:
    if ctx.run is None or ctx.on_path("azd") is None:
        return None
    data = _json_body(ctx.run(["azd", "auth", "token", "--output", "json", "--scope", _azure_scope(ctx)], 30.0).encode())
    token = data.get("token")
    if token is None:
        return None
    parsed = _bearer_from_oauth({"access_token": token}, ctx.now(), "Azure Developer CLI")
    expires = None
    if data.get("expiresOn"):
        try:
            expires = parse_rfc3339(str(data["expiresOn"]))
        except ValueError:
            expires = None
    return BearerToken(parsed.value, expires_at=expires)


def _azure_chain(policy: AccessPolicy) -> list[Rung]:
    door_key = policy.env_keys[0] if policy.env_keys else None

    def narrowed(ctx: ChainContext, name: str, developer: bool) -> bool:
        """AZURE_TOKEN_CREDENTIALS=prod|dev|<CredentialName> narrowing."""
        value = (ctx.env.get("AZURE_TOKEN_CREDENTIALS") or "").strip().lower()
        if not value:
            return False
        if value == "prod":
            return developer
        if value == "dev":
            return not developer
        return value != name.lower()

    def env_probe(ctx: ChainContext):
        if narrowed(ctx, "EnvironmentCredential", False):
            return ("absent", "excluded by AZURE_TOKEN_CREDENTIALS")
        kind = _azure_environment_kind(ctx)
        return ("configured", f"service principal by {kind} (token exchange at request time)") if kind else ("absent", "AZURE_TENANT_ID/AZURE_CLIENT_ID + secret or certificate not set")

    def workload_probe(ctx: ChainContext):
        if narrowed(ctx, "WorkloadIdentityCredential", False):
            return ("absent", "excluded by AZURE_TOKEN_CREDENTIALS")
        return ("configured", "federated token file present (exchange at request time)") if _azure_workload_config(ctx) else ("absent", "AZURE_FEDERATED_TOKEN_FILE not set")

    def msi_probe(ctx: ChainContext):
        if narrowed(ctx, "ManagedIdentityCredential", False):
            return ("absent", "excluded by AZURE_TOKEN_CREDENTIALS")
        return ("configured", f"managed identity ({_azure_msi_flavor(ctx)}) probed at request time")

    def cli_probe(name: str, label: str, command: str):
        def probe(ctx: ChainContext):
            if narrowed(ctx, name, True):
                return ("absent", "excluded by AZURE_TOKEN_CREDENTIALS")
            if ctx.on_path(command) is None:
                return ("absent", f"{command} is not on PATH")
            return ("configured", f"{label} run at request time")
        return probe

    def guard(fn, name: str, developer: bool):
        return lambda ctx: None if narrowed(ctx, name, developer) else fn(ctx)

    rungs: list[Rung] = []
    if door_key:
        rungs.append(Rung(f"env:{door_key}", "env", f"env ${door_key}", "",
                          lambda ctx, k=door_key: ("usable", "set (value never shown)") if ctx.env.get(k) else ("absent", "not set"),
                          lambda ctx, k=door_key: ApiKey(ctx.env[k]) if ctx.env.get(k) else None))
    rungs += [
        Rung("environment", "http-token-exchange", "Entra service principal from AZURE_* env", "network", env_probe, guard(_azure_environment_acquire, "EnvironmentCredential", False)),
        Rung("workload-identity", "http-token-exchange", "Entra workload identity", "network", workload_probe, guard(_azure_workload_acquire, "WorkloadIdentityCredential", False)),
        Rung("managed-identity", "http-metadata", "Azure managed identity", "network", msi_probe, guard(_azure_msi_acquire, "ManagedIdentityCredential", False)),
        Rung("az", "subprocess", "az account get-access-token", "subprocess", cli_probe("AzureCliCredential", "`az`", "az"), guard(_az_cli_acquire, "AzureCliCredential", True)),
        Rung("pwsh", "subprocess", "Azure PowerShell Get-AzAccessToken", "subprocess", cli_probe("AzurePowerShellCredential", "`pwsh`", "pwsh"), guard(_pwsh_acquire, "AzurePowerShellCredential", True)),
        Rung("azd", "subprocess", "azd auth token", "subprocess", cli_probe("AzureDeveloperCliCredential", "`azd`", "azd"), guard(_azd_acquire, "AzureDeveloperCliCredential", True)),
    ]
    return rungs


# ─── Google Cloud ────────────────────────────────────────────────────


def gcp_service_account_assertion(ctx: ChainContext, info: Mapping[str, Any], scope: str = _GCP_SCOPE) -> tuple[str, str]:
    """(token_uri, JWT) for a ``service_account`` file: header typ/alg/kid,
    claims iat, exp=iat+3600, iss, aud, scope (google-auth
    service_account.py:393-420, jwt.py:75-107)."""
    key = rs256.load_private_key(str(info["private_key"]))
    now = int(ctx.now().timestamp())
    token_uri = str(info.get("token_uri") or _GCP_TOKEN_URL)
    header: dict[str, Any] = {"alg": "RS256", "typ": "JWT"}
    if info.get("private_key_id"):
        header["kid"] = str(info["private_key_id"])
    payload = {"iat": now, "exp": now + 3600, "iss": str(info["client_email"]), "aud": token_uri, "scope": scope}
    return token_uri, rs256.jwt_encode(header, payload, key)


def _gcp_credential_file(ctx: ChainContext, path: str) -> dict | None:
    raw = ctx.read(path)
    if raw is None:
        return None
    try:
        data = json.loads(raw)
    except ValueError:
        raise NotConfiguredError(f"{path}: not valid JSON") from None
    return data if isinstance(data, dict) else None


def _gcp_from_info(ctx: ChainContext, info: Mapping[str, Any], where: str) -> BearerToken:
    kind = info.get("type")
    now = ctx.now()
    if kind == "authorized_user":
        for k in ("refresh_token", "client_id", "client_secret"):
            if not info.get(k):
                raise NotConfiguredError(f"{where}: authorized_user file lacks {k}")
        pairs = [("grant_type", "refresh_token"), ("client_id", str(info["client_id"])), ("client_secret", str(info["client_secret"])),
                 ("refresh_token", str(info["refresh_token"]))]
        data = _exchange(ctx, "POST", str(info.get("token_uri") or _GCP_TOKEN_URL), {"content-type": "application/x-www-form-urlencoded"}, _form(pairs), "Google OAuth refresh")
        return _bearer_from_oauth(data, now, "Google OAuth")
    if kind == "service_account":
        token_uri, assertion = gcp_service_account_assertion(ctx, info)
        data = _exchange(ctx, "POST", token_uri, {"content-type": "application/x-www-form-urlencoded"},
                         _form([("grant_type", _JWT_BEARER), ("assertion", assertion)]), "Google service account")
        return _bearer_from_oauth(data, now, "Google service account")
    if kind == "external_account":
        return _gcp_external_account(ctx, info, where)
    if kind == "impersonated_service_account":
        source = info.get("source_credentials")
        if not isinstance(source, dict):
            raise NotConfiguredError(f"{where}: impersonated_service_account lacks source_credentials")
        base = _gcp_from_info(ctx, source, f"{where}.source_credentials")
        return _gcp_impersonate(ctx, base, str(info["service_account_impersonation_url"]), info.get("delegates") or [])
    raise NotConfiguredError(f"{where}: credential type {kind!r} is not supported by lm15 "
                             "(external_account_authorized_user and gdch_service_account are stated gaps)")


def _gcp_impersonate(ctx: ChainContext, source: BearerToken, url: str, delegates: list) -> BearerToken:
    body = json.dumps({"delegates": list(delegates), "scope": [_GCP_SCOPE], "lifetime": "3600s"}).encode()
    data = _exchange(ctx, "POST", url, {"content-type": "application/json", "authorization": f"Bearer {source.value}"}, body, "generateAccessToken")
    token = data.get("accessToken")
    if not token:
        raise AuthError("generateAccessToken: no accessToken")
    expires = parse_rfc3339(str(data["expireTime"])) if data.get("expireTime") else None
    return BearerToken(str(token), expires_at=expires)


def _gcp_external_account(ctx: ChainContext, info: Mapping[str, Any], where: str) -> BearerToken:
    source = info.get("credential_source") or {}
    if "environment_id" in source:
        raise NotConfiguredError(f"{where}: external_account with an AWS credential_source is a stated gap in lm15; "
                                 "use a file/url/executable source or a service account")
    subject: str | None = None
    fmt = source.get("format") or {}
    if source.get("file"):
        raw = ctx.read(str(source["file"]))
        if raw is None:
            raise NotConfiguredError(f"{where}: subject token file {source['file']} is unreadable")
        subject = raw.strip()
    elif source.get("url"):
        status, _, raw = ctx.http("GET", str(source["url"]), {str(k): str(v) for k, v in (source.get("headers") or {}).items()}, None, 30.0)  # type: ignore[misc]
        if status >= 400:
            raise AuthError(f"{where}: subject token url HTTP {status}")
        subject = raw.decode("utf-8", "replace").strip()
    elif source.get("executable"):
        if ctx.run is None:
            raise NotConfiguredError(f"{where}: executable credential source needs subprocess access")
        if (ctx.env.get("GOOGLE_EXTERNAL_ACCOUNT_ALLOW_EXECUTABLES") or "") != "1":
            raise NotConfiguredError(f"{where}: set GOOGLE_EXTERNAL_ACCOUNT_ALLOW_EXECUTABLES=1 to allow the executable source")
        import shlex

        out = ctx.run(shlex.split(str(source["executable"]["command"])), float(source["executable"].get("timeout_millis", 30000)) / 1000)
        data = _json_body(out.encode())
        if not data.get("success", True):
            raise AuthError("external account executable reported failure")
        subject = data.get("id_token") or data.get("saml_response")
        fmt = {"type": "text"}
    if subject is None:
        raise NotConfiguredError(f"{where}: external_account credential_source is not file/url/executable")
    if fmt.get("type") == "json":
        field_name = str(fmt.get("subject_token_field_name") or "")
        subject = str(_json_body(subject.encode()).get(field_name) or "")
    body = json.dumps({
        "grantType": "urn:ietf:params:oauth:grant-type:token-exchange",
        "audience": str(info["audience"]),
        "scope": _GCP_SCOPE,
        "requestedTokenType": "urn:ietf:params:oauth:token-type:access_token",
        "subjectToken": subject,
        "subjectTokenType": str(info["subject_token_type"]),
    }).encode()
    data = _exchange(ctx, "POST", str(info.get("token_url") or _GCP_STS_URL), {"content-type": "application/json"}, body, "Google STS exchange")
    token = _bearer_from_oauth(data, ctx.now(), "Google STS")
    if info.get("service_account_impersonation_url"):
        return _gcp_impersonate(ctx, token, str(info["service_account_impersonation_url"]), [])
    return token


def _gcp_metadata_acquire(ctx: ChainContext) -> BearerToken | None:
    if ctx.http is None or (ctx.env.get("NO_GCE_CHECK") or "").lower() in ("1", "true"):
        return None
    host = ctx.env.get("GCE_METADATA_HOST") or ctx.env.get("GCE_METADATA_ROOT") or "metadata.google.internal"
    try:
        status, _, raw = ctx.http("GET", f"http://{host}/computeMetadata/v1/instance/service-accounts/default/token", {"Metadata-Flavor": "Google"}, None, 1.0)
    except Exception:  # noqa: BLE001 - not on GCE: absent
        return None
    if status != 200:
        return None
    return _bearer_from_oauth(_json_body(raw), ctx.now(), "GCE metadata")


def _gcloud_acquire(ctx: ChainContext) -> BearerToken | None:
    if ctx.run is None or ctx.on_path("gcloud") is None:
        return None
    token = ctx.run(["gcloud", "auth", "print-access-token"], 30.0).strip()
    return BearerToken(token) if token else None


def _adc_file_path(ctx: ChainContext) -> str:
    base = ctx.env.get("CLOUDSDK_CONFIG") or "~/.config/gcloud"
    return f"{base.rstrip('/')}/application_default_credentials.json"


def _gcp_chain(policy: AccessPolicy) -> list[Rung]:
    door_key = policy.env_keys[0] if policy.env_keys else None

    def file_probe(label: str, path_fn):
        def probe(ctx: ChainContext):
            path = path_fn(ctx)
            if not path:
                return ("absent", f"{label} not set")
            info = _gcp_credential_file(ctx, path)
            if info is None:
                return ("absent", f"{path} missing or unreadable")
            return ("configured", f"{info.get('type', '?')} credentials in {path} (token exchange at request time)")
        return probe

    def file_acquire(path_fn):
        def acquire(ctx: ChainContext):
            path = path_fn(ctx)
            if not path:
                return None
            info = _gcp_credential_file(ctx, path)
            return _gcp_from_info(ctx, info, path) if info else None
        return acquire

    env_path = lambda ctx: ctx.env.get("GOOGLE_APPLICATION_CREDENTIALS")  # noqa: E731
    adc_path = lambda ctx: _adc_file_path(ctx)  # noqa: E731

    def metadata_probe(ctx: ChainContext):
        if (ctx.env.get("NO_GCE_CHECK") or "").lower() in ("1", "true"):
            return ("absent", "NO_GCE_CHECK set")
        return ("configured", "GCE metadata server probed at request time")

    rungs: list[Rung] = []
    if door_key:
        rungs.append(Rung(f"env:{door_key}", "env", f"env ${door_key}", "",
                          lambda ctx, k=door_key: ("usable", "set (value never shown)") if ctx.env.get(k) else ("absent", "not set"),
                          lambda ctx, k=door_key: ApiKey(ctx.env[k]) if ctx.env.get(k) else None))
    rungs += [
        Rung("adc-env", "json-file", "GOOGLE_APPLICATION_CREDENTIALS file", "network", file_probe("GOOGLE_APPLICATION_CREDENTIALS", env_path), file_acquire(env_path)),
        Rung("adc-file", "json-file", "gcloud application default credentials file", "network", file_probe("ADC file", adc_path), file_acquire(adc_path)),
        Rung("metadata", "http-metadata", "GCE metadata server", "network", metadata_probe, _gcp_metadata_acquire),
        Rung("gcloud", "subprocess", "gcloud auth print-access-token", "subprocess",
             lambda ctx: ("configured", "`gcloud` run at request time") if ctx.on_path("gcloud") else ("absent", "gcloud is not on PATH"),
             _gcloud_acquire),
    ]
    return rungs


# ─── Settings from the cloud profile (AUTH-10 fallbacks after env) ────


def profile_settings(policy: AccessPolicy, ctx: ChainContext) -> Callable[[str], str | None]:
    """The setting values the cloud's own config files carry: AWS
    ``region`` from the active profile (`~/.aws/config`, then
    `~/.aws/credentials`); GCP ``project`` from the ADC file's
    ``quota_project_id`` / ``project_id``.  Nothing for Azure."""

    def lookup(name: str) -> str | None:
        if policy.credential_policy == "aws-chain" and name == "region":
            creds, conf, profile = _aws_config(ctx)
            value = _aws_profile_section(conf, profile).get("region")
            if not value and creds.has_section(profile):
                value = creds[profile].get("region")
            return value or None
        if policy.credential_policy == "gcp-chain" and name == "project":
            for path in (ctx.env.get("GOOGLE_APPLICATION_CREDENTIALS"), _adc_file_path(ctx)):
                if path:
                    info = _gcp_credential_file(ctx, path) or {}
                    value = info.get("quota_project_id") or info.get("project_id")
                    if value:
                        return str(value)
        return None

    return lookup


# ─── Chains ──────────────────────────────────────────────────────────

_CHAINS = {"aws-chain": _aws_chain, "azure-chain": _azure_chain, "gcp-chain": _gcp_chain}


def chain_for(policy: AccessPolicy) -> list[Rung]:
    builder = _CHAINS.get(policy.credential_policy)
    if builder is None:
        raise ValueError(f"{policy.provider}: not a cloud chain policy")
    return builder(policy)


def explain(policy: AccessPolicy, ctx: ChainContext, *, explicit: bool) -> tuple[list[Step], bool]:
    """The AUTH-7 walk.  ``explicit`` = an api_keys entry exists (rung 0)."""
    steps: list[Step] = []
    selected = False
    if explicit:
        steps.append(Step("api_keys", "explicit api_keys entry", "provided (value never shown)", "selected"))
        selected = True
    else:
        steps.append(Step("api_keys", "explicit api_keys entry", "not provided", "absent"))
    for rung in chain_for(policy):
        try:
            verdict, detail = rung.probe(ctx)
        except NotConfiguredError as exc:
            verdict, detail = "absent", str(exc).splitlines()[0]
        if verdict == "absent":
            state = "absent"
        elif verdict == "configured":
            state = "shadowed" if selected else "unprobed"
        else:
            state = "shadowed" if selected else "selected"
            selected = True
        steps.append(Step(rung.name, rung.source, detail, state))
    # ``configured`` is true when something may supply the credential: a
    # rung selected offline, or a configured network/subprocess rung the
    # offline doctor could not probe.  The report names which.
    return steps, selected or any(s.state == "unprobed" for s in steps)


def resolve(policy: AccessPolicy, ctx: ChainContext) -> CredentialValue:
    """Walk the chain online; the first rung that yields wins.  A rung that
    is configured and fails raises. Azure developer commands are the
    AUTH-1 exception: try all three before reporting their failure.
    Deployed Azure credentials and AWS/GCP failures never fall through."""
    developer_failed = False
    for rung in chain_for(policy):
        try:
            got = rung.acquire(ctx)
        except AuthError:
            if policy.credential_policy == "azure-chain" and rung.name in {"az", "pwsh", "azd"}:
                developer_failed = True
                continue
            raise
        if got is not None:
            return got
    if developer_failed:
        raise AuthError("Azure developer credentials failed; sign in with az, Azure PowerShell, or azd",
                        provider=policy.provider)
    raise NotConfiguredError(
        f"{policy.provider}: no credential found in the {policy.credential_policy} chain"
        + (f"; set {policy.env_keys[0]} or configure the cloud SDK" if policy.env_keys else "; configure the cloud SDK"),
        provider=policy.provider,
        env_keys=policy.env_keys,
    )


class _CachingProvider:
    """AUTH-2/AUTH-3: resolve once, hand out until the skew window, then
    re-resolve.  In memory only; never written to a foreign file."""

    def __init__(self, policy: AccessPolicy, ctx: ChainContext) -> None:
        self._policy = policy
        self._ctx = ctx
        self._lock = threading.Lock()
        self._value: CredentialValue | None = None

    def __call__(self) -> CredentialValue:
        with self._lock:
            if self._value is None or self._value.is_expired(self._ctx.now()):
                value = resolve(self._policy, self._ctx)
                if value.is_expired(self._ctx.now()):
                    raise AuthError("cloud credential is expired; renew the configured credential source",
                                    provider=self._policy.provider)
                # CLI output without an expiry cannot safely be cached forever.
                self._value = value if isinstance(value, (ApiKey, AwsCredentials)) or value.expires_at is not None else None
                return value
            return self._value

    def __repr__(self) -> str:
        return f"<cloud credential provider for {self._policy.provider}>"


def cache_key(policy: AccessPolicy, ctx: ChainContext) -> str:
    """Provider id + the identity-selecting settings (AUTH-3)."""
    e = ctx.env
    parts = [policy.provider, ctx.env.get("AWS_PROFILE") or "", e.get("AZURE_TENANT_ID") or "", e.get("AZURE_CLIENT_ID") or "",
             e.get("GOOGLE_APPLICATION_CREDENTIALS") or "", e.get("CLOUDSDK_CONFIG") or "", str(ctx.home)]
    parts += [f"{k}={v}" for k, v in sorted(ctx.settings.items())]
    return hashlib.sha256("\x1f".join(parts).encode()).hexdigest()


def credential_provider(policy: AccessPolicy, ctx: ChainContext) -> Callable[[], CredentialValue]:
    return _CachingProvider(policy, ctx)


# ─── Harness ops (PROTOCOL.md token_exchange_build / token_exchange_parse) ──


def token_exchange_build(policy: AccessPolicy, rung: str, inputs: Mapping[str, Any], ctx: ChainContext) -> dict[str, Any]:
    """The exact token-exchange request a rung would send under ``ctx.now``."""
    if rung in ("adc-env", "adc-file", "service-account"):
        info = inputs["credential_file"]
        scope = str(inputs.get("scope") or _GCP_SCOPE)
        token_uri, assertion = gcp_service_account_assertion(ctx, info, scope)
        return {"method": "POST", "url": token_uri, "headers": {"content-type": "application/x-www-form-urlencoded"},
                "body_encoding": "form", "body": {"grant_type": _JWT_BEARER, "assertion": assertion}}
    if rung == "environment":
        url, pairs = azure_environment_request(ctx, jti=inputs.get("jti"))
        return {"method": "POST", "url": url, "headers": {"content-type": "application/x-www-form-urlencoded"},
                "body_encoding": "form", "body": dict(pairs)}
    raise ValueError(f"token_exchange_build: rung {rung!r} has no deterministic request")


def token_exchange_parse(policy: AccessPolicy, rung: str, status: int, body: Mapping[str, Any], ctx: ChainContext) -> CredentialValue:
    now = ctx.now()
    if rung in ("adc-env", "adc-file", "service-account", "environment", "workload-identity", "managed-identity", "metadata"):
        if not 200 <= status < 300:
            raise AuthError(f"{rung}: HTTP {status}")
        return _bearer_from_oauth(dict(body), now, rung)
    if rung == "credential_process":
        if status != 0 or body.get("Version") != 1:
            raise AuthError("credential_process failed or returned an unsupported Version")
        return _aws_from_response(body)
    if rung in ("imds", "container"):
        if not 200 <= status < 300:
            raise AuthError(f"{rung}: HTTP {status}")
        return _aws_from_response(body)
    raise ValueError(f"token_exchange_parse: rung {rung!r} is not a parse vector")
