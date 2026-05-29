"""MkDocs hook: fetch live stats at build time and inject into config.extra.stats.

Hits GitHub, PyPI, and Discord APIs once per build. Results are cached to
.cache/stats.json for 1 hour so dev-server rebuilds don't hammer rate limits.
Falls back to cached (or hardcoded) values on any API failure.
"""

import json
import os
import re
import time
import logging
from datetime import datetime, timedelta
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import URLError

log = logging.getLogger("mkdocs.hooks.fetch_stats")

REPO = "stanfordnlp/dspy"
DISCORD_INVITE = "XCGy2WDCQB"
CACHE_DIR = Path(__file__).parent.parent / ".cache"
CACHE_FILE = CACHE_DIR / "stats.json"
CACHE_VERSION = 3
CACHE_TTL = 3600  # 1 hour

DEFAULTS = {
    "release_version": "3.3.0b1",
    "release_major_minor": "3.3",
    "release_date": "May 2026",
    "release_blurb": "New ReActV2 Module and improved LM/BaseLM",
    "monthly_downloads": "7.5M+",
    "contributors": "400+",
    "stars": "34.6k",
    "discord_members": "8.4k",
    "merged_prs_yr": "420+",
}

GITHUB_HEADERS = {"Accept": "application/vnd.github.v3+json"}


def _get(url, headers=None, timeout=10):
    req = Request(url, headers=headers or {})
    with urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read()), resp.headers


def _fmt(n):
    if n >= 1_000_000:
        v = n / 1_000_000
        return f"{v:.1f}M+" if v < 10 else f"{v:.0f}M+"
    if n >= 1_000:
        v = n / 1_000
        return f"{v:.1f}k" if v < 10 else f"{v:.0f}k"
    return f"{n}+"


def _load_cache():
    try:
        if CACHE_FILE.exists():
            data = json.loads(CACHE_FILE.read_text())
            if data.get("_cache_version") != CACHE_VERSION:
                return None
            if time.time() - data.get("_ts", 0) < CACHE_TTL:
                log.info("Using cached stats (age: %ds)", int(time.time() - data["_ts"]))
                return data
    except Exception:
        pass
    return None


def _save_cache(stats):
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        stats["_ts"] = time.time()
        stats["_cache_version"] = CACHE_VERSION
        CACHE_FILE.write_text(json.dumps(stats, indent=2))
    except Exception as e:
        log.warning("Could not write stats cache: %s", e)


def _fetch_release():
    data, _ = _get(
        f"https://api.github.com/repos/{REPO}/releases?per_page=20",
        headers=GITHUB_HEADERS,
    )
    releases = [release for release in data if not release.get("draft") and release.get("published_at")]
    if not releases:
        return {}

    release = max(
        releases,
        key=lambda release: datetime.fromisoformat(release["published_at"].replace("Z", "+00:00")),
    )
    tag = release["tag_name"].lstrip("v")
    parts = tag.split(".")
    major_minor = f"{parts[0]}.{parts[1]}" if len(parts) >= 2 else tag
    pub = datetime.fromisoformat(release["published_at"].replace("Z", "+00:00"))
    return {
        "release_version": tag,
        "release_major_minor": major_minor,
        "release_date": pub.strftime("%b %Y"),
    }


def _fetch_pypi_downloads():
    data, _ = _get("https://pypistats.org/api/packages/dspy/recent")
    return {"monthly_downloads": _fmt(data["data"]["last_month"])}


def _fetch_stars():
    data, _ = _get(
        f"https://api.github.com/repos/{REPO}",
        headers=GITHUB_HEADERS,
    )
    return {"stars": _fmt(data["stargazers_count"])}


def _fetch_contributors():
    _, headers = _get(
        f"https://api.github.com/repos/{REPO}/contributors?per_page=1&anon=true",
        headers=GITHUB_HEADERS,
    )
    link = headers.get("Link", "")
    m = re.search(r'page=(\d+)>; rel="last"', link)
    if m:
        return {"contributors": _fmt(int(m.group(1)))}
    return {}


def _fetch_discord():
    data, _ = _get(
        f"https://discord.com/api/v9/invites/{DISCORD_INVITE}?with_counts=true"
    )
    return {"discord_members": _fmt(data["approximate_member_count"])}


def _fetch_merged_prs():
    one_year_ago = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
    q = f"repo:{REPO}+is:pr+is:merged+merged:>{one_year_ago}"
    data, _ = _get(
        f"https://api.github.com/search/issues?q={q}&per_page=1",
        headers=GITHUB_HEADERS,
    )
    return {"merged_prs_yr": _fmt(data["total_count"])}


FETCHERS = [
    ("release", _fetch_release),
    ("pypi", _fetch_pypi_downloads),
    ("stars", _fetch_stars),
    ("contributors", _fetch_contributors),
    ("discord", _fetch_discord),
    ("merged_prs", _fetch_merged_prs),
]


def _fetch_all():
    stats = dict(DEFAULTS)
    for name, fn in FETCHERS:
        try:
            result = fn()
            stats.update(result)
            log.info("  %s: %s", name, result)
        except Exception as e:
            log.warning("  %s failed, using fallback: %s", name, e)
    return stats


# ── MkDocs hook entry point ──────────────────────────────────────────────────

def on_config(config):
    log.info("Fetching live stats for home page...")

    cached = _load_cache()
    if cached:
        stats = {k: v for k, v in cached.items() if not k.startswith("_")}
    else:
        stats = _fetch_all()
        _save_cache(stats)

    config.setdefault("extra", {})["stats"] = stats
    log.info("Stats injected: %s", stats)
    return config
