#!/usr/bin/env python3
"""Sync the docs and the Python sources into the Mixedbread stores that back
the docs "Ask AI" widget (docs/api/chat.js).

Two stores are synced:

* ``dspy-docs`` — the *published* documentation pages as Markdown. Each page
  is the ``<page>/index.md`` file that ``mkdocs build`` emits through the
  ``mkdocs-llmstxt`` plugin (rendered API reference and notebooks included),
  so what gets indexed is exactly what https://dspy.ai serves.
* ``dspy-code`` — the Python sources under ``dspy/``.

Usage (from the repository root; needs ``pip install mixedbread`` and
``MXBAI_API_KEY`` in the environment or in a repo-root ``.env``):

    python3 docs/scripts/sync_search_index.py [--site docs/site | --site https://dspy.ai] [--dry-run]

The sync is incremental: files are keyed by their page path (``external_id``)
and skipped when their sha256 matches the store's copy; files that no longer
exist locally are removed from the store. Pages come from a local
``mkdocs build`` output (``docs/site``) when present, otherwise from the live
site's ``llms.txt`` page list. CI runs ``mkdocs build`` then this script on every
push to ``main`` that touches ``docs/`` or ``dspy/``.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DOCS = REPO / "docs"
DEFAULT_SITE_DIR = DOCS / "site"
LIVE_SITE = "https://dspy.ai"

DOCS_STORE = "dspy-docs"
CODE_STORE = "dspy-code"
STORES = [DOCS_STORE, CODE_STORE]

# rendered pages that are navigation-only or not worth indexing
SKIP_PAGES = {"index.md", "404.md", "search/index.md"}
CODE_PATTERNS = ["**/*.py"]
CODE_BASE = REPO / "dspy"


# ── credentials ────────────────────────────────────────────────────────────

def load_api_key() -> None:
    if os.environ.get("MXBAI_API_KEY"):
        return
    env = REPO / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            key, _, value = line.strip().partition("=")
            if key == "MXBAI_API_KEY" and value:
                os.environ["MXBAI_API_KEY"] = value.strip().strip("\"'")
                return
    sys.exit("MXBAI_API_KEY is not set (env or .env)")


def client():
    load_api_key()
    try:
        from mixedbread import Mixedbread
    except ImportError:
        sys.exit("missing dependency: pip install mixedbread")
    return Mixedbread()


# ── page sources ───────────────────────────────────────────────────────────

def _fetch(url: str, timeout: int = 60) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "dspy-docs-sync"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def page_key(rel: str) -> str:
    """'getting-started/first-program/index.md' -> same; keeps a stable id."""
    return rel.replace("\\", "/").lstrip("/")


def local_site_pages(site_dir: Path) -> dict[str, bytes]:
    pages = {}
    for path in sorted(site_dir.rglob("*.md")):
        rel = page_key(str(path.relative_to(site_dir)))
        if rel in SKIP_PAGES or rel == "llms.txt":
            continue
        body = path.read_bytes()
        if body.strip():
            pages[rel] = body
    return pages


def live_site_pages(base_url: str) -> dict[str, bytes]:
    """Page list from llms.txt, then each page's rendered Markdown."""
    base_url = base_url.rstrip("/")
    llms = _fetch(f"{base_url}/llms.txt").decode("utf-8", "replace")
    urls = re.findall(r"\]\((https?://[^)\s]+\.md)\)", llms)
    urls = list(dict.fromkeys(urls))
    keys = [page_key(u[len(base_url):]) if u.startswith(base_url) else None for u in urls]

    def get(pair):
        url, key = pair
        if key is None or key in SKIP_PAGES:
            return key, None
        try:
            return key, _fetch(url)
        except Exception as e:  # noqa: BLE001 — one missing page must not kill the sync
            print(f"skip {url}: {e}")
            return key, None

    pages = {}
    with ThreadPoolExecutor(max_workers=8) as pool:
        for key, body in pool.map(get, zip(urls, keys)):
            if body and body.strip():
                pages[key] = body
    return pages


def docs_pages(site: str | None) -> dict[str, bytes]:
    if site is None:
        if DEFAULT_SITE_DIR.is_dir():
            site = str(DEFAULT_SITE_DIR)
        else:
            print(f"no local build at {DEFAULT_SITE_DIR}; fetching pages from {LIVE_SITE}")
            site = LIVE_SITE
    if site.startswith(("http://", "https://")):
        return live_site_pages(site)
    return local_site_pages(Path(site))


def code_files() -> dict[str, bytes]:
    files = {}
    for pattern in CODE_PATTERNS:
        for path in sorted(CODE_BASE.glob(pattern)):
            body = path.read_bytes()
            if not body.strip():
                continue  # the store rejects empty files (e.g. bare __init__.py)
            rel = "dspy/" + page_key(str(path.relative_to(CODE_BASE)))
            files[rel] = body
    return files


# ── store sync ─────────────────────────────────────────────────────────────

def store_files(mxbai, store):
    files, after = {}, None
    while True:
        page = mxbai.stores.files.list(store, limit=100, after=after)
        for f in page.data:
            files[f.external_id or f.filename] = f
        if len(page.data) < 100:
            return files
        after = page.data[-1].id


def sync_store(mxbai, store: str, local: dict[str, bytes], dry_run: bool) -> None:
    try:
        mxbai.stores.retrieve(store)
    except Exception:
        print(f"creating store {store}")
        if not dry_run:
            mxbai.stores.create(name=store)

    digests = {rel: hashlib.sha256(body).hexdigest() for rel, body in local.items()}
    try:
        remote = store_files(mxbai, store)
    except Exception:
        remote = {}

    stale = [f for rel, f in remote.items() if rel not in local]
    changed = [
        rel
        for rel, digest in digests.items()
        if rel not in remote or (remote[rel].metadata or {}).get("sha256") != digest
    ]

    def upload(rel):
        print(f"upload {store}:{rel}")
        if dry_run:
            return
        # explicit content type: the server's sniffing mislabels .md/.py
        mime = "text/markdown" if rel.endswith(".md") else "text/plain"
        # a readable filename ('getting-started__first-program.md') so chunk
        # results are self-describing even without metadata
        name = rel.replace("/index.md", ".md").replace("/", "__")
        mxbai.stores.files.upload_and_poll(
            store_identifier=store,
            file=(name, local[rel], mime),
            external_id=rel,
            overwrite=True,
            metadata={"path": rel, "sha256": digests[rel],
                      "url": f"{LIVE_SITE}/{rel.removesuffix('index.md')}" if store == DOCS_STORE
                      else f"https://github.com/stanfordnlp/dspy/blob/main/{rel}"},
        )

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(upload, changed))
    for f in stale:
        print(f"delete {store}:{f.external_id or f.filename}")
        if not dry_run:
            mxbai.stores.files.delete(f.id, store_identifier=store)
    print(f"{store}: {len(changed)} uploaded, {len(stale)} deleted, "
          f"{len(local) - len(changed)} unchanged")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--site", default=None,
                        help="mkdocs build dir or site URL (default: docs/site, else https://dspy.ai)")
    parser.add_argument("--store", choices=STORES, default=None, help="sync only one store")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    mxbai = client()
    if args.store in (None, DOCS_STORE):
        sync_store(mxbai, DOCS_STORE, docs_pages(args.site), args.dry_run)
    if args.store in (None, CODE_STORE):
        sync_store(mxbai, CODE_STORE, code_files(), args.dry_run)


if __name__ == "__main__":
    main()
