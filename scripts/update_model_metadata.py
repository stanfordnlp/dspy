"""Refresh DSPy's bundled metadata from a released LiteLLM wheel, without importing it.

Run: python scripts/update_model_metadata.py [version]
The runtime loader fetches the current upstream map; this snapshot is its fallback.
"""

import gzip
import hashlib
import io
import json
import sys
import urllib.request
import zipfile
from pathlib import Path


def main(version="1.96.0"):
    with urllib.request.urlopen(f"https://pypi.org/pypi/litellm/{version}/json", timeout=30) as response:
        release = json.load(response)
    artifact = next(item for item in release["urls"] if item["filename"].endswith(".whl"))
    with urllib.request.urlopen(artifact["url"], timeout=60) as response:
        wheel = response.read()
    if hashlib.sha256(wheel).hexdigest() != artifact["digests"]["sha256"]:
        raise ValueError("Wheel checksum mismatch")
    with zipfile.ZipFile(io.BytesIO(wheel)) as archive:
        data = archive.read("litellm/model_prices_and_context_window_backup.json")
        license_path = next(name for name in archive.namelist() if name.endswith("/LICENSE") and ".dist-info/" in name)
        license_text = archive.read(license_path)
    if not isinstance(json.loads(data), dict):
        raise ValueError("Metadata must be a JSON object")
    target = Path(__file__).resolve().parents[1] / "dspy/clients/model_metadata"
    target.mkdir(parents=True, exist_ok=True)
    (target / "snapshot.json.gz").write_bytes(gzip.compress(data, mtime=0))
    (target / "LICENSE").write_bytes(license_text)
    (target / "provenance.json").write_text(json.dumps({
        "source": artifact["url"], "version": version,
        "wheel_sha256": artifact["digests"]["sha256"],
        "snapshot_sha256": hashlib.sha256(data).hexdigest(),
    }, indent=2) + "\n")
    print(f"Updated {target} from LiteLLM {version}")


if __name__ == "__main__":
    main(*sys.argv[1:])
