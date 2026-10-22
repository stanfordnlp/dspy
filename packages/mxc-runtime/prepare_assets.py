"""Explicit build-time asset preparation. Not imported by the installed package."""
import argparse
import hashlib
import json
import shutil
import tarfile
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MANIFEST = json.loads((ROOT / "manifest.json").read_text())


def checked(data, digest):
    if hashlib.sha256(data).hexdigest() != digest:
        raise ValueError("Asset SHA256 mismatch")
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", required=True, choices=["diagnostic", *MANIFEST["targets"]])
    parser.add_argument("--archive", type=Path, help="Previously downloaded MXC ZIP (otherwise download at build time)")
    parser.add_argument("--mxc-license", type=Path, help="Offline copy of upstream LICENSE.md")
    parser.add_argument("--bwrap-bundle", type=Path, help="Source/build/license bundle described in README")
    args = parser.parse_args()
    package = ROOT / "dspy_mxc_runtime"
    # Never let stale platform assets leak into a subsequent diagnostic/other-platform wheel.
    for name in ("bin", "sources", "licenses"):
        shutil.rmtree(package / name, ignore_errors=True)
    (package / "payload.json").unlink(missing_ok=True)
    shutil.rmtree(ROOT / "build", ignore_errors=True)
    if args.target == "diagnostic":
        payload = {"target": "diagnostic", "system": None, "machine": None, "wheel_platform": "any"}
    else:
        target = MANIFEST["targets"][args.target]
        payload = dict(target, target=args.target, executor=Path(target["member"]).name)
        if args.archive:
            archive = args.archive
        else:
            archive = ROOT / "assets/mxc-release-binaries.zip"
            archive.parent.mkdir(exist_ok=True)
            urllib.request.urlretrieve(MANIFEST["url"], archive)
        # Stream the large ZIP hash instead of reading it twice into memory.
        digest = hashlib.sha256()
        with archive.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        if digest.hexdigest() != MANIFEST["sha256"]:
            raise ValueError("MXC ZIP SHA256 mismatch")
        (package / "bin").mkdir()
        (package / "licenses").mkdir()
        with zipfile.ZipFile(archive) as z:
            data = checked(z.read(target["member"]), target["sha256"])
            executor = package / "bin" / payload["executor"]
            executor.write_bytes(data)  # No strip, patch, re-sign or binary rewriting.
            executor.chmod(0o755)
            sbom = target["member"].split("/")[0] + "/_manifest/spdx_2.2/manifest.spdx.json"
            (package / "licenses/MXC-upstream.spdx.json").write_bytes(z.read(sbom))
        license_data = (args.mxc_license.read_bytes() if args.mxc_license else
                        urllib.request.urlopen(MANIFEST["license_url"]).read())
        (package / "licenses/MXC-LICENSE").write_bytes(checked(
            license_data, MANIFEST["license_sha256"]))
        if target["system"] == "Linux":
            bw = MANIFEST["bubblewrap"]
            bwrap_hash = bw.get("qualified_" + target["machine"] + "_sha256")
            if not args.bwrap_bundle or not bwrap_hash:
                raise ValueError("Linux requires a source/license bundle and a qualified static bwrap hash in manifest.json")
            with tarfile.open(args.bwrap_bundle) as bundle:
                names = {
                    "bin/bwrap": bwrap_hash,
                    "sources/bubblewrap-0.13.1.tar.xz": bw["source_sha256"],
                    "sources/libcap-2.78.tar.xz": bw["libcap_source_sha256"],
                    **bw["bundle_files"],
                }
                for name, expected in names.items():
                    stream = bundle.extractfile("dspy_mxc_runtime/" + name)
                    if stream is None:
                        raise ValueError(f"Missing bundle member {name}")
                    dest = package / name
                    dest.parent.mkdir(exist_ok=True)
                    dest.write_bytes(checked(stream.read(), expected))
            (package / "bin/bwrap").chmod(0o755)
            for name in ("build-bwrap.sh", "build_bundle.py", "apk-packages.txt"):
                shutil.copyfile(ROOT / "tools" / name, package / "sources" / name)
            shutil.copyfile(ROOT / "manifest.json", package / "sources/manifest.json")
    (package / "payload.json").write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
