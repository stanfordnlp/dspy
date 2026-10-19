"""Build the hash-qualified Linux x64 bundle in a disposable Alpine rootfs.

Requires Linux user/mount namespaces and bwrap, or Docker. Never runs apk on
the host. Downloads occur only in this explicit build-time command.
"""
import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import tarfile
import tempfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = json.loads((ROOT / "manifest.json").read_text())
ROOTFS_URL = "https://dl-cdn.alpinelinux.org/alpine/v3.24/releases/x86_64/alpine-minirootfs-3.24.2-x86_64.tar.gz"
ROOTFS_SHA256 = "c5ca053cfe1d85c5b96dff8b9bc57045f7f184a30ffb6b65776409ca90388677"


def checked(path, expected):
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError(f"SHA256 mismatch: {path}")


def download(url, path, expected):
    urllib.request.urlretrieve(url, path)
    checked(path, expected)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--engine", choices=["docker", "namespace"], default="docker")
    args = parser.parse_args()
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        raise SystemExit("This recipe qualifies Linux x86_64 only")
    bw = MANIFEST["bubblewrap"]
    with tempfile.TemporaryDirectory(prefix="mxc-bwrap-") as td:
        work = Path(td)
        build = work / "build"
        build.mkdir()
        rootfs = work / "rootfs.tar.gz"
        download(ROOTFS_URL, rootfs, ROOTFS_SHA256)
        for name, key in [("bubblewrap-0.13.1.tar.xz", "source"), ("libcap-2.78.tar.xz", "libcap_source")]:
            download(bw[key + "_url"], build / name, bw[key + "_sha256"])
        for name in ("build-bwrap.sh", "apk-packages.txt"):
            shutil.copyfile(ROOT / "tools" / name, build / name)
        if args.engine == "docker":
            # Import the checked rootfs, not a mutable image tag.
            image = subprocess.check_output(["docker", "import", str(rootfs)], text=True).strip()
            try:
                command = (f"trap 'chown -R {os.getuid()}:{os.getgid()} /build' EXIT; "
                           "sh /build/build-bwrap.sh")
                subprocess.run(["docker", "run", "--rm", "--platform", "linux/amd64", "-e", "PATH=/usr/sbin:/usr/bin:/sbin:/bin", "-v",
                                f"{build}:/build", image, "sh", "-c", command], check=True)
            finally:
                subprocess.run(["docker", "image", "rm", image], check=True)
        else:
            root = work / "root"
            root.mkdir()
            # Extraction is confined to a fresh user namespace; rootfs is hash-checked.
            subprocess.run(["unshare", "-Ur", "tar", "--no-same-owner", "-xzf", str(rootfs), "-C", str(root)], check=True)
            shutil.copyfile("/etc/resolv.conf", root / "etc/resolv.conf")
            subprocess.run(["bwrap", "--unshare-user", "--uid", "0", "--gid", "0", "--unshare-pid",
                            "--setenv", "PATH", "/usr/sbin:/usr/bin:/sbin:/bin",
                            "--bind", str(root), "/", "--bind", str(build), "/build",
                            "--ro-bind", "/etc/ssl/certs", "/etc/ssl/certs", "--proc", "/proc", "--dev", "/dev",
                            "sh", "/build/build-bwrap.sh"], check=True)
        package = work / "dspy_mxc_runtime"
        (package / "bin").mkdir(parents=True)
        (package / "sources").mkdir()
        (package / "licenses").mkdir()
        shutil.copyfile(build / "build-bwrap/bwrap", package / "bin/bwrap")
        for name in ("bubblewrap-0.13.1.tar.xz", "libcap-2.78.tar.xz", "build-packages.txt", "build-bwrap.sh", "apk-packages.txt"):
            shutil.copyfile(build / name, package / "sources" / name)
        shutil.copyfile(build / "bubblewrap-0.13.1/COPYING", package / "licenses/bubblewrap-COPYING")
        with tarfile.open(build / "libcap-2.78.tar.xz") as source:
            (package / "licenses/libcap-License").write_bytes(source.extractfile("libcap-2.78/License").read())
        # musl's installed copyright is supplied by the matching upstream source.
        musl = build / "musl.tar.gz"
        download("https://musl.libc.org/releases/musl-1.2.6.tar.gz", musl, bw["musl_source_sha256"])
        with tarfile.open(musl) as source:
            (package / "licenses/musl-COPYRIGHT").write_bytes(source.extractfile("musl-1.2.6/COPYRIGHT").read())
        for name, digest in {"bin/bwrap": bw["qualified_x86_64_sha256"], **bw["bundle_files"]}.items():
            checked(package / name, digest)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with tarfile.open(args.output, "w:gz") as bundle:
            bundle.add(package, arcname="dspy_mxc_runtime")
        print(args.output.resolve())


if __name__ == "__main__":
    main()
