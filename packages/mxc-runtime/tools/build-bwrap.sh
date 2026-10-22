#!/bin/sh
# Run in an isolated Alpine 3.24.2 build environment, NOT on the runtime host.
# /build contains the pinned bubblewrap source from manifest.json.
# No setuid installation, host policy changes or privileges for wheel consumers.
set -eu
# Pin build inputs; build_bundle.py also verifies the complete transitive
# inventory against manifest.json. Retired versions must fail, not float.
apk add --no-cache $(cat /build/apk-packages.txt)
cd /build
echo '0e85c8702248b8dd64144f594147553650cdd0e4c951690345faace79c217c4d  bubblewrap-0.13.1.tar.xz' | sha256sum -c -
tar -xf bubblewrap-0.13.1.tar.xz
meson setup build-bwrap bubblewrap-0.13.1 --buildtype=minsize \
    -Dprefer_static=true -Dc_link_args=-static -Dselinux=disabled \
    -Dman=disabled -Dbash_completion=disabled -Dzsh_completion=disabled -Dtests=false
ninja -C build-bwrap
strip --strip-unneeded build-bwrap/bwrap
file build-bwrap/bwrap
readelf -l build-bwrap/bwrap
readelf -d build-bwrap/bwrap
if readelf -l build-bwrap/bwrap | grep -q INTERP || readelf -d build-bwrap/bwrap | grep -q NEEDED; then
    echo 'bwrap must be a static ELF without a loader or shared dependencies' >&2
    exit 1
fi
build-bwrap/bwrap --version
apk info -vv > build-packages.txt
