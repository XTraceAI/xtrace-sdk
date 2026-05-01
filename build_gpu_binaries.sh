#!/bin/bash
#
# Build the Paillier GPU and Paillier-Lookup GPU pybind11 extensions inside a
# pinned nvidia/cuda Docker image. Produces the two in-tree .so files imported
# by the normal Python packages under `xtrace_sdk.x_vec.crypto`:
#
#   src/xtrace_sdk/x_vec/crypto/paillier_gpu_ext/paillier_GPU_client*.so
#   src/xtrace_sdk/x_vec/crypto/paillier_lookup_gpu_ext/paillier_GPU_lookup_client*.so
#
# Overridable via env vars:
#   PYTHON_VERSION       Python minor version          (default: 3.11)
#   PYTHON_FULL_VERSION  Full python-build-standalone version  (default: 3.11.15)
#   PYTHON_STANDALONE_TAG  python-build-standalone release tag (default: 20260414)
#   SMS             Space-separated GPU SM list        (default: "70 75 80 86 89 90 90a")
#   KEY_BITS        Paillier key size                  (default: 1024)
#   ALPHA_LEN       Lookup variant alpha length        (default: 280)
#   CUDA_IMAGE      nvidia/cuda devel image            (default: matched to host Ubuntu, see below)
#   CGBN_REPO_URL   CGBN header-only lib source        (default: NVlabs/CGBN on GitHub)
#   PYBIND11_REPO_URL  pybind11 header-only lib source (default: pybind/pybind11 on GitHub)
#
# Python is fetched from astral-sh/python-build-standalone (GitHub-hosted
# tarball) instead of distro packages, so the build does not depend on
# Launchpad/PPAs being reachable from the build host.
#
# Only nvcc is needed to compile — no GPU is required, so this runs on a
# standard CI runner (or Apple Silicon via --platform linux/amd64).
#
set -euo pipefail

PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
PYTHON_FULL_VERSION="${PYTHON_FULL_VERSION:-3.11.15}"
PYTHON_STANDALONE_TAG="${PYTHON_STANDALONE_TAG:-20260414}"
SMS="${SMS:-70 75 80 86 89 90 90a}"
KEY_BITS="${KEY_BITS:-1024}"
ALPHA_LEN="${ALPHA_LEN:-280}"
# Pick a CUDA devel base image whose toolchain (gcc/libstdc++) matches the
# host's, so the resulting .so loads on the host without GLIBCXX_* mismatches.
# Override CUDA_IMAGE explicitly to target a different deployment baseline.
default_cuda_image() {
    local host_id="" host_ver=""
    if [ -r /etc/os-release ]; then
        # shellcheck source=/dev/null
        host_id="$(. /etc/os-release && echo "${ID:-}")"
        host_ver="$(. /etc/os-release && echo "${VERSION_ID:-}")"
    fi
    if [ "$host_id" = "ubuntu" ]; then
        case "$host_ver" in
            20.04) echo "nvidia/cuda:12.4.1-devel-ubuntu20.04"; return ;;
            22.04) echo "nvidia/cuda:12.4.1-devel-ubuntu22.04"; return ;;
            24.04) echo "nvidia/cuda:12.6.3-devel-ubuntu24.04"; return ;;
        esac
    fi
    # Non-Ubuntu host (Debian, RHEL, macOS, etc.): default to the oldest still
    # supported Ubuntu base for broad libstdc++ compatibility.
    echo "nvidia/cuda:12.4.1-devel-ubuntu22.04"
}
CUDA_IMAGE="${CUDA_IMAGE:-$(default_cuda_image)}"
CGBN_REPO_URL="${CGBN_REPO_URL:-https://github.com/NVlabs/CGBN.git}"
PYBIND11_REPO_URL="${PYBIND11_REPO_URL:-https://github.com/pybind/pybind11.git}"

PYTHON_STANDALONE_URL="https://github.com/astral-sh/python-build-standalone/releases/download/${PYTHON_STANDALONE_TAG}/cpython-${PYTHON_FULL_VERSION}+${PYTHON_STANDALONE_TAG}-x86_64-unknown-linux-gnu-install_only.tar.gz"

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
CRYPTO_DIR="$REPO_ROOT/src/xtrace_sdk/x_vec/crypto"

if [ ! -d "$CRYPTO_DIR/paillier_gpu_ext" ] || \
   [ ! -d "$CRYPTO_DIR/paillier_lookup_gpu_ext" ]; then
    echo "Error: expected GPU source dirs under $CRYPTO_DIR" >&2
    exit 1
fi

# Fall back through SUDO_UID/GID so `sudo ./build_gpu_binaries.sh` still chowns
# artifacts back to the invoking user, not root.
HOST_UID="${SUDO_UID:-$(id -u)}"
HOST_GID="${SUDO_GID:-$(id -g)}"

echo "Building Paillier GPU extensions"
echo "  Python:    $PYTHON_FULL_VERSION (python-build-standalone $PYTHON_STANDALONE_TAG)"
echo "  CUDA img:  $CUDA_IMAGE"
echo "  SMs:       $SMS"
echo "  KEY_BITS:  $KEY_BITS"
echo "  ALPHA_LEN: $ALPHA_LEN  (lookup variant only)"
echo "  Output:    $CRYPTO_DIR/paillier{,_lookup}_gpu_ext/*.so"
echo

docker run --rm \
    --platform linux/amd64 \
    --env DEBIAN_FRONTEND=noninteractive \
    --env PYTHON_VERSION="$PYTHON_VERSION" \
    --env PYTHON_STANDALONE_URL="$PYTHON_STANDALONE_URL" \
    --env SMS="$SMS" \
    --env KEY_BITS="$KEY_BITS" \
    --env ALPHA_LEN="$ALPHA_LEN" \
    --env CGBN_REPO_URL="$CGBN_REPO_URL" \
    --env PYBIND11_REPO_URL="$PYBIND11_REPO_URL" \
    --env HOST_UID="$HOST_UID" \
    --env HOST_GID="$HOST_GID" \
    -v "$REPO_ROOT:/build/repo" \
    -w /build/repo \
    "$CUDA_IMAGE" \
    /bin/bash -c '
        set -euo pipefail

        apt-get update
        apt-get install -y --no-install-recommends \
            ca-certificates git build-essential libgmp-dev curl xz-utils

        # Fetch a self-contained CPython build (no PPA / Launchpad needed).
        curl -fsSL "$PYTHON_STANDALONE_URL" -o /tmp/python.tar.gz
        mkdir -p /opt
        tar -xzf /tmp/python.tar.gz -C /opt   # creates /opt/python
        rm /tmp/python.tar.gz
        ln -sf "/opt/python/bin/python${PYTHON_VERSION}" /usr/local/bin/python3
        ln -sf "/opt/python/bin/python${PYTHON_VERSION}" "/usr/local/bin/python${PYTHON_VERSION}"
        python3 --version

        # Materialise the include/{pybind11,cgbn} layout the Makefiles expect.
        INCDIR=/build/include
        mkdir -p "$INCDIR"
        git clone --depth=1 "$PYBIND11_REPO_URL" /opt/pybind11
        ln -sfn /opt/pybind11/include/pybind11 "$INCDIR/pybind11"
        git clone --depth=1 "$CGBN_REPO_URL" /opt/CGBN
        ln -sfn /opt/CGBN/include/cgbn "$INCDIR/cgbn"

        build_one() {
            local dir="$1"
            shift
            echo "=== Building $dir ==="
            cd "/build/repo/src/xtrace_sdk/x_vec/crypto/$dir"
            make clean || true
            make INCDIR="$INCDIR" SMS="$SMS" KEY_BITS="$KEY_BITS" "$@"
            ls -lh ./*.so
        }

        build_one paillier_gpu_ext
        build_one paillier_lookup_gpu_ext ALPHA_LEN="$ALPHA_LEN"

        chown -R "$HOST_UID:$HOST_GID" \
            /build/repo/src/xtrace_sdk/x_vec/crypto/paillier_gpu_ext \
            /build/repo/src/xtrace_sdk/x_vec/crypto/paillier_lookup_gpu_ext

        echo "Done."
    '

echo
echo "Artifacts:"
ls -lh "$CRYPTO_DIR/paillier_gpu_ext/"*.so \
       "$CRYPTO_DIR/paillier_lookup_gpu_ext/"*.so 2>/dev/null || \
       echo "  (no .so produced — check output above)"
