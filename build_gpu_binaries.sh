#!/bin/bash
#
# Build the Paillier GPU and Paillier-Lookup GPU pybind11 extensions inside a
# pinned nvidia/cuda Docker image. Produces the two .so files that
# xtrace_sdk.x_vec.crypto._gpu_loader expects to find in-tree:
#
#   src/xtrace_sdk/x_vec/crypto/paillier-GPU-client/paillier_GPU_client*.so
#   src/xtrace_sdk/x_vec/crypto/paillier-GPU-lookup-client/paillier_GPU_lookup_client*.so
#
# Overridable via env vars:
#   PYTHON_VERSION  Python minor version (default: 3.11)
#   SMS             Space-separated GPU SM list        (default: "70 75 80 86 89 90 90a")
#   KEY_BITS        Paillier key size                  (default: 1024)
#   ALPHA_LEN       Lookup variant alpha length        (default: 280)
#   CUDA_IMAGE      nvidia/cuda devel image            (default: 12.4.0-devel-ubuntu22.04)
#   CGBN_REPO_URL   CGBN header-only lib source        (default: NVlabs/CGBN on GitHub)
#
# Only nvcc is needed to compile — no GPU is required, so this runs on a
# standard CI runner (or Apple Silicon via --platform linux/amd64).
#
set -euo pipefail

PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
SMS="${SMS:-70 75 80 86 89 90 90a}"
KEY_BITS="${KEY_BITS:-1024}"
ALPHA_LEN="${ALPHA_LEN:-280}"
CUDA_IMAGE="${CUDA_IMAGE:-nvidia/cuda:12.4.0-devel-ubuntu22.04}"
CGBN_REPO_URL="${CGBN_REPO_URL:-https://github.com/NVlabs/CGBN.git}"

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
CRYPTO_DIR="$REPO_ROOT/src/xtrace_sdk/x_vec/crypto"

if [ ! -d "$CRYPTO_DIR/paillier-GPU-client" ] || \
   [ ! -d "$CRYPTO_DIR/paillier-GPU-lookup-client" ]; then
    echo "Error: expected GPU source dirs under $CRYPTO_DIR" >&2
    exit 1
fi

# Fall back through SUDO_UID/GID so `sudo ./build_gpu_binaries.sh` still chowns
# artifacts back to the invoking user, not root.
HOST_UID="${SUDO_UID:-$(id -u)}"
HOST_GID="${SUDO_GID:-$(id -g)}"

echo "Building Paillier GPU extensions"
echo "  Python:    $PYTHON_VERSION"
echo "  CUDA img:  $CUDA_IMAGE"
echo "  SMs:       $SMS"
echo "  KEY_BITS:  $KEY_BITS"
echo "  ALPHA_LEN: $ALPHA_LEN  (lookup variant only)"
echo "  Output:    $CRYPTO_DIR/paillier-GPU-{,lookup-}client/*.so"
echo

docker run --rm \
    --platform linux/amd64 \
    --env DEBIAN_FRONTEND=noninteractive \
    --env PYTHON_VERSION="$PYTHON_VERSION" \
    --env SMS="$SMS" \
    --env KEY_BITS="$KEY_BITS" \
    --env ALPHA_LEN="$ALPHA_LEN" \
    --env CGBN_REPO_URL="$CGBN_REPO_URL" \
    --env HOST_UID="$HOST_UID" \
    --env HOST_GID="$HOST_GID" \
    -v "$REPO_ROOT:/build/repo" \
    -w /build/repo \
    "$CUDA_IMAGE" \
    /bin/bash -c '
        set -euo pipefail

        apt-get update
        apt-get install -y --no-install-recommends \
            software-properties-common gnupg2 ca-certificates \
            git build-essential libgmp-dev curl

        add-apt-repository -y ppa:deadsnakes/ppa
        apt-get update
        apt-get install -y --no-install-recommends \
            "python${PYTHON_VERSION}-dev" \
            "python${PYTHON_VERSION}-venv"

        update-alternatives --install /usr/bin/python3 python3 "/usr/bin/python${PYTHON_VERSION}" 1

        "python${PYTHON_VERSION}" -m ensurepip --upgrade
        "python${PYTHON_VERSION}" -m pip install --no-cache-dir pybind11

        # Materialise the ../include/{pybind11,cgbn} layout the Makefiles expect.
        INCDIR=/build/include
        mkdir -p "$INCDIR"
        PYBIND_INC="$("python${PYTHON_VERSION}" -c "import pybind11; print(pybind11.get_include())")"
        ln -sfn "$PYBIND_INC/pybind11" "$INCDIR/pybind11"

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

        build_one paillier-GPU-client
        build_one paillier-GPU-lookup-client ALPHA_LEN="$ALPHA_LEN"

        chown -R "$HOST_UID:$HOST_GID" \
            /build/repo/src/xtrace_sdk/x_vec/crypto/paillier-GPU-client \
            /build/repo/src/xtrace_sdk/x_vec/crypto/paillier-GPU-lookup-client

        echo "Done."
    '

echo
echo "Artifacts:"
ls -lh "$CRYPTO_DIR/paillier-GPU-client/"*.so \
       "$CRYPTO_DIR/paillier-GPU-lookup-client/"*.so 2>/dev/null || \
       echo "  (no .so produced — check output above)"
