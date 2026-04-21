============
Installation
============

Requires Python 3.11 or later.

Base install
------------

The base package includes the full x-vec SDK — encrypted vector storage, retrieval,
execution context management, and the Ollama and OpenAI embedding providers:

.. code-block:: bash

    pip install xtrace-ai-sdk


Optional extras
---------------

``[embedding]`` — Sentence Transformers
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Adds local embedding support via `Sentence Transformers <https://www.sbert.net>`_.
Models are downloaded from Hugging Face on first use.

Ollama and any OpenAI API-compatible embedding service (including OpenAI, Redpill,
and Anthropic) are supported out of the box in the base install — no extra flag needed.
Only install this extra if you want to use Sentence Transformers directly:

.. code-block:: bash

    pip install "xtrace-ai-sdk[embedding]"


``[cli]`` — Interactive CLI
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Adds the ``xtrace`` command-line interface — an interactive shell for managing
knowledge bases, loading data, and running queries without writing Python:

.. code-block:: bash

    pip install "xtrace-ai-sdk[cli]"

See :doc:`cli_quickstart` for setup and usage.


Combined
^^^^^^^^

Extras can be combined:

.. code-block:: bash

    pip install "xtrace-ai-sdk[embedding,cli]"


Install from source
-------------------

.. code-block:: bash

    git clone https://github.com/XTraceAI/xtrace-sdk.git
    cd xtrace-sdk
    pip install -e .

To include extras:

.. code-block:: bash

    pip install -e ".[embedding,cli]"


GPU acceleration (optional)
---------------------------

The Paillier and Paillier-Lookup homomorphic encryption clients ship with an
optional CUDA backend that is roughly **20× faster** than the CPU path on large
embedding collections. It is opt-in — the base install runs fully on CPU and
requires no GPU, no CUDA toolkit, and no Docker.

Prebuilt binaries are **not** currently published on PyPI. To enable the GPU
path, compile the pybind11 extensions from source using the included Docker-based
build script:

.. code-block:: bash

    git clone https://github.com/XTraceAI/xtrace-sdk.git
    cd xtrace-sdk
    ./build_gpu_binaries.sh

The script spins up an ``nvidia/cuda`` devel container, fetches ``pybind11`` and
``CGBN`` headers, and compiles two ``.so`` files that land in place under
``src/xtrace_sdk/x_vec/crypto/paillier-GPU-{,lookup-}client/``. The SDK's runtime
loader picks them up automatically on import. No NVIDIA GPU is required on the
*build* host — only ``nvcc``, which is provided by the Docker image.

Enable the GPU backend at runtime by setting the ``DEVICE`` environment variable:

.. code-block:: python

    import os
    os.environ["DEVICE"] = "gpu"

    from xtrace_sdk.x_vec.crypto.paillier_client import PaillierClient
    client = PaillierClient(embed_len=512, key_len=1024)   # instantiates the GPU backend

``DEVICE`` is read every time a client is instantiated, so it is safe to change
at runtime (for example, to run CPU and GPU clients side by side in the same
process).

Requirements
^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Component
     - Version
   * - Build host
     - Docker (any recent version). ~10 GB free disk for the CUDA devel image.
   * - Runtime host
     - NVIDIA driver **≥ 550** (CUDA 12.x runtime), with an NVIDIA GPU of
       compute capability 7.0–9.0 (V100, T4, A100, RTX 30/40, H100, etc.).

Build-time configuration
^^^^^^^^^^^^^^^^^^^^^^^^

All parameters are environment variables on the build script:

.. list-table::
   :header-rows: 1

   * - Variable
     - Default
     - Description
   * - ``PYTHON_VERSION``
     - ``3.11``
     - CPython minor version. Must match the Python you will import the SDK from
       (the ``.so`` is ABI-tagged per minor version).
   * - ``SMS``
     - ``70 75 80 86 89 90 90a``
     - Space-separated GPU SM targets. Default fat-binary covers V100, T4,
       A100, RTX 30/40, and H100 in a single file.
   * - ``KEY_BITS``
     - ``1024``
     - Paillier modulus size (bits). Must match the ``key_len`` passed to the
       client at runtime.
   * - ``ALPHA_LEN``
     - ``280``
     - Paillier-Lookup alpha length. Lookup variant only.
   * - ``CUDA_IMAGE``
     - ``nvidia/cuda:12.4.0-devel-ubuntu22.04``
     - Docker image used to compile. Swap for ``nvidia/cuda:11.x-devel-*`` if
       you need a CUDA 11 runtime.
   * - ``CGBN_REPO_URL``
     - NVlabs/CGBN on GitHub
     - Override if you are building behind an air-gapped mirror.

Example — build a CUDA 11 / Python 3.12 binary:

.. code-block:: bash

    PYTHON_VERSION=3.12 \
    CUDA_IMAGE=nvidia/cuda:11.8.0-devel-ubuntu22.04 \
    ./build_gpu_binaries.sh
