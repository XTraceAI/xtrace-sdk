Configuration
=============

The XTrace Vec is designed to be highly configurable. You can configure it via environment variables or by passing parameters directly to classes and methods.


Cryptography Configuration
---------------------------

The SDK supports two production-ready homomorphic encryption schemes and one experimental scheme.

Both ``PaillierClient`` and ``PaillierLookupClient`` run on CPU by default.

.. note::

    **GPU backend (internal testing phase).** XTrace maintains a GPU-accelerated implementation of the
    homomorphic encryption layer that is approximately **20× faster** than the CPU path for large
    embedding collections. It is available as a compiled extension that slots into the same
    ``DEVICE=gpu`` switch — no application code changes required.

    The GPU implementation is not open-sourced at this time as it is under internal testing. Contact us at
    `liwen@xtrace.ai <mailto:liwen@xtrace.ai>`_ if you are interested in access.

Set ``DEVICE=gpu`` to activate the GPU backend once the compiled extension is in place.

**Paillier** — standard Paillier encryption:

.. code-block:: python

    from xtrace_sdk.x_vec.crypto.paillier_client import PaillierClient

    paillier_client = PaillierClient(embed_len=512, key_len=1024)

**Paillier-Lookup** — optimised Paillier variant using precomputed tables for faster encryption. Recommended for large collections:

.. code-block:: python

    from xtrace_sdk.x_vec.crypto.paillier_lookup_client import PaillierLookupClient

    paillier_client = PaillierLookupClient(embed_len=512, key_len=1024)

``embed_len`` must be strictly less than ``key_len``. Use at least ``key_len=1024`` for security as it is bit-length for prime modulus.

.. note::

    **Goldwasser-Micali** (``xtrace_sdk.x_vec.crypto.goldwasser_micali_client``) is included for research
    purposes and is **experimental** — it is not supported by ``DataLoader`` or ``Retriever`` and
    should not be used in production.


Execution Context Configuration
---------------------------------

``ExecutionContext`` bundles the homomorphic client and AES encryption under a single passphrase-protected object. Create one with:

.. code-block:: python

    from xtrace_sdk.x_vec.utils.execution_context import ExecutionContext
    from xtrace_sdk.x_vec.crypto.paillier_client import PaillierClient

    paillier_client = PaillierClient(embed_len=512, key_len=1024)
    execution_context = ExecutionContext(
        homomorphic_client=paillier_client,
        passphrase="your-secret-passphrase",
    )

.. note::

    The passphrase encrypts the secret homomorphic key at rest. There is no way
    to recover it through the SDK — manage it securely in your application.

You can persist and reload the execution context:

.. code-block:: python

    # Save to disk
    execution_context.save_to_disk("data/exec_context")

    # Load from disk
    execution_context = ExecutionContext.load_from_disk("your-secret-passphrase", "data/exec_context")

Or store/load it remotely via XTrace:

.. code-block:: python

    from xtrace_sdk.integrations.xtrace import XTraceIntegration
    xtrace = XTraceIntegration(org_id="your_org_id", api_key="your_api_key")

    # Save to remote
    await execution_context.save_to_remote(xtrace)

    # Load from remote
    execution_context = await ExecutionContext.load_from_remote(
        "your-secret-passphrase", "ctx_id", xtrace
    )

Each ``ExecutionContext`` has a unique ``id`` attribute you can use to reference it later.

You can also use ``ExecutionContext.create`` to generate and save in one call:

.. code-block:: python

    ctx = ExecutionContext.create(
        passphrase="your-secret-passphrase",
        homomorphic_client_type="paillier",
        embedding_length=512,
        key_len=1024,
        path="data/exec_context",   # optional: save immediately
    )


DataLoader Configuration
--------------------------

``DataLoader`` requires an execution context and an XTrace integration instance:

.. code-block:: python

    from xtrace_sdk.x_vec.data_loaders.loader import DataLoader
    from xtrace_sdk.integrations.xtrace import XTraceIntegration

    xtrace = XTraceIntegration(org_id="your_org_id", api_key="your_api_key")
    data_loader = DataLoader(execution_context, xtrace)

To reconstruct a ``DataLoader`` from a saved execution context:

.. code-block:: python

    from xtrace_sdk.x_vec.utils.execution_context import ExecutionContext

    ctx = ExecutionContext.load_from_disk("your-secret-passphrase", "data/exec_context")
    data_loader = DataLoader(ctx, xtrace)


Retriever Configuration
--------------------------

``Retriever`` mirrors the ``DataLoader`` setup. Pass ``parallel=True`` to decode Hamming distances using multiprocessing (useful for large KBs):

.. code-block:: python

    from xtrace_sdk.x_vec.retrievers.retriever import Retriever

    retriever = Retriever(execution_context, xtrace)

    # parallel decoding mode
    retriever = Retriever(execution_context, xtrace, parallel=True)


Environment Variables
----------------------

The following environment variables are read automatically:

.. list-table::
   :header-rows: 1

   * - Variable
     - Description
   * - ``XTRACE_API_KEY``
     - XTrace API key (used by ``XTraceIntegration`` when ``api_key`` is not passed explicitly).
   * - ``XTRACE_ORG_ID``
     - Organisation ID (used by ``XTraceIntegration`` when ``org_id`` is not passed explicitly).
   * - ``XTRACE_API_URL``
     - API base URL — defaults to ``https://api.production.xtrace.ai``.
   * - ``XTRACE_EXECUTION_CONTEXT_PATH``
     - Default path to a saved execution context.
   * - ``INFERENCE_API_KEY``
     - API key for your inference provider (OpenAI, Redpill, etc.).
