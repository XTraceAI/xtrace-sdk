Using the XTrace Managed Service
=================================

``XTraceIntegration`` is the single entry point for all communication with the XTrace API. It handles chunk storage, encrypted Hamming distance computation, metadata search, and execution context management.


Security Model
--------------

Understanding what XTrace can and cannot see is the core guarantee of this SDK.

**What XTrace cannot see**

.. list-table::
   :header-rows: 1

   * - Item
     - How it is protected
   * - Chunk content
     - AES-encrypted on the client before upload. The server stores only ciphertext.
       The AES key is supplied by a key provider (passphrase-derived or AWS KMS envelope encryption)
       and never leaves your environment.
   * - Embedding vectors
     - Encrypted with Paillier homomorphic encryption on the client before being sent.
       The server computes nearest-neighbor Hamming distances directly on the ciphertexts
       — it never sees the original binary vectors, not even during search.
   * - Query vectors
     - Same as stored vectors: homomorphically encrypted on the client before the
       search request is transmitted.
   * - Paillier private key
     - Never transmitted in plaintext. See *Execution Context* below.

**What XTrace can see**

.. list-table::
   :header-rows: 1

   * - Item
     - Notes
   * - Metadata tags (``tag1``–``tag5``, ``facets``)
     - Stored and indexed in plaintext. See :doc:`metadata` for mitigation strategies.
   * - Paillier public key
     - Stored in plaintext by design — public keys are not secret.
   * - Collection structure
     - Number of chunks, their ``kb_id`` assignments, and chunk-level metadata are visible.
   * - Encrypted blobs
     - The server stores AES ciphertexts and Paillier ciphertexts, but cannot decrypt them.


Execution Context
-----------------

The ``ExecutionContext`` bundles the Paillier key pair and the AES key under a single
protected object. A **key provider** controls how the AES key is generated and protected.

The SDK ships two providers:

* ``PassphraseKeyProvider`` — derives a 256-bit AES key from a passphrase via scrypt.
  Simple and self-contained — no cloud dependencies.
* ``AWSKMSKeyProvider`` — generates a data encryption key (DEK) via AWS KMS envelope
  encryption. The DEK is never stored in plaintext — only the KMS-wrapped ciphertext
  (EDEK) is persisted.

When stored remotely on XTrace (``execution_context.save_to_remote(xtrace)``), only the
following is transmitted:

* The **public key** in plaintext (intentional — it is not secret).
* The **secret key** encrypted with the AES key supplied by the key provider.
* Non-sensitive configuration (key length, embedding length).

Neither your passphrase nor your KMS plaintext DEK are ever transmitted. Without the
corresponding key provider, the stored blob cannot be decrypted — XTrace cannot
recover the secret key and cannot decrypt your chunk content or vectors.

See :doc:`configuration` for the full key provider reference.


.. rubric:: Passphrase-based context

.. code-block:: python

    from xtrace_sdk.x_vec.utils.execution_context import ExecutionContext
    from xtrace_sdk.x_vec.crypto.key_provider import PassphraseKeyProvider

    provider = PassphraseKeyProvider("your-secret-passphrase")

    ctx = ExecutionContext.create(
        key_provider=provider,
        homomorphic_client_type="paillier_lookup",
        embedding_length=512,
        key_len=1024,
        path="data/exec_context",          # optional: save immediately
    )

    # Restore from disk
    ctx = ExecutionContext.load_from_disk("your-secret-passphrase", "data/exec_context")

    # Or save/restore via XTrace
    await ctx.save_to_remote(xtrace)
    ctx = await ExecutionContext.load_from_remote("your-secret-passphrase", ctx.id, xtrace)


.. rubric:: AWS KMS-based context

.. code-block:: python

    import boto3
    from xtrace_sdk.x_vec.utils.execution_context import ExecutionContext
    from xtrace_sdk.x_vec.crypto.key_provider import AWSKMSKeyProvider

    kms = boto3.client("kms")
    provider = AWSKMSKeyProvider.create(kms, "alias/xtrace")

    ctx = ExecutionContext.create(
        key_provider=provider,
        homomorphic_client_type="paillier_lookup",
        embedding_length=512,
        key_len=1024,
    )

    # Restore: reconstruct the provider from the stored EDEK
    import base64, json
    with open("data/exec_context") as f:
        edek = base64.b64decode(json.load(f)["wrapped_key"])
    provider = AWSKMSKeyProvider.from_wrapped(edek, kms_client=kms, key_id="alias/xtrace")
    ctx = ExecutionContext.load_from_disk(path="data/exec_context", key_provider=provider)


Connecting
----------

.. code-block:: python

    from xtrace_sdk.integrations.xtrace import XTraceIntegration

    xtrace = XTraceIntegration(
        org_id="your_org_id",
        api_key="your_api_key",         # or set XTRACE_API_KEY env var
        api_url="https://api.production.xtrace.ai",  # default
    )

Use it as an async context manager to manage the HTTP session automatically:

.. code-block:: python

    async with XTraceIntegration(org_id="your_org_id", api_key="your_api_key") as xtrace:
        ...


Loading Data
------------

Use ``DataLoader`` with an ``XTraceIntegration`` to encrypt and store documents:

.. code-block:: python

    from xtrace_sdk.x_vec.data_loaders.loader import DataLoader
    from xtrace_sdk.x_vec.inference.embedding import Embedding

    embed = Embedding("sentence_transformer", "mixedbread-ai/mxbai-embed-large-v1", 512)
    data_loader = DataLoader(execution_context, xtrace)

    # Provide your documents as a list of chunk dicts
    collection = [
        {
            "chunk_content": "...",
            "meta_data": {
                "tag1": "user_123",
                "tag2": "my-project",
                "tag5": "doc1.txt",
                "facets": ["finance", "q1"],
            },
        },
    ]
    vectors = [embed.bin_embed(item["chunk_content"]) for item in collection]  # coroutines, awaited by loader
    index, db = await data_loader.load_data_from_memory(collection, vectors)
    await data_loader.dump_db(db, index=index, kb_id="your_kb_id")


Querying
--------

Use ``Retriever`` to run encrypted nearest-neighbor search:

.. code-block:: python

    from xtrace_sdk.x_vec.retrievers.retriever import Retriever
    from xtrace_sdk.x_vec.inference.embedding import Embedding

    embed = Embedding("ollama", "mxbai-embed-large", 1024)
    retriever = Retriever(execution_context, xtrace)

    vec     = await embed.bin_embed("What is XTrace?")
    ids     = await retriever.nn_search_for_ids(vec, k=3, kb_id="your_kb_id")
    results = await retriever.retrieve_and_decrypt(ids, kb_id="your_kb_id")


Metadata Search
---------------

See :doc:`metadata` for the full filter syntax, operator reference, and performance guidance.

Search and filter chunks by metadata without running a vector query:

.. code-block:: python

    results = await xtrace.meta_search(
        kb_id="your_kb_id",
        meta_filter={"tag1": "user_123", "tag2": "my-project"},
        context_id=execution_context.id,
    )

    # Paginated variant
    page = await xtrace.meta_search_paginated(
        kb_id="your_kb_id",
        context_id=execution_context.id,
        meta_filter={"tag1": "user_123"},
        limit=20,
        offset=0,
        return_content=True,   # include encrypted chunk_content in results
    )


Chunk Operations
----------------

.. code-block:: python

    # Delete specific chunks
    await xtrace.delete_chunks(chunk_ids=[0, 1, 2], kb_id="your_kb_id")

    # Delete by metadata filter
    await xtrace.delete_chunks_by_meta(
        kb_id="your_kb_id",
        context_id=execution_context.id,
        meta_filter={"tag1": "user_123", "tag2": "old-project"},
    )

    # Patch metadata fields on matching chunks
    await xtrace.patch_chunks_by_meta(
        kb_id="your_kb_id",
        context_id=execution_context.id,
        meta_filter={"tag1": "user_123", "facets": {"$contains": "draft"}},
        patch={"facets": ["published"]},
    )


Execution Context Management
-----------------------------

Use the :class:`~xtrace_sdk.x_vec.utils.execution_context.ExecutionContext` helpers rather than
calling the low-level API directly:

.. code-block:: python

    # Save to XTrace (preferred — secret key is encrypted by the key provider before upload)
    ctx_id = await execution_context.save_to_remote(xtrace)

    # Restore from XTrace (passphrase-based)
    execution_context = await ExecutionContext.load_from_remote(
        "your-secret-passphrase", ctx_id, xtrace
    )

    # Restore from XTrace (KMS-based — reconstruct the provider first)
    provider = AWSKMSKeyProvider.from_wrapped(edek, kms_client=kms, key_id="alias/xtrace")
    execution_context = await ExecutionContext.load_from_remote(
        key_provider=provider, context_id=ctx_id, integration=xtrace
    )

    # List all stored context IDs for your org
    ctx_ids = await xtrace.list_exec_contexts()

    # Delete a context
    await xtrace.delete_exec_context(ctx_id)


Notes
-----

* ``kb_id`` and ``org_id`` are available from the XTrace dashboard.
* Metadata fields use the fixed schema: ``tag1``–``tag5`` and ``facets``.
  See :doc:`metadata` for field semantics and operator reference.
* The ``concurrent=True`` flag on ``store_db`` enables parallel batch ingestion — useful for large loads.
