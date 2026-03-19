x-vec Tutorial
==============

This page introduces the core concepts behind x-vec and walks through a complete
end-to-end example. See :doc:`install` first if you haven't installed the SDK yet.
If you prefer the CLI workflow, see :doc:`cli_quickstart`.


How it works
------------

XTrace stores your documents as encrypted chunks inside a **knowledge base**. Before
anything leaves your machine, the SDK does two things: it AES-encrypts the chunk content
using a key derived from your passphrase, and it encodes the embedding vector into a binary
format and encrypts it with Paillier homomorphic encryption.

When you query, the query vector is encrypted the same way and sent to the server.
XTrace computes nearest-neighbor Hamming distances directly on the ciphertexts — without
decrypting them — and returns the closest chunk IDs. You decrypt the content locally.
The server never sees your documents, your vectors, or your query intent, even during active
search.


Core concepts
-------------

Five objects make up the SDK. Understanding how they relate to each other makes everything else
straightforward.

**ExecutionContext** is your private cryptographic state. It holds a Paillier key pair (for
encrypting vectors) and an AES key (for encrypting content), all locked by a passphrase.
Every chunk you store and every query you run must use the same execution context.
Create it once and save it — losing it means losing the ability to decrypt anything stored
with it.

**Embedding** converts text to a binary vector. The output dimension (``embed_len``) must
match the value you set when creating the execution context. The SDK supports Sentence
Transformers, Ollama, and OpenAI; you can also convert existing float vectors with
``Embedding.float_2_bin``.

**XTraceIntegration** is the async HTTP client that communicates with XTrace. It
authenticates with your API key and org ID but never transmits plaintext data — only
ciphertexts.

**DataLoader** orchestrates encryption and upload. It accepts a list of chunk dicts, encrypts
each one using the execution context, and stores them in a knowledge base via
``XTraceIntegration``.

**Retriever** orchestrates search. It encrypts a query vector, asks XTrace to compute Hamming
distances over all stored ciphertexts, fetches the top-k chunk IDs, and decrypts the results.

A **knowledge base** (KB) is a namespace on XTrace where your encrypted chunks live. Create
one from the XTrace dashboard or with ``xtrace kb create-kb``.


Prerequisites
-------------

You need an XTrace account with an API key, an org ID, and at least one knowledge base.
Create a knowledge base from the dashboard, or via the CLI after running ``xtrace init``:

.. code-block:: bash

    xtrace kb create-kb my-kb


Step 1 — Create an execution context
-------------------------------------

The execution context is the cryptographic core of the system. Create it once and save it to
disk. In future sessions you reload it with your passphrase instead of generating a new one.

.. code-block:: python

    from xtrace_sdk.x_vec.utils.execution_context import ExecutionContext

    ctx = ExecutionContext.create(
        passphrase="your-secret-passphrase",
        homomorphic_client_type="paillier_lookup",  # fastest CPU option
        embedding_length=512,                        # must match your embedding model
        key_len=1024,                                # Paillier key size in bits (≥ 1024)
        path="data/exec_context",                    # saved to disk immediately
    )
    print("Context ID:", ctx.id)  # note this — you need it to reload from XTrace

To reload in a future session:

.. code-block:: python

    from xtrace_sdk.x_vec.utils.execution_context import ExecutionContext

    ctx = ExecutionContext.load_from_disk("your-secret-passphrase", "data/exec_context")

You can also back the context up to XTrace so you can restore it from any machine:

.. code-block:: python

    from xtrace_sdk.integrations.xtrace import XTraceIntegration

    xtrace = XTraceIntegration(org_id="your_org_id", api_key="your_api_key")
    await ctx.save_to_remote(xtrace)  # secret key is AES-encrypted before upload

    # Restore
    ctx = await ExecutionContext.load_from_remote("your-secret-passphrase", ctx.id, xtrace)


Step 2 — Set up embedding and connect to XTrace
-------------------------------------------------

The embedding model converts text to binary vectors. Use Sentence Transformers for a
fully local setup. See :doc:`embedding` for Ollama and OpenAI options.

.. code-block:: python

    from xtrace_sdk.x_vec.inference.embedding import Embedding
    from xtrace_sdk.integrations.xtrace import XTraceIntegration

    embed  = Embedding("sentence_transformer", "mixedbread-ai/mxbai-embed-large-v1", dim=512)
    xtrace = XTraceIntegration(org_id="your_org_id", api_key="your_api_key")

``XTraceIntegration`` reads ``XTRACE_API_KEY`` from the environment if ``api_key`` is omitted.


Step 3 — Encrypt and store documents
--------------------------------------

``DataLoader`` encrypts your chunks and sends them to a knowledge base. Each chunk is a dict
with a ``chunk_content`` string and an optional ``meta_data`` dict. Metadata is stored in
plaintext — see :doc:`metadata` for the field schema and privacy implications.

.. code-block:: python

    from xtrace_sdk.x_vec.data_loaders.loader import DataLoader

    loader = DataLoader(ctx, xtrace)

    docs = [
        {
            "chunk_content": "XTrace encrypts vectors with Paillier homomorphic encryption.",
            "meta_data": {"tag1": "user_123", "tag2": "intro", "facets": ["security"]},
        },
        {
            "chunk_content": "The server computes nearest-neighbor search on ciphertexts.",
            "meta_data": {"tag1": "user_123", "tag2": "intro", "facets": ["search"]},
        },
    ]

    vectors = [embed.bin_embed(d["chunk_content"]) for d in docs]  # coroutines, awaited by loader
    index, db = await loader.load_data_from_memory(docs, vectors)
    await loader.dump_db(db, index=index, kb_id="your_kb_id")


Step 4 — Query
--------------

``Retriever`` encrypts a query vector, asks XTrace to find the nearest neighbors, then
decrypts and returns the matching chunks.

.. code-block:: python

    from xtrace_sdk.x_vec.retrievers.retriever import Retriever

    retriever = Retriever(ctx, xtrace)

    vec     = await embed.bin_embed("How does XTrace protect my data?")
    ids     = await retriever.nn_search_for_ids(vec, k=3, kb_id="your_kb_id")
    results = await retriever.retrieve_and_decrypt(ids, kb_id="your_kb_id")
    for r in results:
        print(r["chunk_content"])


Next steps
----------

- :doc:`embedding` — choose your embedding provider (Ollama, OpenAI, Sentence Transformers)
- :doc:`inference` — add LLM synthesis over retrieved results
- :doc:`metadata` — filter search results by metadata tags; understand the privacy trade-offs
- :doc:`xtrace` — full reference for ``XTraceIntegration``: chunk operations, context
  management, pagination
- :doc:`configuration` — lower-level configuration of ``PaillierClient``,
  ``ExecutionContext``, ``DataLoader``, and ``Retriever``
