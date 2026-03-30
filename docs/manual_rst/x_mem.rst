x-mem (Coming Soon)
====================

``xtrace_sdk.x_mem`` is the encrypted memory layer built on top of XTrace SDK.

It will allow you to give AI agents long-term, privacy-preserving memory backed
by a pluggable vector store — with ``x_vec`` (XTrace's encrypted vector DB) as
the first-class backend, and support for other vector stores such as Chroma,
Pinecone, and pgvector.

Planned features
----------------

- **Pluggable vector stores** — use ``x_vec`` for end-to-end encryption, or
  bring your own backend via the ``VectorStore`` protocol.
- **Memory read/write API** — simple ``remember`` / ``recall`` interface
  regardless of the backend.
- **Shared integrations** — reuses ``xtrace_sdk.integrations.xtrace`` so
  credentials and org context are configured once across the whole SDK.

Stay tuned for updates.
