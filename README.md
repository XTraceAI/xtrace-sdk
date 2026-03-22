# XTrace SDK

XTrace is an encrypted vector and memory database. Your document content is AES-encrypted and your embedding vectors are homomorphically encrypted before they leave your machine, so the server stores and searches over ciphertexts without ever seeing the underlying data. When you run a query, the query vector is encrypted on the client too — XTrace computes nearest-neighbor Hamming distances directly on ciphertexts and returns the closest chunk IDs for you to decrypt locally.

This SDK has two modules. **x-vec** is the encrypted vector search layer: store and query text chunks with end-to-end encryption. **x-mem** is the encrypted memory layer for AI agents, currently under development. Both share the same credentials, execution context, and integration with XTrace.

## Install

```bash
pip install xtrace-ai-sdk
```

With embedding support (Sentence Transformers):

```bash
pip install "xtrace-ai-sdk[embedding]"
```

## Example

```python
import asyncio
from xtrace_sdk.x_vec.utils.execution_context import ExecutionContext
from xtrace_sdk.x_vec.crypto.key_provider import PassphraseKeyProvider
from xtrace_sdk.x_vec.data_loaders.loader import DataLoader
from xtrace_sdk.x_vec.inference.embedding import Embedding
from xtrace_sdk.integrations.xtrace import XTraceIntegration
from xtrace_sdk.x_vec.retrievers.retriever import Retriever

# One-time setup: generate your private cryptographic state and save it
provider = PassphraseKeyProvider("your-secret-passphrase")
ctx = ExecutionContext.create(
    key_provider=provider,
    homomorphic_client_type="paillier_lookup",
    embedding_length=512,
    key_len=1024,
    path="data/exec_context",
)

embed  = Embedding("sentence_transformer", "mixedbread-ai/mxbai-embed-large-v1", 512)
xtrace = XTraceIntegration(org_id="your_org_id", api_key="your_api_key")

# Encrypt and store documents — content and vectors never leave in plaintext
loader = DataLoader(ctx, xtrace)
docs   = [{"chunk_content": "XTrace encrypts your embeddings.", "meta_data": {}}]
index, db = loader.load_data_from_memory(docs, [embed.bin_embed(d["chunk_content"]) for d in docs])
asyncio.run(loader.dump_db(db, index=index, kb_id="your_kb_id"))

# Query with an encrypted vector — the server never sees the query in plaintext
retriever = Retriever(ctx, xtrace)

async def search(q):
    ids = await retriever.nn_search_for_ids(embed.bin_embed(q), k=3, kb_id="your_kb_id")
    return await retriever.retrieve_and_decrypt(ids, kb_id="your_kb_id")

results = asyncio.run(search("How does XTrace protect my data?"))
```

## Verifying the encryption

This repository is published so that anyone can independently verify that the vector search is genuinely end-to-end encrypted. The tests run fully offline and require no XTrace account.

```bash
pip install -e ".[dev]"
pytest tests/x_vec/
```

`test_paillier_encryption.py` and `test_paillier_lookup_encryption.py` verify encrypt/decrypt round-trips and homomorphic addition on ciphertexts — the same cryptographic primitives used when the SDK sends data to XTrace. The secret key never leaves your machine.

## Documentation

Full documentation at [docs.xtrace.ai](https://docs.xtrace.ai), or build locally with `make html` inside `docs/`.

## License

Apache 2.0 — see [LICENSE](LICENSE).
