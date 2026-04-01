All notable changes to this project will be documented in this file.

## [0.1.1] - 2026-03-31

### Fixed

- CLI commands failed to locate `.env` when installed from PyPI (`pip install xtrace-ai-sdk`). `dotenv.load_dotenv()` without arguments searches from the caller's file directory (site-packages), never reaching the user's working directory. All 13 call sites now use `find_dotenv(usecwd=True)` for cwd-based discovery. Editable installs (`pip install -e .`) were unaffected because the source tree sits under the user's home directory.

## [0.1.0] - 2026-03-30

### Initial release

- `XTraceIntegration` — HTTP client for the XTrace encrypted vector DB API (chunk CRUD, Hamming distance, metadata search, execution context management)
- `DataLoader` — encrypt and ingest document collections using Paillier homomorphic encryption + AES
- `Retriever` — encrypted nearest-neighbor search with optional multiprocessing decode (`parallel=True`)
- `ExecutionContext` — key-provider-protected container for all crypto state; save/load locally or via XTrace
- `KeyProvider` protocol with `PassphraseKeyProvider` (scrypt-based) and `AWSKMSKeyProvider` (envelope encryption via AWS KMS)
- `PaillierClient` and `PaillierLookupClient` — Paillier homomorphic encryption clients optimised for Hamming distance
- `GoldwasserMicaliClient` — Goldwasser-Micali homomorphic encryption client
- `Embedding` — embedding provider wrapper (Ollama, OpenAI, Sentence Transformers)
- `InferenceClient` — RAG inference wrapper (OpenAI, Anthropic, Redpill, Ollama)
