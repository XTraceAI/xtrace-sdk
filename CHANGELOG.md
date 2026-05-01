All notable changes to this project will be documented in this file.

## [Unreleased]

### Changed

- **Device selection moved from the `DEVICE` environment variable to a `device=` constructor keyword** on `PaillierClient`, `PaillierLookupClient`, `ExecutionContext.create`, `ExecutionContext.load_from_disk`, and `ExecutionContext.load_from_remote`. The new default is `device="auto"`: clients probe for the GPU extension at construction time and fall back to CPU silently when no GPU is present. If a GPU is detected on the host (`/dev/nvidia0`, `nvidia-smi`, or `CUDA_VISIBLE_DEVICES`) but the extension fails to load, a warning is emitted instead of a silent CPU fallback so misconfigured GPU hosts don't quietly run on CPU. `device="gpu"` raises if the extension is unavailable; `device="cpu"` skips probing entirely.
- The `DEVICE` environment variable is no longer read. Existing callers must migrate to the `device=` kwarg.

### Added

- `xtrace_sdk.x_vec.crypto.device.resolve_device` — shared device-resolution helper used by both Paillier clients and intended for future homomorphic clients with optional GPU backends.
- `test_cross_device_interop` — parametrised test covering CPU↔GPU portability for both `PaillierClient` and `PaillierLookupClient`. Verifies that contexts saved on one device load correctly on the other (hash equality), and that ciphertexts produced on either backend round-trip correctly through the other for both encryption and server-side homomorphic add.

## [0.2.0] - 2026-04-18

### Added

- **CUDA backend for Paillier and Paillier-Lookup homomorphic encryption.** The `.cu` sources are now in-tree under `src/xtrace_sdk/x_vec/crypto/paillier-GPU-{,lookup-}client/`, and the new `build_gpu_binaries.sh` script compiles both pybind11 extensions inside a pinned `nvidia/cuda:12.4.0-devel-ubuntu22.04` Docker container. No local CUDA toolkit is required on the build host — only Docker and `nvcc` (provided by the image). Runtime requires NVIDIA driver ≥ 550. Opt-in via `DEVICE=gpu`; CPU remains the default.
- `PaillierGPUClient` and `PaillierLookupGPUClient` — standalone GPU-backed Hamming clients for `ExecutionContext`. Type dispatch in `ExecutionContext.create` now accepts these as first-class alternatives to their CPU counterparts.
- Documented `DEVICE` environment variable in the configuration reference.
- "GPU acceleration" section in the installation docs covering build script usage, runtime requirements, and per-build knobs (`PYTHON_VERSION`, `SMS`, `KEY_BITS`, `ALPHA_LEN`, `CUDA_IMAGE`).

### Changed

- `DEVICE` is now resolved at client instantiation time rather than at module import time. Previously, setting `os.environ["DEVICE"]` after importing a client (common in notebooks) had no effect; the module-level constant was already frozen to `cpu`. The env var is now read fresh inside `PaillierClient.__init__`, `PaillierLookupClient.__init__`, and `PaillierLookupClient.__setstate__`, so CPU and GPU clients can be instantiated in the same process and ordering of imports no longer matters.
- Error messages raised when the GPU extension cannot be loaded now point users at `./build_gpu_binaries.sh` and document runtime driver requirements, replacing the previous stale reference to `src/crypto/` and non-existent documentation.

## [0.1.1] - 2026-03-31

### Fixed

- CLI commands failed to locate `.env` when installed from PyPI (`pip install xtrace-ai-sdk`). `dotenv.load_dotenv()` without arguments searches from the caller's file directory (site-packages), never reaching the user's working directory. All 13 call sites now use `find_dotenv(usecwd=True)` for cwd-based discovery. Editable installs (`pip install -e .`) were unaffected because the source tree sits under the user's home directory.
- `_chunk_into_pages` emitted oversized chunks when a single sentence exceeded `page_size`. These are now hard-split by character boundary. `_chunk_json` gains a `cap_json_elements` option to further split array elements that exceed the page size.
- `init` no longer warns about `.env` key conflicts when the existing value already matches the value being written. T

### Added

- `EmbeddingError` exception in `x_vec.inference.embedding` with `status` and `chunk_len` fields. The Ollama provider now parses error responses instead of raising a bare `aiohttp` `ClientResponseError`.
- `load` and `upsert-file` accept `--max-chunk-chars` to cap chunk size (splits oversized chunks via the text chunker) and `--max-parallel-embeddings` to limit concurrent embedding requests.
- All CLI commands that embed (`load`, `upsert`, `upsert-file`, `retrieve`) catch `EmbeddingError` and print context-specific hints (e.g. suggesting `--max-chunk-chars` for context-length errors, `--max-parallel-embeddings` for server-busy errors).

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
