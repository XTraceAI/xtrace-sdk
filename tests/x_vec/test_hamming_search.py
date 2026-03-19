"""Integration tests for encrypted Hamming distance search (nn_search_for_ids).

The module-scoped fixture creates a fresh KB, stores 12 documents with
deterministic binary vectors, then verifies that nearest-neighbour queries
return the expected chunk IDs.  Teardown deletes the KB.

Vector round-trip
-----------------
Vectors are stored as binary (0/1) via DataLoader.
Queries are sent as floats where  0 → -1.0  and  1 → +1.0.
Embedding.float_2_bin converts (x > 0), so the round-trip is exact:
exact same vector → Hamming distance 0 → top-1 is always the right chunk.

Required environment variables
-------------------------------
XTRACE_API_KEY    – XTrace API key
XTRACE_ADMIN_KEY  – XTrace admin key (KB lifecycle)
XTRACE_ORG_ID     – XTrace organisation ID
"""

import os
import numpy as np
import pytest
import pytest_asyncio

from xtrace_sdk.integrations.xtrace import XTraceIntegration
from xtrace_sdk.x_vec.data_loaders.loader import DataLoader
from xtrace_sdk.x_vec.retrievers.retriever import Retriever

# ---------------------------------------------------------------------------
# Skip if credentials are absent
# ---------------------------------------------------------------------------
_REQUIRED = ("XTRACE_API_KEY", "XTRACE_ADMIN_KEY", "XTRACE_ORG_ID")
_missing = [v for v in _REQUIRED if not os.getenv(v)]
pytestmark = pytest.mark.skipif(
    bool(_missing),
    reason=f"Integration env vars not set: {', '.join(_missing)}",
)

# ---------------------------------------------------------------------------
# Deterministic vectors
# float_2_bin  →  (x > 0).astype(int8)
# So: store binary = float_2_bin(float_vec)
#     query float  = binary * 2.0 - 1.0   (0 → -1.0, 1 → +1.0)
#     round-trip   = exact match → distance 0
# ---------------------------------------------------------------------------
_RNG = np.random.default_rng(seed=7)
_EMBED_LEN = 512
_N_DOCS = 12
# float vectors of -1/+1; binarised for storage, used directly for queries
_FLOAT_VECS: list[np.ndarray] = [_RNG.choice([-1.0, 1.0], size=_EMBED_LEN) for _ in range(_N_DOCS)]
_BIN_VECS:   list[np.ndarray] = [(v > 0).astype(np.int8) for v in _FLOAT_VECS]

_DOCUMENTS = [
    {
        "chunk_id": i + 1,
        "chunk_content": f"[c{i+1:02d}] Hamming test document {i+1}.",
        "meta_data": {
            "tag1": f"group_{(i % 3) + 1}",   # groups: 1, 2, 3 (4 docs each)
            "tag2": "hamming-test",
            "tag3": f"{(i + 1) * 10:013d}",
            "tag4": f"2024-0{(i % 9) + 1}-01T00:00:00Z",
            "tag5": f"doc/{i+1}",
            "facets": [],
        },
    }
    for i in range(_N_DOCS)
]

# ---------------------------------------------------------------------------
# Module-scoped fixture
# ---------------------------------------------------------------------------
@pytest_asyncio.fixture(scope="module")
async def test_env():
    integration = XTraceIntegration(
        org_id=os.environ["XTRACE_ORG_ID"],
        api_key=os.environ["XTRACE_API_KEY"],
        admin_key=os.environ["XTRACE_ADMIN_KEY"],
    )

    kb = await integration.create_kb(
        name="test-hamming-search",
        description="Ephemeral KB for Hamming distance integration tests — safe to delete",
    )
    kb_id: str = kb["id"]

    exec_context = ExecutionContext.create(
        passphrase="test-hamming-search-passphrase",
        homomorphic_client_type="paillier_lookup",
        embedding_length=_EMBED_LEN,
        key_len=512,
    )
    await exec_context.save_to_remote(integration)
    ctx_id: str = exec_context.id

    dl = DataLoader(exec_context, integration)
    index, db = dl.load_data_from_memory(_DOCUMENTS, _BIN_VECS, disable_progress=True)
    await dl.dump_db(db, index=index, kb_id=kb_id)

    retriever = Retriever(exec_context, integration)

    yield integration, retriever, kb_id

    await integration.delete_kb(kb_id)
    await integration.delete_exec_context(ctx_id)
    await integration.close()

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------
def _query_vec(doc_idx: int) -> list[float]:
    """Float query that round-trips to the exact binary vector of doc_idx."""
    return (_BIN_VECS[doc_idx] * 2.0 - 1.0).tolist()

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
@pytest.mark.parametrize("doc_idx", [0, 3, 7, 11], ids=["doc_1", "doc_4", "doc_8", "doc_12"])
async def test_exact_match_is_top1(
    test_env: tuple,
    doc_idx: int,
) -> None:
    """Querying with a document's exact vector should return it as top-1."""
    _, retriever, kb_id = test_env
    expected_id = _DOCUMENTS[doc_idx]["chunk_id"]
    ids = await retriever.nn_search_for_ids(_query_vec(doc_idx), k=1, kb_id=kb_id)
    assert ids[0] == expected_id


@pytest.mark.asyncio
@pytest.mark.parametrize("doc_idx", [0, 5, 11], ids=["doc_1", "doc_6", "doc_12"])
async def test_exact_match_in_topk(
    test_env: tuple,
    doc_idx: int,
) -> None:
    """Querying with a document's exact vector should always appear in top-3."""
    _, retriever, kb_id = test_env
    expected_id = _DOCUMENTS[doc_idx]["chunk_id"]
    ids = await retriever.nn_search_for_ids(_query_vec(doc_idx), k=3, kb_id=kb_id)
    assert expected_id in ids


@pytest.mark.asyncio
async def test_meta_filter_restricts_results(test_env: tuple) -> None:
    """Top-1 from a filtered search must belong to the filtered group."""
    _, retriever, kb_id = test_env
    # Query with doc_1's vector but restrict to group_2 (chunk_ids 2, 5, 8, 11)
    group_2_ids = {d["chunk_id"] for d in _DOCUMENTS if d["meta_data"]["tag1"] == "group_2"}
    ids = await retriever.nn_search_for_ids(
        _query_vec(0), k=1, kb_id=kb_id, meta_filter={"tag1": "group_2"}
    )
    assert ids[0] in group_2_ids


@pytest.mark.asyncio
async def test_include_scores_returns_non_negative(test_env: tuple) -> None:
    """include_scores=True should return non-negative integer Hamming distances."""
    _, retriever, kb_id = test_env
    ids, scores = await retriever.nn_search_for_ids(
        _query_vec(0), k=3, kb_id=kb_id, include_scores=True
    )
    assert len(ids) == len(scores) == 3
    assert all(s >= 0 for s in scores)


@pytest.mark.asyncio
async def test_exact_match_has_zero_distance(test_env: tuple) -> None:
    """An exact-vector query must return distance 0 for the matching document."""
    _, retriever, kb_id = test_env
    ids, scores = await retriever.nn_search_for_ids(
        _query_vec(2), k=_N_DOCS, kb_id=kb_id, include_scores=True
    )
    expected_id = _DOCUMENTS[2]["chunk_id"]
    idx = ids.index(expected_id)
    assert scores[idx] == 0
