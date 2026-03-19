"""Integration tests for XTraceIntegration.meta_search.

The module-scoped fixture creates a fresh KB via the admin API, loads 12
canonical test documents, runs all parametrized query cases, then deletes
the KB on teardown.

Required environment variables
-------------------------------
XTRACE_API_KEY      – XTrace API key (data-plane access)
XTRACE_ADMIN_KEY    – XTrace admin key (KB + key lifecycle)
XTRACE_ORG_ID       – XTrace organisation ID

All tests are skipped when any variable is missing.
"""

import os
import numpy as np
import pytest
import pytest_asyncio

from xtrace_sdk.integrations.xtrace import XTraceIntegration
from xtrace_sdk.x_vec.data_loaders.loader import DataLoader

# ---------------------------------------------------------------------------
# Skip the entire module if credentials are absent
# ---------------------------------------------------------------------------
_REQUIRED = ("XTRACE_API_KEY", "XTRACE_ADMIN_KEY", "XTRACE_ORG_ID")
_missing = [v for v in _REQUIRED if not os.getenv(v)]
pytestmark = pytest.mark.skipif(
    bool(_missing),
    reason=f"Integration env vars not set: {', '.join(_missing)}",
)

# ---------------------------------------------------------------------------
# Canonical test documents (12 chunks with deterministic chunk_ids)
# ---------------------------------------------------------------------------
_DOCUMENTS = [
    {"chunk_id": 1,  "chunk_content": "[c01] Finance invoice for client A.",       "meta_data": {"tag1": "uA",         "tag2": "projX", "tag3": "0000000010", "tag4": "2024-01-01T00:00:00Z", "tag5": "invoices/client_a",    "facets": ["finance", "tax"]}},
    {"chunk_id": 2,  "chunk_content": "[c02] Finance report for client A.",         "meta_data": {"tag1": "uA",         "tag2": "projX", "tag3": "0000000020", "tag4": "2024-01-02T00:00:00Z", "tag5": "reports/client_a",     "facets": ["finance"]}},
    {"chunk_id": 3,  "chunk_content": "[c03] Tech memo for client A.",              "meta_data": {"tag1": "uA",         "tag2": "projY", "tag3": "0000000030", "tag4": "2024-01-03T00:00:00Z", "tag5": "memos/client_a",       "facets": ["tech", "ai"]}},
    {"chunk_id": 4,  "chunk_content": "[c04] Tech roadmap for client A.",           "meta_data": {"tag1": "uA",         "tag2": "projY", "tag3": "0000000040", "tag4": "2024-01-04T00:00:00Z", "tag5": "roadmaps/client_a",    "facets": ["tech"]}},
    {"chunk_id": 5,  "chunk_content": "[c05] Finance + AI analysis for client B.",  "meta_data": {"tag1": "uB",         "tag2": "projX", "tag3": "0000000050", "tag4": "2024-02-01T00:00:00Z", "tag5": "analysis/client_b",    "facets": ["finance", "ai"]}},
    {"chunk_id": 6,  "chunk_content": "[c06] Tax filing notes for client B.",       "meta_data": {"tag1": "uB",         "tag2": "projX", "tag3": "0000000060", "tag4": "2024-02-02T00:00:00Z", "tag5": "notes/client_b",       "facets": ["tax"]}},
    {"chunk_id": 7,  "chunk_content": "[c07] Finance + tech summary for client B.", "meta_data": {"tag1": "uB",         "tag2": "projY", "tag3": "0000000070", "tag4": "2024-02-03T00:00:00Z", "tag5": "summaries/client_b",   "facets": ["finance", "tech"]}},
    {"chunk_id": 8,  "chunk_content": "[c08] AI + tech experiment for client B.",   "meta_data": {"tag1": "uB",         "tag2": "projY", "tag3": "0000000080", "tag4": "2024-02-04T00:00:00Z", "tag5": "experiments/client_b", "facets": ["ai", "tech"]}},
    {"chunk_id": 9,  "chunk_content": "[c09] Misc note for client C.",              "meta_data": {"tag1": "uC",         "tag2": "projZ", "tag3": "0000000090", "tag4": "2024-03-01T00:00:00Z", "tag5": "notes/client_c",       "facets": ["misc"]}},
    {"chunk_id": 10, "chunk_content": "[c10] Empty category doc for client C.",     "meta_data": {"tag1": "uC",         "tag2": "projZ", "tag3": "0000000100", "tag4": "2024-03-02T00:00:00Z", "tag5": "internal/empty",       "facets": []}},
    {"chunk_id": 11, "chunk_content": "[c11] Prefixed user finance tech ai doc.",   "meta_data": {"tag1": "u-prefix-1", "tag2": "alpha", "tag3": "0000000110", "tag4": "2024-03-03T00:00:00Z", "tag5": "imports/batch_1",      "facets": ["finance", "tech", "ai"]}},
    {"chunk_id": 12, "chunk_content": "[c12] Prefixed user tech doc.",              "meta_data": {"tag1": "u-prefix-2", "tag2": "alpha", "tag3": "0000000120", "tag4": "2024-03-04T00:00:00Z", "tag5": "imports/batch_1",      "facets": ["tech"]}},
]

# ---------------------------------------------------------------------------
# Query test cases
# Note: facets_none expected IDs corrected from notebook — [6,9,10] is right
# ---------------------------------------------------------------------------
META_QUERY_TESTS = [
    pytest.param({"tag1": "uA"},                                                                                                [1, 2, 3, 4],                        id="scalar_eq"),
    pytest.param({"tag3": {"$gte": "0000000030", "$lte": "0000000050"}},                                                       [3, 4, 5],                           id="tag3_between"),
    pytest.param({"tag1": {"$ne": "uA"}},                                                                                       [5, 6, 7, 8, 9, 10, 11, 12],         id="tag1_ne"),
    pytest.param({"tag1": {"$in": ["uA", "uC"]}},                                                                              [1, 2, 3, 4, 9, 10],                 id="tag1_in"),
    pytest.param({"facets": "finance"},                                                                                         [1, 2, 5, 7, 11],                    id="facets_contains_scalar"),
    pytest.param({"facets": {"$all": ["finance", "ai"]}},                                                                      [5, 11],                             id="facets_all_alias"),
    pytest.param({"facets": {"$any": ["tax", "misc"]}},                                                                        [1, 6, 9],                           id="facets_any"),
    pytest.param({"facets": {"$none": ["finance", "tech"]}},                                                                   [6, 9, 10],                          id="facets_none"),
    pytest.param({"facets": {"$size": 3}},                                                                                     [11],                                id="facets_size"),
    pytest.param({"facets": {"$exists": True}},                                                                                [1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12], id="facets_exists"),
    pytest.param({"tag5": {"$begins_with": "imports/"}},                                                                       [11, 12],                            id="tag5_begins_with"),
    pytest.param({"tag2": {"$contains": "proj"}},                                                                              [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],    id="tag2_contains_substring"),
    pytest.param({"tag1": "uB", "facets": {"$none": ["finance"]}},                                                            [6, 8],                              id="client_b_not_finance"),
    pytest.param({"tag1": {"$begins_with": "u-prefix-"}, "tag3": {"$gt": "0000000115"}, "facets": {"$any": ["tech", "ai"]}},  [12],                                id="complex_refinement"),
]

# ---------------------------------------------------------------------------
# Module-scoped fixture: create KB → load data → yield → delete KB
# ---------------------------------------------------------------------------
@pytest_asyncio.fixture(scope="module")
async def test_env():
    integration = XTraceIntegration(
        org_id=os.environ["XTRACE_ORG_ID"],
        api_key=os.environ["XTRACE_API_KEY"],
        admin_key=os.environ["XTRACE_ADMIN_KEY"],
    )

    kb = await integration.create_kb(
        name="test-meta-search",
        description="Ephemeral KB for meta_search integration tests — safe to delete",
    )
    kb_id: str = kb["id"]

    # Small crypto params — meta_search ignores vectors, speed > fidelity here
    exec_context = ExecutionContext.create(
        passphrase="test-meta-search-passphrase",
        homomorphic_client_type="paillier_lookup",
        embedding_length=512,
        key_len=512,
    )
    await exec_context.save_to_remote(integration)
    ctx_id: str = exec_context.id

    # Dummy binary vectors — content is irrelevant for metadata-only queries
    rng = np.random.default_rng(seed=42)
    vectors = [rng.integers(0, 2, 512).tolist() for _ in _DOCUMENTS]

    dl = DataLoader(exec_context, integration)
    index, db = dl.load_data_from_memory(_DOCUMENTS, vectors, disable_progress=True)
    await dl.dump_db(db, index=index, kb_id=kb_id)

    yield integration, kb_id, ctx_id

    # Teardown — delete KB (cascades to all chunks) and execution context
    await integration.delete_kb(kb_id)
    await integration.delete_exec_context(ctx_id)
    await integration.close()

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
@pytest.mark.parametrize("meta_filter,expected_ids", META_QUERY_TESTS)
async def test_meta_search(
    test_env: tuple[XTraceIntegration, str, str],
    meta_filter: dict,
    expected_ids: list[int],
) -> None:
    integration, kb_id, ctx_id = test_env
    results = await integration.meta_search(kb_id, meta_filter, ctx_id)
    assert sorted(r["chunk_id"] for r in results) == expected_ids
