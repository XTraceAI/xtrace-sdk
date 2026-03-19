from __future__ import annotations

import inspect
import logging
import multiprocessing
import os
from time import perf_counter

import numpy as np

from xtrace_sdk.x_vec.crypto.hamming_client_base import HammingClientBase
from xtrace_sdk.x_vec.inference.embedding import Embedding
from xtrace_sdk.integrations.xtrace import XTraceIntegration
from xtrace_sdk.x_vec.utils.execution_context import ExecutionContext

_log = logging.getLogger(__name__)


def _decode_item(homomorphic_client: HammingClientBase, ciphertext: list[int | bytes]) -> int:
    """Top-level helper for multiprocessing — must be picklable."""
    return homomorphic_client.decode_hamming_client_one(ciphertext)


class Retriever:
    """Retrieves and decrypts chunks from XTrace using encrypted hamming distance search.

    Args:
        execution_context: Holds AES + homomorphic clients and context ID.
        integration: XTrace API integration instance.
        parallel: If True, decode hamming distances in parallel using multiprocessing.
    """

    def __init__(
        self,
        execution_context: ExecutionContext,
        integration: XTraceIntegration,
        parallel: bool = False,
    ) -> None:
        self.execution_context = execution_context
        self.integration = integration
        self.parallel = parallel


    async def nn_search_for_ids(
        self,
        query_vector: list[float],
        k: int = 3,
        kb_id: str = "",
        meta_filter: dict | None = None,
        range_filter: list[int] | None = None,
        include_scores: bool = False,
    ) -> list[int] | tuple[list[int], list[int]]:
        """Find the k nearest neighbors by encrypted hamming distance.

        Args:
            query_vector: Float embedding vector to search with.
            k: Number of nearest neighbors to return.
            kb_id: Knowledge-base ID to search.
            meta_filter: Optional metadata filter dict (MongoDB-style operators).
            range_filter: Optional ``[min, max]`` range to restrict which chunks are searched.
            include_scores: If True, also return the plain hamming distances.

        Returns:
            List of chunk IDs, or (chunk_ids, scores) if include_scores=True.
        """
        if inspect.isawaitable(query_vector):
            query_vector = await query_vector
        assert len(query_vector) == self.execution_context.embed_len(), (
            f"Query dimension {len(query_vector)} does not match "
            f"homomorphic client dimension {self.execution_context.embed_len()}"
        )
        query_bin: list[int] = Embedding.float_2_bin(query_vector).tolist()

        t0 = perf_counter()
        encrypted_query = self.execution_context.homomorphic.encrypt_vec_one(query_bin)
        _log.info("query_encrypt_done", extra={"timing_ms": (perf_counter() - t0) * 1000})

        hamming_kwargs: dict = {"kb_id": kb_id}
        if meta_filter is not None:
            hamming_kwargs["meta_filter"] = meta_filter
        if range_filter is not None:
            hamming_kwargs["range"] = range_filter

        t0 = perf_counter()
        encrypted_hamming = await self.integration.compute_hamming_distances(
            encrypted_query, self.execution_context.id, **hamming_kwargs
        )
        _log.info("query_hamming_network_done", extra={"timing_ms": (perf_counter() - t0) * 1000})

        t0 = perf_counter()
        if self.parallel:
            os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
            tasks = [(self.execution_context.homomorphic, c[1]) for c in encrypted_hamming]
            with multiprocessing.Pool() as pool:
                plain_hamming = pool.starmap(_decode_item, tasks)
        else:
            plain_hamming = self.execution_context.homomorphic.decode_hamming_client_batch(
                [c[1] for c in encrypted_hamming]
            )
        _log.info("query_decrypt_done", extra={"timing_ms": (perf_counter() - t0) * 1000})

        selected_ids = [encrypted_hamming[i][0] for i in np.argsort(plain_hamming)[:k]]
        if include_scores:
            return selected_ids, plain_hamming
        return selected_ids

    async def retrieve_and_decrypt(
        self,
        chunk_ids: list[int],
        kb_id: str,
        projection: list[str] | None = None,
    ) -> list[dict]:
        """Fetch chunks by ID and AES-decrypt their content.

        Args:
            chunk_ids: List of chunk IDs to retrieve.
            kb_id: Knowledge-base ID the chunks belong to.
            projection: Fields to return; defaults to all standard fields.

        Returns:
            List of dicts with decrypted chunk_content and meta_data.
        """
        if projection is None:
            projection = ["chunk_id", "chunk_content", "tag1", "tag2", "tag3", "tag4", "tag5", "facets"]

        t0 = perf_counter()
        res = await self.integration.get_chunk_by_id(chunk_ids, kb_id, projection=projection)
        _log.info(
            "query_retrieve_network_done",
            extra={"timing_ms": (perf_counter() - t0) * 1000, "chunk_count": len(chunk_ids)},
        )

        t0 = perf_counter()
        context_plain = [
            {"chunk_content": self.execution_context.aes.decrypt(i["chunk_content"]), "meta_data": i["meta_data"]}
            for i in res
        ]
        _log.info(
            "query_aes_decrypt_done",
            extra={"timing_ms": (perf_counter() - t0) * 1000, "chunk_count": len(res)},
        )
        return context_plain

    @staticmethod
    def format_context(contexts: list) -> str:
        """Format a list of context strings for use in an LLM prompt."""
        return "".join(f"{i}. {v}\n" for i, v in enumerate(contexts))
