import asyncio
import base64
import json
import os
from typing import Any, Optional, Union

import aiohttp
import gmpy2
import msgpack

from xtrace_sdk.x_vec.utils.xtrace_types import Chunk, EncryptedDB


class XTraceIntegration:
    """Async HTTP client for the XTrace encrypted vector database API.

    Handles all network communication with XTrace: uploading encrypted chunks, querying
    Hamming distances, metadata search, and execution context management.

    Supports use as an async context manager for automatic session cleanup:

    .. code-block:: python

        async with XTraceIntegration(org_id="your_org_id") as xtrace:
            await xtrace.store_db(...)

    :param org_id: Your XTrace organisation ID.
    :param api_key: XTrace API key. If omitted, read from the ``XTRACE_API_KEY`` environment variable.
    :param api_url: API base URL, defaults to ``https://api.production.xtrace.ai``.
    """

    def __init__(
        self,
        org_id: str,
        api_key: str | None = None,
        admin_key: str | None = None,
        api_url: str = "https://api.production.xtrace.ai",
    ) -> None:
        self.org_id = org_id
        self.api_url = api_url
        self.batch_size = 500
        self.session: aiohttp.ClientSession | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._lock = asyncio.Lock()

        if not api_key:
            api_key = os.environ["XTRACE_API_KEY"]
        self.api_key = api_key

        self.admin_key: str | None = admin_key or os.environ.get("XTRACE_ADMIN_KEY")

    # ── Session lifecycle ──────────────────────────────────────────

    async def init_session(self) -> aiohttp.ClientSession:
        """Ensure a live ``aiohttp.ClientSession`` exists for the current event loop.

        Creates or recreates the session if it is closed or belongs to a different loop.
        Called automatically before every API request.

        :return: The active client session.
        :rtype: aiohttp.ClientSession
        """
        loop = asyncio.get_running_loop()
        async with self._lock:
            if self.session is None or self.session.closed or self._loop is not loop:
                if self.session is not None and not self.session.closed:
                    try:
                        await self.session.close()
                    except Exception:
                        pass
                self.session = aiohttp.ClientSession(
                    connector=aiohttp.TCPConnector(limit=50, enable_cleanup_closed=True)
                )
                self._loop = loop
        return self.session

    async def close(self) -> None:
        """Close the underlying HTTP session and release all connections."""
        if self.session is not None and not self.session.closed:
            await self.session.close()
            self.session = None
            self._loop = None

    async def __aenter__(self) -> "XTraceIntegration":
        await self.init_session()
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        await self.close()

    # ── Internal helpers ───────────────────────────────────────────

    def _headers(self) -> dict[str, str]:
        return {"x-api-key": self.api_key}

    def _admin_headers(self) -> dict[str, str]:
        if not self.admin_key:
            raise RuntimeError("admin_key is required for admin operations. Pass it to __init__ or set XTRACE_ADMIN_KEY.")
        return {"xtrace-admin-key": self.admin_key, "xtrace-org-id": self.org_id}

    def _encode_index(self, index_item: gmpy2.mpz) -> str:
        return base64.b64encode(int(index_item).to_bytes((index_item.bit_length() + 7) // 8, "little")).decode("utf-8")

    def _preprocess_chunks(self, db: EncryptedDB, index: list[list[int]], update: bool = False) -> list[list[dict]]:
        assert len(db) == len(index), "db and index must be the same length"
        batch: list[dict] = []
        partitioned: list[list[dict]] = []
        for j in range(len(db)):
            raw_content = db[j]["chunk_content"]
            assert isinstance(raw_content, bytes), "chunk_content must be bytes (AES-encrypted) before storing"
            chunk: dict = {
                "chunk_content": raw_content.decode("utf-8"),
                "index": [self._encode_index(i) for i in index[j]],
                "meta_data": db[j].get("meta_data", {}),
                "name": db[j].get("name", f"default_name_chunk_{j}"),
            }
            if update:
                chunk["chunk_id"] = db[j]["chunk_id"]
            batch.append(chunk)
            if len(batch) == self.batch_size:
                partitioned.append(batch)
                batch = []
        if batch:
            partitioned.append(batch)
        return partitioned

    # ── Chunks ─────────────────────────────────────────────────────

    async def get_chunk_by_id(self, chunk_ids: list[int], kb_id: str, projection: list[str] | None = None) -> list[dict]:
        """Fetch one or more chunks by their IDs.

        :param chunk_ids: List of integer chunk IDs to retrieve.
        :type chunk_ids: list[int]
        :param kb_id: Knowledge-base ID the chunks belong to.
        :type kb_id: str
        :param projection: Optional list of fields to include in the response
            (e.g. ``["chunk_id", "meta_data"]``). All fields returned if omitted.
        :type projection: list[str], optional
        :return: List of chunk dicts. The ``index`` field, if present, is base64-decoded to bytes.
        :rtype: list[dict]
        """
        await self.init_session()
        params: dict = {"chunk_ids": chunk_ids, "kb_id": kb_id}
        if projection:
            params["projection"] = projection
        async with self.session.get(  # type: ignore
            f"{self.api_url}/v1/chunk/{self.org_id}", params=params, headers=self._headers()
        ) as res:
            res.raise_for_status()
            data = await res.json()
            for chunk in data:
                if chunk.get("index"):
                    chunk["index"] = [base64.b64decode(i) for i in chunk["index"]]
            return data

    async def store_db(self, db: EncryptedDB, index: list[list[int]], kb_id: str, context_id: str, concurrent: bool = False) -> list[dict]:
        """Upload an encrypted database to XTrace, automatically batching into groups of 500.

        :param db: Encrypted document collection produced by ``DataLoader``.
        :type db: EncryptedDB
        :param index: Encrypted embedding vectors, one list per chunk.
        :type index: list[list[int]]
        :param kb_id: Destination knowledge-base ID.
        :type kb_id: str
        :param context_id: Execution context ID to associate with the data.
        :type context_id: str
        :param concurrent: If ``True``, upload batches concurrently with ``asyncio.gather``.
            Defaults to sequential upload.
        :type concurrent: bool, optional
        :return: List of server responses, one per batch.
        :rtype: list[dict]
        """
        await self.init_session()
        partitioned = self._preprocess_chunks(db, index)
        if concurrent:
            return await asyncio.gather(*[self.insert_chunks(kb_id, b, context_id) for b in partitioned])
        responses = []
        for b in partitioned:
            responses.append(await self.insert_chunks(kb_id, b, context_id))
        return responses

    async def insert_chunks(self, kb_id: str, chunks: list[dict], context_id: str) -> dict:
        """Insert a pre-batched list of chunks. Retries up to 3 times on server errors.

        Prefer :meth:`store_db` for most use cases — it handles batching automatically.

        :param kb_id: Destination knowledge-base ID.
        :type kb_id: str
        :param chunks: List of serialised chunk dicts ready for the API.
        :type chunks: list[dict]
        :param context_id: Execution context ID to associate with the data.
        :type context_id: str
        :return: Server response dict.
        :rtype: dict
        """
        await self.init_session()
        req_body = {"kb_id": kb_id, "chunks": chunks, "context_id": context_id}
        for attempt in range(3):
            try:
                async with self.session.post(  # type: ignore
                    f"{self.api_url}/v1/chunk/{self.org_id}", json=req_body, headers=self._headers()
                ) as res:
                    if res.status == 200:
                        return await res.json()
                    error_text = await res.text()
                    if res.status >= 500 and attempt < 2:
                        wait = (2 ** attempt) + 1
                        await asyncio.sleep(wait)
                        continue
                    raise Exception(f"Failed to insert chunks: {res.status} - {error_text}")
            except (TimeoutError, aiohttp.ClientError):
                if attempt < 2:
                    await asyncio.sleep((2 ** attempt) + 1)
                    continue
                raise
        raise RuntimeError("insert_chunks: exhausted retries")

    async def update_chunks(self, index_updates: list[list[int]], chunk_updates: list[Chunk], context_id: str, kb_id: str) -> list[dict]:
        """Replace existing chunks with updated content and re-encrypted vectors.

        Each entry in ``chunk_updates`` must include a ``chunk_id`` field.

        :param index_updates: New encrypted embedding vectors, one list per chunk.
        :type index_updates: list[list[int]]
        :param chunk_updates: Updated chunk dicts, each containing a ``chunk_id``.
        :type chunk_updates: list[Chunk]
        :param context_id: Execution context ID.
        :type context_id: str
        :param kb_id: Knowledge-base ID the chunks belong to.
        :type kb_id: str
        :return: List of server responses, one per batch.
        :rtype: list[dict]
        """
        await self.init_session()
        partitioned = self._preprocess_chunks(chunk_updates, index_updates, update=True)
        results = []
        for batch in partitioned:
            assert all("chunk_id" in c for c in batch), "All chunk updates must have a chunk_id"
            req_body = {"kb_id": kb_id, "chunk_updates": batch, "context_id": context_id}
            async with self.session.put(  # type: ignore
                f"{self.api_url}/v1/chunk/{self.org_id}", json=req_body, headers=self._headers()
            ) as res:
                if res.status != 200:
                    raise Exception(f"Failed to update chunks: {res.status} - {await res.text()}")
                results.append(await res.json())
        return results

    async def delete_chunks(self, chunk_ids: list[int], kb_id: str) -> dict:
        """Delete chunks by ID from a knowledge base.

        :param chunk_ids: IDs of the chunks to delete.
        :type chunk_ids: list[int]
        :param kb_id: Knowledge-base ID the chunks belong to.
        :type kb_id: str
        :return: Server response dict.
        :rtype: dict
        """
        await self.init_session()
        async with self.session.delete(  # type: ignore
            f"{self.api_url}/v1/chunk/{self.org_id}",
            params={"kb_id": kb_id, "chunk_ids": chunk_ids},
            headers=self._headers(),
        ) as res:
            if res.status != 200:
                raise Exception(f"Failed to delete chunks: {res.status} - {await res.text()}")
            return await res.json()

    async def patch_chunks(self, kb_id: str, context_id: str, chunk_patches: list[dict]) -> dict:
        """Apply partial updates to existing chunks (e.g. metadata only).

        :param kb_id: Knowledge-base ID the chunks belong to.
        :type kb_id: str
        :param context_id: Execution context ID.
        :type context_id: str
        :param chunk_patches: List of patch dicts, each containing a ``chunk_id`` and the fields
            to update.
        :type chunk_patches: list[dict]
        :return: Server response dict.
        :rtype: dict
        """
        await self.init_session()
        req_body = {"kb_id": kb_id, "context_id": context_id, "chunk_patches": chunk_patches}
        async with self.session.patch(  # type: ignore
            f"{self.api_url}/v1/chunk/{self.org_id}", json=req_body, headers=self._headers()
        ) as res:
            if res.status != 200:
                raise Exception(f"Failed to patch chunks: {res.status} - {await res.text()}")
            return await res.json()

    # ── Hamming distances ──────────────────────────────────────────

    async def compute_hamming_distances(
        self,
        query: list[gmpy2.mpz],
        context_id: str,
        kb_id: str,
        meta_filter: str | dict | None = None,
        range: list[int] | None = None,
    ) -> list[tuple[int, list[int | bytes]]]:
        """Submit an encrypted query vector and retrieve encrypted Hamming distances.

        The server computes distances between ``query`` and all stored vectors without decrypting
        either side. Results must be decrypted client-side with the homomorphic secret key.

        :param query: Encrypted query vector produced by the homomorphic client.
        :type query: list[gmpy2.mpz]
        :param context_id: Execution context ID that matches the stored data.
        :type context_id: str
        :param kb_id: Knowledge-base ID to search.
        :type kb_id: str
        :param meta_filter: Optional metadata filter expression (dict or JSON string).
        :param range: Optional ``[min, max]`` range to limit which chunks are searched.
        :return: List of ``(chunk_id, encrypted_distance)`` tuples.
        :rtype: list[tuple[int, list[int | bytes]]]
        """
        await self.init_session()
        req_body: dict = {
            "kb_id": kb_id,
            "query": [self._encode_index(i) for i in query],
            "context_id": context_id,
        }
        if meta_filter is not None:
            req_body["meta_filter"] = json.dumps(meta_filter) if isinstance(meta_filter, dict) else meta_filter
        if range is not None:
            assert isinstance(range, list) and len(range) == 2 and range[0] < range[1], "range must be [min_int, max_int]"
            req_body["range"] = range

        async with self.session.post(  # type: ignore
            f"{self.api_url}/v1/compute_hamming/{self.org_id}", json=req_body, headers=self._headers()
        ) as res:
            res.raise_for_status()
            content = await res.read()
            if "application/x-msgpack" in res.headers.get("Content-Type", ""):
                ham = msgpack.unpackb(content, use_list=True, raw=False)
            else:
                ham = json.loads(content)

        return [(h[1], h[0]) for h in ham]

    # ── Meta search ────────────────────────────────────────────────

    async def meta_search(
        self,
        kb_id: str,
        meta_filter: str | dict,
        context_id: str,
        return_content: bool = False,
        projection: list[str] | None = None,
    ) -> list[dict]:
        """Search chunks by metadata filter expression, returning all matching results.

        :param kb_id: Knowledge-base ID to search.
        :type kb_id: str
        :param meta_filter: Filter expression as a dict or JSON string.
            Uses MongoDB-style operators (``$eq``, ``$in``, ``$and``, etc.).
        :type meta_filter: str or dict
        :param context_id: Execution context ID.
        :type context_id: str
        :param return_content: If ``True``, include the (AES-encrypted) chunk content in results.
            Defaults to ``False``.
        :type return_content: bool, optional
        :param projection: Optional list of fields to include in each result.
        :type projection: list[str], optional
        :return: List of matching chunk dicts.
        :rtype: list[dict]
        """
        await self.init_session()
        req_body: dict[str, Any] = {
            "kb_id": kb_id,
            "meta_filter": json.dumps(meta_filter) if isinstance(meta_filter, (dict, list)) else meta_filter,
            "context_id": context_id,
        }
        if projection is not None:
            req_body["projection"] = projection
        if return_content:
            req_body["return_content"] = return_content
        async with self.session.post(  # type: ignore
            f"{self.api_url}/v1/chunk/meta/{self.org_id}", json=req_body, headers=self._headers()
        ) as res:
            if res.status != 200:
                raise Exception(f"Failed to search chunks by metadata: {res.status} - {await res.text()}")
            return await res.json()

    async def meta_search_paginated(
        self,
        kb_id: str,
        context_id: str,
        meta_filter: str | dict | None = None,
        projection: list[str] | None = None,
        limit: int | None = None,
        offset: int = 0,
        sort_order: str = "desc",
        sort_by: str = "chunk_id",
        return_content: bool = False,
    ) -> dict:
        """Search chunks by metadata filter with pagination support.

        :param kb_id: Knowledge-base ID to search.
        :type kb_id: str
        :param context_id: Execution context ID.
        :type context_id: str
        :param meta_filter: Optional filter expression as a dict or JSON string.
        :type meta_filter: str or dict, optional
        :param projection: Optional list of fields to include in each result.
        :type projection: list[str], optional
        :param limit: Maximum number of results to return. Returns all if omitted.
        :type limit: int, optional
        :param offset: Number of results to skip (for pagination), defaults to ``0``.
        :type offset: int, optional
        :param sort_order: ``"asc"`` or ``"desc"``, defaults to ``"desc"``.
        :type sort_order: str, optional
        :param sort_by: Field to sort by, defaults to ``"chunk_id"``.
        :type sort_by: str, optional
        :param return_content: If ``True``, include the (AES-encrypted) chunk content in results.
            Defaults to ``False``.
        :type return_content: bool, optional
        :return: Dict containing ``results`` (list of chunk dicts) and pagination metadata.
        :rtype: dict
        """
        await self.init_session()
        req_body: dict[str, Any] = {
            "kb_id": kb_id,
            "context_id": context_id,
            "offset": offset,
            "sort_order": sort_order,
            "sort_by": sort_by,
        }
        if meta_filter is not None:
            req_body["meta_filter"] = json.loads(meta_filter) if isinstance(meta_filter, str) else meta_filter
        if projection is not None:
            req_body["projection"] = projection
        if limit is not None:
            req_body["limit"] = limit
        if return_content:
            req_body["return_content"] = return_content
        async with self.session.post(  # type: ignore
            f"{self.api_url}/v1/chunk/meta/page/{self.org_id}", json=req_body, headers=self._headers()
        ) as res:
            if res.status != 200:
                raise Exception(f"Failed to perform paginated meta search: {res.status} - {await res.text()}")
            return await res.json()

    async def delete_chunks_by_meta(
        self, kb_id: str, context_id: str, meta_filter: str | dict, dry_run: bool = False
    ) -> dict:
        """Delete all chunks matching a metadata filter.

        :param kb_id: Knowledge-base ID to delete from.
        :type kb_id: str
        :param context_id: Execution context ID.
        :type context_id: str
        :param meta_filter: Filter expression as a dict or JSON string.
        :type meta_filter: str or dict
        :param dry_run: If ``True``, return a count of chunks that would be deleted without
            actually deleting them. Defaults to ``False``.
        :type dry_run: bool, optional
        :return: Server response dict (includes ``deleted_count`` or ``would_delete_count``).
        :rtype: dict
        """
        await self.init_session()
        req_body = {
            "kb_id": kb_id,
            "context_id": context_id,
            "meta_filter": json.loads(meta_filter) if isinstance(meta_filter, str) else meta_filter,
            "dry_run": dry_run,
        }
        async with self.session.delete(  # type: ignore
            f"{self.api_url}/v1/chunk/by-meta/{self.org_id}", json=req_body, headers=self._headers()
        ) as res:
            if res.status != 200:
                raise Exception(f"Failed to delete chunks by meta: {res.status} - {await res.text()}")
            return await res.json()

    async def patch_chunks_by_meta(
        self, kb_id: str, context_id: str, meta_filter: str | dict, patch: dict, dry_run: bool = False
    ) -> dict:
        """Apply a metadata patch to all chunks matching a filter expression.

        :param kb_id: Knowledge-base ID to update.
        :type kb_id: str
        :param context_id: Execution context ID.
        :type context_id: str
        :param meta_filter: Filter expression as a dict or JSON string.
        :type meta_filter: str or dict
        :param patch: Metadata fields to set on each matching chunk.
        :type patch: dict
        :param dry_run: If ``True``, return a count of chunks that would be updated without
            modifying anything. Defaults to ``False``.
        :type dry_run: bool, optional
        :return: Server response dict.
        :rtype: dict
        """
        await self.init_session()
        req_body = {
            "kb_id": kb_id,
            "context_id": context_id,
            "meta_filter": json.loads(meta_filter) if isinstance(meta_filter, str) else meta_filter,
            "patch": patch,
            "dry_run": dry_run,
        }
        async with self.session.patch(  # type: ignore
            f"{self.api_url}/v1/chunk/by-meta/{self.org_id}", json=req_body, headers=self._headers()
        ) as res:
            if res.status != 200:
                raise Exception(f"Failed to patch chunks by meta: {res.status} - {await res.text()}")
            return await res.json()

    # ── Execution contexts ─────────────────────────────────────────

    async def store_exec_context(self, exec_context_dict: dict, context_id: str) -> str:
        """Upload a serialised execution context to XTrace remote storage.

        The secret key inside ``exec_context_dict`` must already be AES-encrypted before calling
        this method. Use :meth:`~xtrace_vec.utils.execution_context.ExecutionContext.save_to_remote`
        instead of calling this directly.

        :param exec_context_dict: Encrypted execution context produced by
            :meth:`~xtrace_vec.utils.execution_context.ExecutionContext.to_dict_enc`.
        :type exec_context_dict: dict
        :param context_id: Desired context ID (typically the SHA-256 fingerprint of the context).
        :type context_id: str
        :return: The ``context_id`` confirmed by the server.
        :rtype: str
        """
        await self.init_session()
        async with self.session.post(  # type: ignore
            f"{self.api_url}/v1/exec_context/{self.org_id}",
            json={"data": exec_context_dict, "context_id": context_id},
            headers=self._headers(),
        ) as res:
            if not res.ok:
                raise Exception(f"Failed to store execution context: {res.status} - {await res.text()}")
            return (await res.json())["context_id"]

    async def get_serialized_exec_context(self, context_id: str) -> dict:
        """Fetch a serialised execution context from XTrace remote storage.

        :param context_id: ID of the context to retrieve.
        :type context_id: str
        :return: Encrypted execution context dict (secret key is still AES-encrypted).
        :rtype: dict
        """
        await self.init_session()
        async with self.session.get(  # type: ignore
            f"{self.api_url}/v1/exec_context/{self.org_id}/{context_id}", headers=self._headers()
        ) as res:
            res.raise_for_status()
            return await res.json()

    async def list_exec_contexts(self) -> list[str]:
        """List all execution context IDs stored under this organisation.

        :return: List of context ID strings.
        :rtype: list[str]
        """
        await self.init_session()
        async with self.session.get(  # type: ignore
            f"{self.api_url}/v1/exec_context/{self.org_id}", headers=self._headers()
        ) as res:
            if res.status != 200:
                raise Exception(f"Failed to list execution contexts: {res.status} - {await res.text()}")
            return (await res.json())["contexts"]

    async def delete_exec_context(self, context_id: str) -> bool:
        """Delete an execution context from XTrace remote storage.

        :param context_id: ID of the context to delete.
        :type context_id: str
        :return: ``True`` on success.
        :rtype: bool
        """
        await self.init_session()
        async with self.session.delete(  # type: ignore
            f"{self.api_url}/v1/exec_context/{self.org_id}/{context_id}", headers=self._headers()
        ) as res:
            if res.status != 200:
                raise Exception(f"Failed to delete execution context: {res.status} - {await res.text()}")
            return True

    # ── KB ↔ context bindings ──────────────────────────────────────

    async def get_ctx_for_kb(self, kb_id: str) -> list[str] | None:
        """Return the execution context IDs bound to a knowledge base.

        :param kb_id: Knowledge-base ID to look up.
        :type kb_id: str
        :return: List of context ID strings, or ``None`` if the lookup fails.
        :rtype: list[str] or None
        """
        await self.init_session()
        try:
            async with self.session.get(  # type: ignore
                f"{self.api_url}/v1/exec_context/{self.org_id}/{kb_id}/context", headers=self._headers()
            ) as res:
                return (await res.json()).get("context_ids") if res.status == 200 else None
        except Exception:
            return None

    async def get_kbs_for_ctx(self, ctx_id: str) -> list[str] | None:
        """Return the knowledge-base IDs bound to an execution context.

        :param ctx_id: Execution context ID to look up.
        :type ctx_id: str
        :return: List of knowledge-base ID strings, or ``None`` if the lookup fails.
        :rtype: list[str] or None
        """
        await self.init_session()
        try:
            async with self.session.get(  # type: ignore
                f"{self.api_url}/v1/exec_context/{self.org_id}/{ctx_id}/kb", headers=self._headers()
            ) as res:
                return (await res.json()).get("kb_ids", []) if res.status == 200 else None
        except Exception:
            return None

    async def register_kb_binding(self, kb_id: str, ctx_id: str) -> bool:
        """Bind a knowledge base to an execution context on the server.

        :param kb_id: Knowledge-base ID to bind.
        :type kb_id: str
        :param ctx_id: Execution context ID to bind to.
        :type ctx_id: str
        :return: ``True`` on success, ``False`` otherwise.
        :rtype: bool
        """
        await self.init_session()
        try:
            async with self.session.post(  # type: ignore
                f"{self.api_url}/v1/exec_context/{self.org_id}/{kb_id}/bind/{ctx_id}", headers=self._headers()
            ) as res:
                return res.status == 200
        except Exception:
            return False

    # ── Admin API ──────────────────────────────────────────────────

    async def list_kbs(self) -> list[dict]:
        """List all knowledge bases for the organisation.

        Requires an admin key (``admin_key`` constructor arg or ``XTRACE_ADMIN_KEY`` env var).

        :return: List of KB records.
        :rtype: list[dict]
        """
        await self.init_session()
        async with self.session.get(  # type: ignore
            f"{self.api_url}/api/v1/kbs",
            headers=self._admin_headers(),
        ) as res:
            res.raise_for_status()
            return await res.json()

    async def get_kb(self, kb_id: str) -> dict:
        """Fetch metadata for a single knowledge base.

        Requires an admin key (``admin_key`` constructor arg or ``XTRACE_ADMIN_KEY`` env var).

        :param kb_id: Knowledge base ID to fetch.
        :type kb_id: str
        :return: KB record dict.
        :rtype: dict
        """
        await self.init_session()
        async with self.session.get(  # type: ignore
            f"{self.api_url}/api/v1/kbs/{kb_id}",
            headers=self._admin_headers(),
        ) as res:
            res.raise_for_status()
            return await res.json()

    async def list_api_keys(self) -> list[dict]:
        """List all API keys for the organisation.

        Requires an admin key (``admin_key`` constructor arg or ``XTRACE_ADMIN_KEY`` env var).

        :return: List of API key records.
        :rtype: list[dict]
        """
        await self.init_session()
        async with self.session.get(  # type: ignore
            f"{self.api_url}/api/v1/keys",
            headers=self._admin_headers(),
        ) as res:
            res.raise_for_status()
            return await res.json()

    async def get_key_permissions(self, key_hash: str) -> list[dict]:
        """Get KB permissions for an API key.

        Requires an admin key (``admin_key`` constructor arg or ``XTRACE_ADMIN_KEY`` env var).

        :param key_hash: Hash of the API key.
        :type key_hash: str
        :return: List of permission records.
        :rtype: list[dict]
        """
        await self.init_session()
        async with self.session.get(  # type: ignore
            f"{self.api_url}/api/v1/keys/{key_hash}/permissions",
            headers=self._admin_headers(),
        ) as res:
            res.raise_for_status()
            return await res.json()

    async def set_key_permission(self, key_hash: str, kb_id: str, permission: int) -> dict:
        """Set KB permission for an API key.

        Requires an admin key (``admin_key`` constructor arg or ``XTRACE_ADMIN_KEY`` env var).

        :param key_hash: Hash of the API key to update.
        :type key_hash: str
        :param kb_id: Knowledge base ID to set permission on.
        :type kb_id: str
        :param permission: Permission level: 1=read, 3=write, 7=delete.
        :type permission: int
        :return: Server response dict.
        :rtype: dict
        """
        await self.init_session()
        async with self.session.put(  # type: ignore
            f"{self.api_url}/api/v1/keys/{key_hash}/permissions/{kb_id}",
            headers=self._admin_headers(),
            json={"permission": permission},
        ) as res:
            res.raise_for_status()
            return await res.json()

    async def create_kb(self, name: str, description: str = "") -> dict:
        """Create a new knowledge base.

        Requires an admin key (``admin_key`` constructor arg or ``XTRACE_ADMIN_KEY`` env var).

        :param name: Human-readable name for the knowledge base.
        :type name: str
        :param description: Optional description.
        :type description: str
        :return: Created KB record with ``id``, ``name``, and ``description``.
        :rtype: dict
        """
        await self.init_session()
        async with self.session.post(  # type: ignore
            f"{self.api_url}/api/v1/kbs",
            headers=self._admin_headers(),
            json={"name": name, "description": description},
        ) as res:
            res.raise_for_status()
            return await res.json()

    async def delete_kb(self, kb_id: str) -> None:
        """Permanently delete a knowledge base and all its chunks.

        Requires an admin key (``admin_key`` constructor arg or ``XTRACE_ADMIN_KEY`` env var).

        :param kb_id: ID of the knowledge base to delete.
        :type kb_id: str
        """
        await self.init_session()
        async with self.session.delete(  # type: ignore
            f"{self.api_url}/api/v1/kbs/{kb_id}",
            headers=self._admin_headers(),
        ) as res:
            res.raise_for_status()

    async def create_api_key(self, name: str, description: str = "") -> dict:
        """Create a new API key for the organisation.

        Requires an admin key (``admin_key`` constructor arg or ``XTRACE_ADMIN_KEY`` env var).

        :param name: Human-readable name for the API key.
        :type name: str
        :param description: Optional description.
        :type description: str
        :return: Dict with ``apiKey`` (metadata) and ``secretKey`` (the actual key — store it, it won't be shown again).
        :rtype: dict
        """
        await self.init_session()
        async with self.session.post(  # type: ignore
            f"{self.api_url}/api/v1/keys",
            headers=self._admin_headers(),
            json={"name": name, "description": description},
        ) as res:
            res.raise_for_status()
            return await res.json()

    async def revoke_api_key(self, key_hash: str) -> None:
        """Revoke an API key by its hash.

        Requires an admin key (``admin_key`` constructor arg or ``XTRACE_ADMIN_KEY`` env var).

        :param key_hash: Hash of the API key to revoke (from :meth:`create_api_key`).
        :type key_hash: str
        """
        await self.init_session()
        async with self.session.patch(  # type: ignore
            f"{self.api_url}/api/v1/keys/{key_hash}/status/revoke",
            headers=self._admin_headers(),
        ) as res:
            res.raise_for_status()
