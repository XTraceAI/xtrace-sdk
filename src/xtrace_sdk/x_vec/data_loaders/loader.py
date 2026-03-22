import inspect
from pathlib import Path
from typing import Optional

from tqdm import tqdm

from xtrace_sdk.x_vec.inference.embedding import Embedding
from xtrace_sdk.integrations.xtrace import XTraceIntegration
from xtrace_sdk.x_vec.utils.bench import bench
from xtrace_sdk.x_vec.utils.execution_context import ExecutionContext
from xtrace_sdk.x_vec.utils.xtrace_types import Chunk, DocumentCollection, EncryptedChunk, EncryptedDB, EncryptedIndex


class DataLoader:
    """Encrypts and uploads document collections to XTrace.

    ``DataLoader`` handles the two encryption steps required before data reaches XTrace:

    1. **AES encryption** of chunk content (text → ciphertext bytes).
    2. **Homomorphic encryption** of embedding vectors (float vector → encrypted index).

    The encrypted data is then uploaded via :class:`~xtrace_sdk.integrations.xtrace.XTraceIntegration`.
    Neither the plaintext chunk content nor the raw embedding vectors leave the client.

    :param execution_context: Initialised :class:`~xtrace_vec.utils.execution_context.ExecutionContext`
        containing the AES key and homomorphic client.
    :param integration: Authenticated :class:`~xtrace_sdk.integrations.xtrace.XTraceIntegration` instance.
    """

    def __init__(self, execution_context: ExecutionContext, integration: XTraceIntegration) -> None:
        self.execution_context = execution_context
        self.integration = integration



    async def dump_db(self, db: EncryptedDB, index: EncryptedIndex, kb_id: str, concurrent: bool = False) -> list[dict]:
        """Upload a pre-encrypted database to XTrace.

        Typically called after :meth:`load_data_from_memory` or :meth:`load_data_from_memory_batch`
        have produced an ``(index, encrypted_db)`` pair.

        :param db: Encrypted document collection produced by :meth:`load_data_from_memory`.
        :type db: EncryptedDB
        :param index: Encrypted embedding vectors produced by :meth:`load_data_from_memory`.
        :type index: EncryptedIndex
        :param kb_id: Destination knowledge-base ID.
        :type kb_id: str
        :param concurrent: Upload batches concurrently. Defaults to ``False``.
        :type concurrent: bool, optional
        :return: List of server responses, one per upload batch.
        :rtype: list[dict]
        """
        with bench("ingest_store_network"):
            result = await self.integration.store_db(db, index, kb_id, context_id=self.execution_context.id, concurrent=concurrent)
        return result

    async def upsert_one(self, chunk: Chunk, vector: list[float], kb_id: str) -> list[dict]:
        """Encrypt and upload a single chunk.

        :param chunk: Chunk dict with at minimum a ``chunk_content`` string field.
        :type chunk: Chunk
        :param vector: Float embedding vector for this chunk.
        :type vector: list[float]
        :param kb_id: Destination knowledge-base ID.
        :type kb_id: str
        :return: Server response list.
        :rtype: list[dict]
        :raises ValueError: If ``chunk`` is not a dict.
        """
        if not isinstance(chunk, dict):
            raise ValueError("Invalid chunk type. Expected Chunk instance.")
        index, db = await self.load_data_from_memory([chunk], [vector])
        return await self.integration.store_db(db, index, kb_id, context_id=self.execution_context.id)

    async def delete_chunks(self, chunk_ids: list[int], kb_id: str) -> dict:
        """Delete chunks by ID.

        :param chunk_ids: List of chunk IDs to delete.
        :type chunk_ids: list[int]
        :param kb_id: Knowledge-base ID the chunks belong to.
        :type kb_id: str
        :return: Server response.
        :rtype: dict
        """
        return await self.integration.delete_chunks(chunk_ids, kb_id)

    async def update_chunks(self, chunk_updates: list[Chunk], vectors: list[list[float]], kb_id: str) -> list[dict]:
        """Re-encrypt updated chunks and upload them.

        Each chunk in ``chunk_updates`` must include a ``chunk_id`` field identifying the
        record to replace.

        :param chunk_updates: Updated chunk dicts, each containing a ``chunk_id``.
        :type chunk_updates: list[Chunk]
        :param vectors: New float embedding vectors, one per chunk.
        :type vectors: list[list[float]]
        :param kb_id: Knowledge-base ID the chunks belong to.
        :type kb_id: str
        :return: Server response list.
        :rtype: list[dict]
        """
        index, db = await self.load_data_from_memory(chunk_updates, vectors)
        return await self.integration.update_chunks(index, db, context_id=self.execution_context.id, kb_id=kb_id)

    async def load_data_from_memory(
        self,
        chunks: DocumentCollection,
        vectors: list[list[float]],
        disable_progress: bool = False,
    ) -> tuple[EncryptedIndex, EncryptedDB]:
        """Encrypt a document collection one chunk at a time.

        AES-encrypts each chunk's ``chunk_content`` and homomorphically encrypts each float
        embedding vector into an encrypted index. Results are ready to pass to :meth:`dump_db`.

        :param chunks: Document collection — each item must have a ``chunk_content`` string field.
        :type chunks: DocumentCollection
        :param vectors: Float embedding vectors, one per chunk. Each entry may also be a coroutine
            (e.g. an unawaited ``bin_embed()`` call) — it will be awaited automatically.
        :type vectors: list[list[float]]
        :param disable_progress: If ``True``, suppress the tqdm progress bar, defaults to ``False``.
        :type disable_progress: bool, optional
        :return: Tuple of ``(index, encrypted_db)`` where ``index`` contains the encrypted vectors
            and ``encrypted_db`` contains chunks with AES-encrypted content.
        :rtype: tuple[EncryptedIndex, EncryptedDB]
        """
        if len(chunks) != len(vectors):
            raise ValueError("Number of chunks and vectors must be the same")
        index = []
        encrypted_db = []
        for i in tqdm(range(len(chunks)), disable=disable_progress, desc="Encrypting", unit="chunk"):
            vec = vectors[i]
            if inspect.isawaitable(vec):
                vec = await vec
            if len(vec) != self.execution_context.embed_len():
                raise ValueError(
                    f"Vector dimension {len(vec)} does not match "
                    f"homomorphic client dimension {self.execution_context.embed_len()}"
                )
            bin_vector: list[int] = Embedding.float_2_bin(vec).tolist()
            index.append(self.execution_context.homomorphic.encrypt_vec_one(bin_vector))
            encrypted_db.append(EncryptedChunk(
                chunk_content=self.execution_context.aes.encrypt(chunks[i]["chunk_content"]),
                meta_data=chunks[i].get("meta_data", {}),
                name=chunks[i].get("name", ""),
                chunk_id=chunks[i].get("chunk_id", 0),
            ))
        return index, encrypted_db

    async def load_data_from_memory_batch(
        self,
        chunks: DocumentCollection,
        vectors: list[list[float]],
        disable_progress: bool = False,
    ) -> tuple[EncryptedIndex, EncryptedDB]:
        """Encrypt a document collection using batch homomorphic encryption.

        Faster than :meth:`load_data_from_memory` for large collections because all embedding
        vectors are passed to the homomorphic client in a single batch call instead of one
        at a time. AES encryption is still applied per chunk.

        :param chunks: Document collection — each item must have a ``chunk_content`` string field.
        :type chunks: DocumentCollection
        :param vectors: Float embedding vectors, one per chunk. Each entry may also be a coroutine
            (e.g. an unawaited ``bin_embed()`` call) — it will be awaited automatically.
        :type vectors: list[list[float]]
        :param disable_progress: Unused; kept for API compatibility with :meth:`load_data_from_memory`.
        :type disable_progress: bool, optional
        :return: Tuple of ``(index, encrypted_db)`` where ``index`` contains the encrypted vectors
            and ``encrypted_db`` contains chunks with AES-encrypted content.
        :rtype: tuple[EncryptedIndex, EncryptedDB]
        """
        if len(chunks) != len(vectors):
            raise ValueError("Number of chunks and vectors must be the same")
        encrypted_db = []
        bin_vecs = []

        with bench("ingest_aes_encrypt"):
            for i in range(len(chunks)):
                vec = vectors[i]
                if inspect.isawaitable(vec):
                    vec = await vec
                encrypted_db.append(EncryptedChunk(
                    chunk_content=self.execution_context.aes.encrypt(chunks[i]["chunk_content"]),
                    meta_data=chunks[i].get("meta_data", {}),
                    name=chunks[i].get("name", ""),
                    chunk_id=chunks[i].get("chunk_id", 0),
                ))
                bin_vecs.append(Embedding.float_2_bin(vec))

        with bench("ingest_paillier_encrypt"):
            index = self.execution_context.homomorphic.encrypt_vec_batch([v.tolist() for v in bin_vecs])

        return index, encrypted_db
