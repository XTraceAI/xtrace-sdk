from collections.abc import Sequence
from typing import TypeAlias, TypedDict

import gmpy2


# ── Paillier (regular) ─────────────────────────────────────────────────────

class PaillierPublicKey(TypedDict):
    g: gmpy2.mpz
    n: gmpy2.mpz
    n_squared: gmpy2.mpz


class PaillierSecretKey(TypedDict):
    phi: gmpy2.mpz
    inv: gmpy2.mpz


class PaillierKeyPair(TypedDict):
    pk: PaillierPublicKey
    sk: PaillierSecretKey


# ── Paillier Lookup ────────────────────────────────────────────────────────

class PaillierLookupPublicKey(TypedDict):
    g: gmpy2.mpz
    n: gmpy2.mpz
    n_squared: gmpy2.mpz
    g_n: gmpy2.mpz


class PaillierLookupSecretKey(TypedDict):
    phi: gmpy2.mpz
    a: gmpy2.mpz
    g_a_inv: gmpy2.mpz


class PaillierLookupKeyPair(TypedDict):
    pk: PaillierLookupPublicKey
    sk: PaillierLookupSecretKey
    g_table: dict[int, list[int]]
    noise_table: list[int]
    key_len: int
    message_chunks: int


# ── Goldwasser-Micali ──────────────────────────────────────────────────────

class GoldwasserMicaliPublicKey(TypedDict):
    x: int
    n: int


class GoldwasserMicaliSecretKey(TypedDict):
    p: int
    q: int


class GoldwasserMicaliKeyPair(TypedDict):
    pk: GoldwasserMicaliPublicKey
    sk: GoldwasserMicaliSecretKey


# ── Encrypted vector type ──────────────────────────────────────────────────

PaillierEncryptedNumber: TypeAlias = list[int]


# ── Chunk / collection types ───────────────────────────────────────────────

class MetaData(TypedDict, total=False):
    tag1: str
    tag2: str
    tag3: str
    tag4: str
    tag5: str
    facets: list[str]


class Chunk(TypedDict, total=False):
    chunk_content: str | bytes  # str for plaintext input; bytes after AES encryption
    meta_data: MetaData
    name: str
    chunk_id: int


EncryptedDB: TypeAlias = list[Chunk]
DocumentCollection: TypeAlias = Sequence[Chunk]
