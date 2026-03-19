import json
import os
from typing import Union, cast

import gmpy2

from xtrace_sdk.x_vec.crypto.encryption.paillier import Paillier
from xtrace_sdk.x_vec.crypto.hamming_client_base import HammingClientBase
from xtrace_sdk.x_vec.utils.xtrace_types import PaillierEncryptedNumber, PaillierKeyPair

DEVICE = os.getenv("DEVICE", "cpu")


class PaillierGPU:
    """Wrapper for the C++ GPU implementation of the Paillier cryptography system.

    This class is not instantiated directly — use :class:`PaillierClient` which dispatches
    to this backend automatically when ``DEVICE=gpu``.
    """

    def __init__(self, embed_len: int = 512, key_len: int = 1024, skip_key_gen: bool = False) -> None:
        try:
            from .paillier_GPU_client import PaillierGPUClient
        except Exception as e:
            raise ImportError(
                "GPU backend unavailable. Expected Paillier GPU .so under src/crypto/ "
                "Please refer to the documentation for compilation instructions."
            ) from e
        self.gpu_client = PaillierGPUClient(embed_len=embed_len, key_len=key_len, skip_key_gen=skip_key_gen)

    def encrypt(self, embds: list[list[int]]) -> list[PaillierEncryptedNumber]:
        """Encrypt a batch of binary embedding vectors on GPU.

        :param embds: List of binary vectors with values in {0, 1}.
        :type embds: list[list[int]]
        :return: List of encrypted vectors.
        :rtype: list[PaillierEncryptedNumber]
        """
        bytes_batch = self.gpu_client.encrypt_bytes(embds)
        return [[int.from_bytes(b, byteorder='little') for b in bytes_list] for bytes_list in bytes_batch]

    def decode_hamming_client(self, cipher: list) -> list[int]:
        """Decrypt a batch of encrypted Hamming distances on GPU.

        :param cipher: Batch of encrypted Hamming encodings from the server.
        :type cipher: list
        :return: List of plain-text Hamming distances.
        :rtype: list[int]
        """
        chunk_byte_len = (self.gpu_client.key_len * 4) // 8
        bytes_batch = []
        for c in cipher:
            record_bytes = []
            for h in c:
                if isinstance(h, bytes):
                    if len(h) != chunk_byte_len:
                        h = int.from_bytes(h, byteorder='little').to_bytes(chunk_byte_len, byteorder='little')
                    record_bytes.append(h)
                else:
                    record_bytes.append(int(h).to_bytes(chunk_byte_len, byteorder='little'))
            bytes_batch.append(record_bytes)
        return self.gpu_client.decode_hamming_client_bytes(bytes_batch)

    def stringify_pk(self) -> str:
        """Return a JSON string representation of the public key."""
        return self.gpu_client.stringify_pk()

    def stringify_sk(self) -> str:
        """Return a JSON string representation of the secret key."""
        return self.gpu_client.stringify_sk()

    def stringify_config(self) -> str:
        """Return a JSON string representation of the encryption configuration."""
        return self.gpu_client.stringify_config()

    def load_stringified_keys(self, pk: str, sk: str) -> None:
        """Load public and secret keys from their JSON string representations.

        :param pk: JSON-encoded public key produced by :meth:`stringify_pk`.
        :param sk: JSON-encoded secret key produced by :meth:`stringify_sk`.
        """
        self.gpu_client.load_stringified_keys(pk, sk)

    def load_config(self, config: dict) -> None:
        """Load encryption configuration from a dict produced by :meth:`stringify_config`.

        :param config: Configuration dict (``embed_len``, ``key_len``).
        :type config: dict
        """
        self.gpu_client.load_config(config)


class PaillierCPU:
    """CPU implementation of Paillier homomorphic encryption optimised for Hamming distance."""

    def __init__(self, embed_len: int = 512, key_len: int = 1024, skip_key_gen: bool = False) -> None:
        """Generate a Paillier keypair and configure chunking parameters.

        :param embed_len: Dimension of the binary embedding vectors to encrypt, defaults to 512.
        :type embed_len: int, optional
        :param key_len: RSA modulus size in bits for key generation, defaults to 1024.
        :type key_len: int, optional
        :param skip_key_gen: If True, skip key generation (keys must be loaded later via
            ``load_stringified_keys``), defaults to False.
        :type skip_key_gen: bool, optional
        """
        self.chunk_len = key_len * 2
        self.key_len = key_len
        if embed_len > key_len:
            self.chunk_num = 2 * embed_len // self.chunk_len + int((2 * embed_len) % self.chunk_len != 0)
        else:
            self.chunk_num = 1

        self.keys: PaillierKeyPair | None
        if not skip_key_gen:
            self.keys = Paillier.key_gen(key_len)
        else:
            self.keys = None

        self.embed_len = embed_len

    def stringify_pk(self) -> str:
        """Return a JSON string representation of the public key."""
        assert self.keys is not None, "Keys not initialized"
        pk = self.keys['pk']
        return json.dumps({'g': str(pk['g']), 'n': str(pk['n']), 'n_squared': str(pk['n_squared'])})

    def stringify_sk(self) -> str:
        """Return a JSON string representation of the secret key."""
        assert self.keys is not None, "Keys not initialized"
        sk = self.keys['sk']
        return json.dumps({'phi': str(sk['phi']), 'inv': str(sk['inv'])})

    def stringify_config(self) -> str:
        """Return a JSON string representation of the encryption configuration."""
        return json.dumps({"embed_len": self.embed_len, "key_len": self.key_len})

    def load_stringified_keys(self, pk: str, sk: str) -> None:
        """Load public and secret keys from their JSON string representations.

        :param pk: JSON-encoded public key produced by :meth:`stringify_pk`.
        :param sk: JSON-encoded secret key produced by :meth:`stringify_sk`.
        """
        pk_data = json.loads(pk)
        sk_data = json.loads(sk)
        self.keys = {
            'pk': {
                'g': gmpy2.mpz(pk_data['g']),
                'n': gmpy2.mpz(pk_data['n']),
                'n_squared': gmpy2.mpz(pk_data['n_squared']),
            },
            'sk': {
                'phi': gmpy2.mpz(sk_data['phi']),
                'inv': gmpy2.mpz(sk_data['inv']),
            },
        }

    def load_config(self, config: dict) -> None:
        """Load encryption configuration from a dict produced by :meth:`stringify_config`.

        :param config: Configuration dict (``embed_len``, ``key_len``).
        :type config: dict
        """
        self.embed_len = config['embed_len']
        self.key_len = config['key_len']

    def id2power(self, id_: int) -> tuple[int, int]:
        """Return the chunk index and bit power for a given position in the padded embedding.

        :param id_: Index of an entry in the embedding vector.
        :type id_: int
        :return: Tuple of (chunk_index, bit_power).
        :rtype: tuple[int, int]
        """
        return id_ // self.chunk_len, self.chunk_len - 1 - id_ % self.chunk_len

    def encrypt(self, embd: list[int]) -> PaillierEncryptedNumber:
        """Encrypt a single binary embedding vector.

        :param embd: Binary vector of length ``embed_len`` with values in {0, 1}.
        :type embd: list[int]
        :return: List of Paillier ciphertexts (one per chunk).
        :rtype: PaillierEncryptedNumber
        """
        assert len(embd) == self.embed_len
        assert self.keys is not None, "Keys not initialized"
        padded_embd = []
        for i in range(self.embed_len):
            assert embd[i] in [0, 1], "Embedding vector must be binary"
            padded_embd += ['0', str(embd[i])]
        int_repr = [
            gmpy2.mpz(int("".join(padded_embd[i * self.chunk_len:(i + 1) * self.chunk_len]), 2))
            for i in range(self.chunk_num)
        ]
        return [Paillier.encrypt(i, self.keys['pk']) for i in int_repr]

    def decode_hamming_client(self, cipher: list[int | bytes]) -> int:
        """Decrypt an encoded Hamming distance returned by the XTrace server.

        :param cipher: Encrypted Hamming encoding as returned by the server.
        :type cipher: PaillierEncryptedNumber
        :return: Plain-text Hamming distance.
        :rtype: int
        """
        assert self.keys is not None, "Keys not initialized"
        de_c = [Paillier.decrypt(int.from_bytes(c, byteorder='little') if isinstance(c, bytes) else c, self.keys) for c in cipher]
        bin_c_truncated = [f"{c:b}" for c in de_c]
        bin_c_str = ""
        for c in bin_c_truncated:
            if len(c) != self.chunk_len and len(c) != 0:
                c = "0" * (self.chunk_len - len(c)) + c
            bin_c_str += c
        ham = 0
        for i in range(1, len(bin_c_str), 2):
            ham += int(bin_c_str[i])
        return ham


class PaillierClient(HammingClientBase):
    """Paillier homomorphic encryption client optimised for computing encrypted Hamming distances
    between binary embedding vectors. Dispatches to CPU or GPU backend via the ``DEVICE``
    environment variable (``cpu`` by default).
    """

    def __init__(self, embed_len: int = 512, key_len: int = 1024, skip_key_gen: bool = False) -> None:
        """Generate a Paillier keypair and configure the client.

        :param embed_len: Dimension of the binary embedding vectors to encrypt, defaults to 512.
        :type embed_len: int, optional
        :param key_len: RSA modulus size in bits for key generation, defaults to 1024.
        :type key_len: int, optional
        :param skip_key_gen: If True, skip key generation (keys must be loaded later via
            ``load_stringified_keys``), defaults to False.
        :type skip_key_gen: bool, optional
        """
        self.embed_len = embed_len
        self.key_len = key_len
        self.chunk_len = key_len * 2
        if embed_len > key_len:
            self.chunk_num = 2 * embed_len // self.chunk_len + int((2 * embed_len) % self.chunk_len != 0)
        else:
            self.chunk_num = 1

        self.device = DEVICE
        self.client: PaillierGPU | PaillierCPU
        if self.device == "gpu":
            if not self.has_gpu():
                raise RuntimeError("GPU requested via DEVICE=gpu, but GPU backend is unavailable.")
            self.client = PaillierGPU(embed_len=embed_len, key_len=key_len, skip_key_gen=skip_key_gen)
        else:
            self.client = PaillierCPU(embed_len=embed_len, key_len=key_len, skip_key_gen=skip_key_gen)

    @staticmethod
    def has_gpu() -> bool:
        """Return ``True`` if the GPU backend extension is available and loadable.

        :rtype: bool
        """
        try:
            from xtrace_sdk.x_vec.crypto.paillier_client import PaillierGPU
            PaillierGPU(skip_key_gen=True)
            return True
        except Exception:
            return False

    def encrypt_vec_one(self, embd: list[int]) -> PaillierEncryptedNumber:
        """Encrypt a single binary embedding vector.

        :param embd: Binary vector of length ``embed_len`` with values in {0, 1}.
        :type embd: list[int]
        :return: Encrypted vector.
        :rtype: PaillierEncryptedNumber
        """
        if isinstance(self.client, PaillierCPU):
            return self.client.encrypt(embd)
        else:
            return self.client.encrypt([embd])[0]

    def encrypt_vec_batch(self, embds: list[list[int]]) -> list[PaillierEncryptedNumber]:
        """Encrypt a batch of binary embedding vectors.

        :param embds: List of binary vectors, each of length ``embed_len`` with values in {0, 1}.
        :type embds: list[list[int]]
        :return: List of encrypted vectors.
        :rtype: list[PaillierEncryptedNumber]
        """
        if isinstance(self.client, PaillierCPU):
            return [self.client.encrypt(embd) for embd in embds]
        else:
            return self.client.encrypt(embds)

    def decode_hamming_client_one(self, cipher: list[int | bytes]) -> int:
        """Decrypt a single encrypted Hamming distance returned by the XTrace server.

        :param cipher: Encrypted Hamming encoding.
        :type cipher: PaillierEncryptedNumber
        :return: Plain-text Hamming distance.
        :rtype: int
        """
        if isinstance(self.client, PaillierCPU):
            return self.client.decode_hamming_client(cipher)
        else:
            return self.client.decode_hamming_client([cipher])[0]

    def decode_hamming_client_batch(self, ciphers: list[list[int | bytes]]) -> list[int]:
        """Decrypt a batch of encrypted Hamming distances returned by the XTrace server.

        :param ciphers: List of encrypted Hamming encodings.
        :type ciphers: list[PaillierEncryptedNumber]
        :return: List of plain-text Hamming distances.
        :rtype: list[int]
        """
        if isinstance(self.client, PaillierCPU):
            return [self.client.decode_hamming_client(c) for c in ciphers]
        else:
            return self.client.decode_hamming_client(ciphers)

    def stringify_pk(self) -> str:
        """Return a JSON string representation of the public key."""
        return self.client.stringify_pk()

    def stringify_sk(self) -> str:
        """Return a JSON string representation of the secret key."""
        return self.client.stringify_sk()

    def stringify_config(self) -> str:
        """Return a JSON string representation of the encryption configuration."""
        return self.client.stringify_config()

    def load_stringified_keys(self, pk: str, sk: str) -> None:
        """Load public and secret keys from their JSON string representations.

        :param pk: JSON-encoded public key produced by :meth:`stringify_pk`.
        :param sk: JSON-encoded secret key produced by :meth:`stringify_sk`.
        """
        self.client.load_stringified_keys(pk, sk)

    def load_config(self, config: dict) -> None:
        """Load encryption configuration from a dict produced by :meth:`stringify_config`.

        :param config: Configuration dict (``embed_len``, ``key_len``).
        :type config: dict
        """
        self.client.load_config(config)
