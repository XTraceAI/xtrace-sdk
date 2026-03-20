from abc import ABC, abstractmethod

from xtrace_sdk.x_vec.utils.xtrace_types import EncryptedVector


class HammingClientBase(ABC):
    """Abstract base class for homomorphic encryption clients used in encrypted Hamming distance search.

    Concrete implementations (``PaillierClient``, ``PaillierLookupClient``) encrypt binary embedding
    vectors on the client side, send the ciphertexts to XTrace, and later decrypt the encoded
    Hamming distances returned by the server — without the server ever seeing plaintext vectors
    or distances.
    """

    @abstractmethod
    def encrypt_vec_one(self, embd: list[int]) -> EncryptedVector:
        """Encrypt a single binary embedding vector.

        :param embd: Binary vector of length ``embed_len`` with values in {0, 1}.
        :type embd: list[int]
        :return: Encrypted representation of the vector.
        :rtype: EncryptedVector
        """
        ...

    @abstractmethod
    def encrypt_vec_batch(self, embds: list[list[int]]) -> list[EncryptedVector]:
        """Encrypt a batch of binary embedding vectors.

        :param embds: List of binary vectors, each of length ``embed_len`` with values in {0, 1}.
        :type embds: list[list[int]]
        :return: List of encrypted representations, one per input vector.
        :rtype: list[EncryptedVector]
        """
        ...

    @abstractmethod
    def decode_hamming_client_one(self, cipher: list[int | bytes]) -> int:
        """Decrypt a single encrypted Hamming distance returned by the XTrace server.

        :param cipher: Encrypted Hamming encoding as returned by the server.
        :type cipher: list[int]
        :return: Plain-text Hamming distance.
        :rtype: int
        """
        ...

    @abstractmethod
    def decode_hamming_client_batch(self, ciphers: list[list[int | bytes]]) -> list[int]:
        """Decrypt a batch of encrypted Hamming distances returned by the XTrace server.

        :param ciphers: List of encrypted Hamming encodings as returned by the server.
        :type ciphers: list[list[int]]
        :return: List of plain-text Hamming distances, one per input cipher.
        :rtype: list[int]
        """
        ...
