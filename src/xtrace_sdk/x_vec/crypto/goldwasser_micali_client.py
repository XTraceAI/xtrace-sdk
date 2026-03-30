import json

from xtrace_sdk.x_vec.crypto.encryption.goldwasser_micali import GoldwasserMicali


class GoldwasserMicaliClient:
    """Goldwasser-Micali homomorphic encryption client optimised for computing
    encrypted Hamming distances between binary embedding vectors.
    """

    def __init__(self, chunk_num: int = 2, embd_len: int = 512, key_len: int = 1024) -> None:
        """Generate a Goldwasser-Micali keypair and configure chunking.

        :param chunk_num: Number of ciphertexts produced per encrypted vector.
            Must evenly divide 2 * embd_len, defaults to 2.
        :type chunk_num: int, optional
        :param embd_len: Dimension of the binary embedding vectors to encrypt, defaults to 512.
        :type embd_len: int, optional
        :param key_len: RSA modulus size in bits for key generation, defaults to 1024.
        :type key_len: int, optional
        :raises Exception: If ``2 * embd_len`` is not divisible by ``chunk_num``.
        """

        if 2*embd_len % chunk_num != 0:
                 raise ValueError("chunk_num must divide 2*embd_len")

        self.keys = GoldwasserMicali.key_gen(key_len)
        self.embd_len = embd_len
        self.chunk_num = chunk_num
        self.chunk_len = self.embd_len*2 // self.chunk_num

    def serial_pk(self) -> str:
        """Serialize the public key to a JSON string for networking or storage.

        :return: JSON-encoded public key.
        :rtype: str
        """
        return json.dumps(self.keys['pk'])

    def serial_sk(self) -> str:
        """Serialize the secret key to a JSON string for networking or storage.

        :return: JSON-encoded secret key.
        :rtype: str
        """
        return json.dumps(self.keys['sk'])

    def dump_pk(self, file_path: str) -> None:
        """Save the public key to disk as JSON.

        :param file_path: Destination file path.
        :type file_path: str
        """
        with open(file_path, "w") as f:
            json.dump(self.keys['pk'], f)

    def dump_sk(self, file_path: str) -> None:
        """Save the secret key to disk as JSON.

        :param file_path: Destination file path.
        :type file_path: str
        """
        with open(file_path, "w") as f:
            json.dump(self.keys['sk'], f)

    def id2power(self, id_: int) -> tuple[int, int]:
        """Return the chunk index and bit position for a given embedding entry index.

        :param id_: Index of an entry in the padded embedding array.
        :type id_: int
        :return: Tuple of (chunk_index, bit_position).
        :rtype: tuple[int, int]
        """
        return id_ // self.chunk_len, self.chunk_len - 1 - id_%self.chunk_len

    def encrypt(self, embd: list[int]) -> list[int]:
        """Encrypt a binary embedding vector using the Goldwasser-Micali scheme.

        :param embd: Binary embedding vector of length ``embd_len`` with values in {0, 1}.
        :type embd: list[int]
        :return: List of ``chunk_num`` Goldwasser-Micali ciphertexts.
        :rtype: list[int]
        """


        return GoldwasserMicali.encrypt([int(embd[i]) for i in range(self.embd_len)], self.keys['pk'])

    def decode_hamming_client(self, cipher: list[int]) -> int:
        """Decrypt an encoded Hamming distance returned by the server.

        :param cipher: Encrypted Hamming encoding returned by the server.
        :type cipher: list[int]
        :return: Plain-text Hamming distance between the two original embeddings.
        :rtype: int
        """

        return sum(GoldwasserMicali.decrypt(cipher, self.keys))
