import json

import gmpy2

from xtrace_sdk.x_vec.crypto._gpu_loader import load_gpu_extension
from xtrace_sdk.x_vec.crypto.encryption.paillier import Paillier
from xtrace_sdk.x_vec.crypto.hamming_client_base import HammingClientBase
from xtrace_sdk.x_vec.utils.xtrace_types import EncryptedVector, PaillierKeyPair


def _build_plain_paillier_keys(pk: str, sk: str) -> PaillierKeyPair:
    pk_data = json.loads(pk)
    sk_data = json.loads(sk)
    return {
        "pk": {
            "g": gmpy2.mpz(pk_data["g"]),
            "n": gmpy2.mpz(pk_data["n"]),
            "n_squared": gmpy2.mpz(pk_data["n_squared"]),
        },
        "sk": {
            "phi": gmpy2.mpz(sk_data["phi"]),
            "inv": gmpy2.mpz(sk_data["inv"]),
        },
    }


class PaillierGPUClient(HammingClientBase):
    """Standalone GPU Paillier wrapper for ExecutionContext integration and tests."""

    def __init__(self, embed_len: int = 512, key_len: int = 1024, skip_key_gen: bool = False) -> None:
        module = load_gpu_extension("paillier_GPU_client", "paillier-GPU-client")
        self._gpu_cls = module.PaillierGPUClient
        self._gpu_client = self._gpu_cls(embed_len=embed_len, key_len=key_len, skip_key_gen=skip_key_gen)
        self.embed_len = embed_len
        self.key_len = key_len
        self.chunk_len = key_len * 2
        if embed_len > key_len:
            self.chunk_num = 2 * embed_len // self.chunk_len + int((2 * embed_len) % self.chunk_len != 0)
        else:
            self.chunk_num = 1
        self.device = "gpu"
        self._backend_has_keys = not skip_key_gen
        self._cpu_keys: PaillierKeyPair | None = None
        self._pk_string: str | None = None
        self._sk_string: str | None = None
        if not skip_key_gen:
            self._sync_serialized_keys_from_gpu()

    @staticmethod
    def is_available() -> bool:
        try:
            PaillierGPUClient(skip_key_gen=True)
            return True
        except Exception:
            return False

    def _sync_serialized_keys_from_gpu(self) -> None:
        n_hex, n_squared_hex, phi_hex, inv_hex = self._gpu_client.get_keys_hex()
        n = int(n_hex, 16)
        n_squared = int(n_squared_hex, 16)
        self._pk_string = json.dumps(
            {
                "g": str(n + 1),
                "n": str(n),
                "n_squared": str(n_squared),
            }
        )
        self._sk_string = json.dumps(
            {
                "phi": str(int(phi_hex, 16)),
                "inv": str(int(inv_hex, 16)),
            }
        )
        self._cpu_keys = _build_plain_paillier_keys(self._pk_string, self._sk_string)

    def _require_cpu_keys(self) -> PaillierKeyPair:
        if self._cpu_keys is None:
            raise RuntimeError("Keys not initialized")
        return self._cpu_keys

    def _encrypt_with_cpu(self, embd: list[int]) -> EncryptedVector:
        keys = self._require_cpu_keys()
        if len(embd) != self.embed_len:
            raise ValueError(f"Embedding length {len(embd)} does not match expected {self.embed_len}")
        padded_embd: list[str] = []
        for i, bit in enumerate(embd):
            if bit not in (0, 1):
                raise ValueError(f"Embedding vector must be binary, got {bit} at index {i}")
            padded_embd += ["0", str(bit)]
        int_repr = [
            gmpy2.mpz(int("".join(padded_embd[i * self.chunk_len : (i + 1) * self.chunk_len]), 2))
            for i in range(self.chunk_num)
        ]
        return [int(Paillier.encrypt(value, keys["pk"])) for value in int_repr]

    def _decode_with_cpu(self, cipher: list[int | bytes]) -> int:
        keys = self._require_cpu_keys()
        decrypted = [
            Paillier.decrypt(int.from_bytes(c, byteorder="little") if isinstance(c, bytes) else int(c), keys)
            for c in cipher
        ]
        bin_chunks = [f"{value:b}" for value in decrypted]
        bin_string = ""
        for chunk in bin_chunks:
            if len(chunk) != self.chunk_len and len(chunk) != 0:
                chunk = "0" * (self.chunk_len - len(chunk)) + chunk
            bin_string += chunk
        return sum(int(bin_string[i]) for i in range(1, len(bin_string), 2))

    def encrypt_vec_one(self, embd: list[int]) -> EncryptedVector:
        if not self._backend_has_keys:
            return self._encrypt_with_cpu(embd)
        return [int(cipher, 16) for cipher in self._gpu_client.encrypt([embd])[0]]

    def encrypt_vec_batch(self, embds: list[list[int]]) -> list[EncryptedVector]:
        if not self._backend_has_keys:
            return [self._encrypt_with_cpu(embd) for embd in embds]
        return [[int(cipher, 16) for cipher in row] for row in self._gpu_client.encrypt(embds)]

    def decode_hamming_client_one(self, cipher: list[int | bytes]) -> int:
        if not self._backend_has_keys:
            return self._decode_with_cpu(cipher)
        hex_cipher = [
            format(
                int.from_bytes(value, byteorder="little") if isinstance(value, bytes) else int(value),
                "x",
            )
            for value in cipher
        ]
        return int(self._gpu_client.decode_hamming_client([hex_cipher])[0])

    def decode_hamming_client_batch(self, ciphers: list[list[int | bytes]]) -> list[int]:
        return [self.decode_hamming_client_one(cipher) for cipher in ciphers]

    def stringify_sk(self) -> str:
        if self._sk_string is None:
            raise RuntimeError("Keys not initialized")
        return self._sk_string

    def stringify_pk(self) -> str:
        if self._pk_string is None:
            raise RuntimeError("Keys not initialized")
        return self._pk_string

    def stringify_config(self) -> str:
        return json.dumps({"embed_len": self.embed_len, "key_len": self.key_len})

    def load_stringified_keys(self, pk: str, sk: str) -> None:
        self._pk_string = pk
        self._sk_string = sk
        self._cpu_keys = _build_plain_paillier_keys(pk, sk)
        self._backend_has_keys = False

    def load_config(self, config: dict) -> None:
        self.embed_len = config["embed_len"]
        self.key_len = config["key_len"]
        self.chunk_len = self.key_len * 2
        if self.embed_len > self.key_len:
            self.chunk_num = 2 * self.embed_len // self.chunk_len + int((2 * self.embed_len) % self.chunk_len != 0)
        else:
            self.chunk_num = 1

    def encode_hamming_server(self, ct1: list[int], ct2: list[int]) -> EncryptedVector:
        if not self._backend_has_keys:
            keys = self._require_cpu_keys()
            return [int(Paillier.add(int(a), int(b), keys["pk"])) for a, b in zip(ct1, ct2, strict=True)]
        pk_data = json.loads(self.stringify_pk())
        result = self._gpu_cls.encode_hamming_server(
            [format(int(cipher), "x") for cipher in ct1],
            [format(int(cipher), "x") for cipher in ct2],
            {"n_squared": format(int(pk_data["n_squared"]), "x")},
        )
        return [int(cipher, 16) for cipher in result]
