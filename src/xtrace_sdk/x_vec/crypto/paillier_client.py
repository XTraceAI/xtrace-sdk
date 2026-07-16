import json
from typing import Any

import gmpy2

from xtrace_sdk.x_vec.crypto.device import DeviceMode, resolve_device
from xtrace_sdk.x_vec.crypto.encryption.paillier import Paillier
from xtrace_sdk.x_vec.crypto.hamming_client_base import HammingClientBase
from xtrace_sdk.x_vec.utils.xtrace_types import EncryptedVector, PaillierEncryptedNumber, PaillierKeyPair


def _load_gpu_backend() -> type[Any]:
    try:
        from xtrace_sdk.x_vec.crypto.paillier_gpu_ext import PaillierGPUClient
    except ImportError as e:
        raise ImportError(
            "GPU extension not built. Run `./build_gpu_binaries.sh` "
            "(requires Docker, NVIDIA driver >= 550)."
        ) from e
    return PaillierGPUClient


class PaillierCPU:
    """CPU implementation of Paillier homomorphic encryption for Hamming-distance search."""

    def __init__(self, embed_len: int = 512, key_len: int = 1024, skip_key_gen: bool = False) -> None:
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
        if self.keys is None:
            raise RuntimeError("Keys not initialized")
        pk = self.keys["pk"]
        return json.dumps({"g": str(pk["g"]), "n": str(pk["n"]), "n_squared": str(pk["n_squared"])})

    def stringify_sk(self) -> str:
        if self.keys is None:
            raise RuntimeError("Keys not initialized")
        sk = self.keys["sk"]
        return json.dumps({"phi": str(sk["phi"]), "inv": str(sk["inv"])})

    def stringify_config(self) -> str:
        return json.dumps({"embed_len": self.embed_len, "key_len": self.key_len})

    def load_stringified_keys(self, pk: str, sk: str) -> None:
        pk_data = json.loads(pk)
        sk_data = json.loads(sk)
        self.keys = {
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

    def load_config(self, config: dict) -> None:
        self.embed_len = config["embed_len"]
        self.key_len = config["key_len"]
        self.chunk_len = self.key_len * 2
        if self.embed_len > self.key_len:
            self.chunk_num = 2 * self.embed_len // self.chunk_len + int((2 * self.embed_len) % self.chunk_len != 0)
        else:
            self.chunk_num = 1

    def encrypt(self, embd: list[int]) -> PaillierEncryptedNumber:
        if len(embd) != self.embed_len:
            raise ValueError(f"Embedding length {len(embd)} does not match expected {self.embed_len}")
        if self.keys is None:
            raise RuntimeError("Keys not initialized")
        padded_embd: list[str] = []
        for i in range(self.embed_len):
            if embd[i] not in (0, 1):
                raise ValueError(f"Embedding vector must be binary, got {embd[i]} at index {i}")
            padded_embd += ["0", str(embd[i])]
        int_repr = [
            gmpy2.mpz(int("".join(padded_embd[i * self.chunk_len : (i + 1) * self.chunk_len]), 2))
            for i in range(self.chunk_num)
        ]
        return [Paillier.encrypt(i, self.keys["pk"]) for i in int_repr]

    def decode_hamming_client(self, cipher: list[int | bytes]) -> int:
        if self.keys is None:
            raise RuntimeError("Keys not initialized")
        decrypted = [
            Paillier.decrypt(int.from_bytes(c, byteorder="little") if isinstance(c, bytes) else c, self.keys)
            for c in cipher
        ]
        bin_chunks = [f"{value:b}" for value in decrypted]
        bin_string = ""
        for chunk in bin_chunks:
            if len(chunk) != self.chunk_len and len(chunk) != 0:
                chunk = "0" * (self.chunk_len - len(chunk)) + chunk
            bin_string += chunk
        return sum(int(bin_string[i]) for i in range(1, len(bin_string), 2))


class PaillierClient(HammingClientBase):
    """Paillier client that dispatches to CPU or GPU based on the ``device`` kwarg.

    With ``device="auto"`` (the default), the GPU extension is probed at
    construction time and used if available, otherwise the CPU implementation
    runs. ``"cpu"`` and ``"gpu"`` force a backend; ``"gpu"`` raises if the
    extension cannot be loaded.
    """

    def __init__(
        self,
        embed_len: int = 512,
        key_len: int = 1024,
        skip_key_gen: bool = False,
        device: DeviceMode = "auto",
    ) -> None:
        self.embed_len = embed_len
        self.key_len = key_len
        self.chunk_len = key_len * 2
        if embed_len > key_len:
            self.chunk_num = 2 * embed_len // self.chunk_len + int((2 * embed_len) % self.chunk_len != 0)
        else:
            self.chunk_num = 1

        self.device, gpu_cls = resolve_device(device, _load_gpu_backend)
        if self.device == "gpu" and gpu_cls is not None:
            self.client: Any = gpu_cls(embed_len=embed_len, key_len=key_len, skip_key_gen=skip_key_gen)
        else:
            self.client = PaillierCPU(embed_len=embed_len, key_len=key_len, skip_key_gen=skip_key_gen)

    @staticmethod
    def has_gpu() -> bool:
        try:
            _load_gpu_backend()(skip_key_gen=True)
            return True
        except Exception:
            return False

    @staticmethod
    def _cipher_to_int(value: Any) -> int:
        if isinstance(value, bytes):
            return int.from_bytes(value, byteorder="little")
        if isinstance(value, str):
            return int(value, 16)
        return int(value)

    @staticmethod
    def _cipher_to_hex(value: int | bytes) -> str:
        if isinstance(value, bytes):
            return format(int.from_bytes(value, byteorder="little"), "x")
        return format(int(value), "x")

    def encrypt_vec_one(self, embd: list[int]) -> EncryptedVector:
        if isinstance(self.client, PaillierCPU):
            return self.client.encrypt(embd)
        row = self.client.encrypt([embd])[0]
        return [self._cipher_to_int(cipher) for cipher in row]

    def encrypt_vec_batch(self, embds: list[list[int]]) -> list[EncryptedVector]:
        if isinstance(self.client, PaillierCPU):
            return [self.client.encrypt(embd) for embd in embds]
        return [[self._cipher_to_int(cipher) for cipher in row] for row in self.client.encrypt(embds)]

    def decode_hamming_client_one(self, cipher: list[int | bytes]) -> int:
        if isinstance(self.client, PaillierCPU):
            return self.client.decode_hamming_client(cipher)
        return int(self.client.decode_hamming_client([[self._cipher_to_hex(value) for value in cipher]])[0])

    def decode_hamming_client_batch(self, ciphers: list[list[int | bytes]]) -> list[int]:
        if isinstance(self.client, PaillierCPU):
            return [self.client.decode_hamming_client(cipher) for cipher in ciphers]
        return [
            int(value)
            for value in self.client.decode_hamming_client(
                [[self._cipher_to_hex(entry) for entry in cipher] for cipher in ciphers]
            )
        ]

    def stringify_pk(self) -> str:
        return self.client.stringify_pk()

    def stringify_sk(self) -> str:
        return self.client.stringify_sk()

    def stringify_config(self) -> str:
        return self.client.stringify_config()

    def load_stringified_keys(self, pk: str, sk: str) -> None:
        self.client.load_stringified_keys(pk, sk)

    def load_config(self, config: dict) -> None:
        sanitized_config = {k: v for k, v in config.items() if k != "device"}
        self.client.load_config(sanitized_config)
        self.embed_len = config["embed_len"]
        self.key_len = config["key_len"]
        self.chunk_len = self.key_len * 2
        if self.embed_len > self.key_len:
            self.chunk_num = 2 * self.embed_len // self.chunk_len + int((2 * self.embed_len) % self.chunk_len != 0)
        else:
            self.chunk_num = 1

    def encode_hamming_server(self, ct1: list[int | bytes], ct2: list[int | bytes]) -> EncryptedVector:
        if len(ct1) != len(ct2):
            raise ValueError("ct1 and ct2 must have the same length")
        if isinstance(self.client, PaillierCPU):
            if self.client.keys is None:
                raise RuntimeError("Keys not initialized")
            return [
                int(Paillier.add(self._cipher_to_int(a), self._cipher_to_int(b), self.client.keys["pk"]))
                for a, b in zip(ct1, ct2, strict=True)
            ]

        pk = json.loads(self.stringify_pk())
        result = type(self.client).encode_hamming_server(
            [self._cipher_to_hex(cipher) for cipher in ct1],
            [self._cipher_to_hex(cipher) for cipher in ct2],
            {"n_squared": format(int(pk["n_squared"]), "x")},
        )
        return [self._cipher_to_int(cipher) for cipher in result]
