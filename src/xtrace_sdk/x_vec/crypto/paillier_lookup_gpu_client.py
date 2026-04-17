import json

from xtrace_sdk.x_vec.crypto._gpu_loader import load_gpu_extension
from xtrace_sdk.x_vec.crypto.hamming_client_base import HammingClientBase
from xtrace_sdk.x_vec.utils.xtrace_types import EncryptedVector


class PaillierLookupGPUClient(HammingClientBase):
    """Standalone GPU Paillier-Lookup wrapper for ExecutionContext integration and tests."""

    def __init__(
        self,
        embed_len: int = 512,
        key_len: int = 1024,
        alpha_len: int = 50,
        skip_key_gen: bool = False,
    ) -> None:
        module = load_gpu_extension("paillier_GPU_lookup_client", "paillier-GPU-lookup-client")
        self._gpu_cls = module.PaillierGPULookupClient
        self._gpu_client = self._gpu_cls(
            embed_len=embed_len,
            key_len=key_len,
            alpha_len=alpha_len,
            skip_key_gen=skip_key_gen,
        )
        self.embed_len = embed_len
        self.key_len = key_len
        self.alpha_len = alpha_len
        self.device = "gpu"

    @staticmethod
    def is_available() -> bool:
        try:
            PaillierLookupGPUClient(skip_key_gen=True)
            return True
        except Exception:
            return False

    def encrypt_vec_one(self, embd: list[int]) -> EncryptedVector:
        return [int(cipher) for cipher in self._gpu_client.encrypt([embd])[0]]

    def encrypt_vec_batch(self, embds: list[list[int]]) -> list[EncryptedVector]:
        return [[int(cipher) for cipher in row] for row in self._gpu_client.encrypt(embds)]

    def decode_hamming_client_one(self, cipher: list[int | bytes]) -> int:
        normalized = [
            int.from_bytes(value, byteorder="little") if isinstance(value, bytes) else int(value)
            for value in cipher
        ]
        return int(self._gpu_client.decode_hamming_client([normalized])[0])

    def decode_hamming_client_batch(self, ciphers: list[list[int | bytes]]) -> list[int]:
        normalized = [
            [
                int.from_bytes(value, byteorder="little") if isinstance(value, bytes) else int(value)
                for value in cipher
            ]
            for cipher in ciphers
        ]
        return [int(value) for value in self._gpu_client.decode_hamming_client(normalized)]

    def stringify_sk(self) -> str:
        return self._gpu_client.stringify_sk()

    def stringify_pk(self) -> str:
        return self._gpu_client.stringify_pk()

    def stringify_config(self) -> str:
        return self._gpu_client.stringify_config()

    def load_stringified_keys(self, pk: str, sk: str) -> None:
        self._gpu_client.load_stringified_keys(pk, sk)

    def load_config(self, config: dict, precomputed_tables: dict | None = None) -> None:
        self._gpu_client.load_config({k: v for k, v in config.items() if k != "device"})
        self.embed_len = config["embed_len"]
        self.key_len = config["key_len"]
        self.alpha_len = config.get("alpha_len", self.alpha_len)

    def encode_hamming_server(self, ct1: list[int], ct2: list[int]) -> EncryptedVector:
        result = self._gpu_cls.encode_hamming_server(ct1, ct2, json.loads(self.stringify_pk()))
        return [int(cipher) for cipher in result]
