import json

import pytest

from xtrace_sdk.x_vec.crypto.paillier_gpu_client import PaillierGPUClient
from xtrace_sdk.x_vec.crypto.paillier_lookup_client import PaillierLookupCPU
from xtrace_sdk.x_vec.crypto.paillier_lookup_gpu_client import PaillierLookupGPUClient
from xtrace_sdk.x_vec.utils.execution_context import ExecutionContext


_PASSPHRASE = "test-gpu-exec-ctx-passphrase"
_EMBED_LEN = 8
_KEY_LEN = 64
_VECTORS = [
    [0, 1, 0, 1, 1, 0, 1, 0],
    [1, 1, 0, 0, 1, 0, 0, 1],
    [1, 0, 1, 0, 1, 1, 0, 0],
]


def _hamming(lhs: list[int], rhs: list[int]) -> int:
    return sum(int(a != b) for a, b in zip(lhs, rhs, strict=True))


def _clone_client(client_type: str) -> PaillierGPUClient | PaillierLookupGPUClient:
    if client_type == "paillier_gpu":
        return PaillierGPUClient(embed_len=_EMBED_LEN, key_len=_KEY_LEN, skip_key_gen=True)
    return PaillierLookupGPUClient(embed_len=_EMBED_LEN, key_len=_KEY_LEN, skip_key_gen=True)


@pytest.mark.parametrize(
    ("client_type", "client_cls"),
    [
        pytest.param("paillier_gpu", PaillierGPUClient, id="paillier_gpu"),
        pytest.param("paillier_lookup_gpu", PaillierLookupGPUClient, id="paillier_lookup_gpu"),
    ],
)
def test_gpu_clients_expose_execution_context_protocol(
    client_type: str,
    client_cls: type[PaillierGPUClient] | type[PaillierLookupGPUClient],
) -> None:
    if not client_cls.is_available():
        pytest.skip(f"{client_type} backend is unavailable on this machine")

    ctx = ExecutionContext.create(
        passphrase=_PASSPHRASE,
        homomorphic_client_type=client_type,
        embedding_length=_EMBED_LEN,
        key_len=_KEY_LEN,
    )
    client = ctx.homomorphic

    assert ctx.device == "gpu"
    assert ctx.embed_len() == _EMBED_LEN
    assert ctx.key_len() == _KEY_LEN

    pk = client.stringify_pk()
    sk = client.stringify_sk()
    config = client.stringify_config()
    config_dict = json.loads(config)

    assert json.loads(pk)
    assert json.loads(sk)
    assert config_dict["embed_len"] == _EMBED_LEN
    assert config_dict["key_len"] == _KEY_LEN

    cipher_one = client.encrypt_vec_one(_VECTORS[0])
    batch_ciphers = client.encrypt_vec_batch(_VECTORS)

    assert len(batch_ciphers) == len(_VECTORS)
    assert len(cipher_one) == len(batch_ciphers[0]) > 0

    encoded_one = client.encode_hamming_server(cipher_one, batch_ciphers[1])
    assert client.decode_hamming_client_one(encoded_one) == _hamming(_VECTORS[0], _VECTORS[1])

    encoded_batch = [
        client.encode_hamming_server(batch_ciphers[0], batch_ciphers[1]),
        client.encode_hamming_server(batch_ciphers[1], batch_ciphers[2]),
    ]
    assert client.decode_hamming_client_batch(encoded_batch) == [
        _hamming(_VECTORS[0], _VECTORS[1]),
        _hamming(_VECTORS[1], _VECTORS[2]),
    ]

    clone = _clone_client(client_type)
    clone.load_stringified_keys(pk, sk)
    clone.load_config(config_dict)

    clone_cipher = clone.encrypt_vec_one(_VECTORS[0])
    clone_encoded = clone.encode_hamming_server(clone_cipher, clone.encrypt_vec_one(_VECTORS[1]))

    assert json.loads(clone.stringify_pk()) == json.loads(pk)
    assert json.loads(clone.stringify_sk()) == json.loads(sk)
    assert json.loads(clone.stringify_config()) == config_dict
    assert clone.decode_hamming_client_one(clone_encoded) == _hamming(_VECTORS[0], _VECTORS[1])


@pytest.mark.parametrize(
    ("client_type", "client_cls"),
    [
        pytest.param("paillier_gpu", PaillierGPUClient, id="paillier_gpu"),
        pytest.param("paillier_lookup_gpu", PaillierLookupGPUClient, id="paillier_lookup_gpu"),
    ],
)
def test_gpu_execution_context_serialization_roundtrip(
    client_type: str,
    client_cls: type[PaillierGPUClient] | type[PaillierLookupGPUClient],
) -> None:
    if not client_cls.is_available():
        pytest.skip(f"{client_type} backend is unavailable on this machine")

    ctx = ExecutionContext.create(
        passphrase=_PASSPHRASE,
        homomorphic_client_type=client_type,
        embedding_length=_EMBED_LEN,
        key_len=_KEY_LEN,
    )
    restored = ExecutionContext._from_serialized_exec_context(
        json.loads(ctx.serialize_exec_context()),
        passphrase=_PASSPHRASE,
    )

    assert restored.hash() == ctx.hash()
    assert restored.device == "gpu"
    assert type(restored.homomorphic).__name__ == type(ctx.homomorphic).__name__

    encoded = restored.homomorphic.encode_hamming_server(
        restored.homomorphic.encrypt_vec_one(_VECTORS[0]),
        restored.homomorphic.encrypt_vec_one(_VECTORS[2]),
    )
    assert restored.homomorphic.decode_hamming_client_one(encoded) == _hamming(_VECTORS[0], _VECTORS[2])


def test_paillier_lookup_gpu_accepts_precomputed_tables() -> None:
    if not PaillierLookupGPUClient.is_available():
        pytest.skip("paillier_lookup_gpu backend is unavailable on this machine")

    ctx = ExecutionContext.create(
        passphrase=_PASSPHRASE,
        homomorphic_client_type="paillier_lookup_gpu",
        embedding_length=_EMBED_LEN,
        key_len=_KEY_LEN,
    )
    client = ctx.homomorphic
    config_dict = json.loads(client.stringify_config())
    pk = client.stringify_pk()
    sk = client.stringify_sk()

    cpu_clone = PaillierLookupCPU(
        embed_len=_EMBED_LEN,
        key_len=_KEY_LEN,
        alpha_len=config_dict["alpha_len"],
        skip_key_gen=True,
    )
    cpu_clone.load_stringified_keys(pk, sk)
    cpu_clone.load_config(config_dict)
    precomputed_tables = cpu_clone.dump_tables()

    gpu_clone = PaillierLookupGPUClient(
        embed_len=_EMBED_LEN,
        key_len=_KEY_LEN,
        alpha_len=config_dict["alpha_len"],
        skip_key_gen=True,
    )
    gpu_clone.load_stringified_keys(pk, sk)
    gpu_clone.load_config(config_dict, precomputed_tables=precomputed_tables)

    encoded = gpu_clone.encode_hamming_server(
        gpu_clone.encrypt_vec_one(_VECTORS[0]),
        gpu_clone.encrypt_vec_one(_VECTORS[1]),
    )
    assert gpu_clone.decode_hamming_client_one(encoded) == _hamming(_VECTORS[0], _VECTORS[1])
