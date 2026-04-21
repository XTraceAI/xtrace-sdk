import json
import pickle

import pytest

from xtrace_sdk.x_vec.crypto.paillier_client import PaillierClient
from xtrace_sdk.x_vec.crypto.paillier_lookup_client import PaillierLookupClient
from xtrace_sdk.x_vec.utils.execution_context import ExecutionContext


_PASSPHRASE = "test-gpu-exec-ctx-passphrase"
_EMBED_LEN = 8
_KEY_LEN = 1024
_VECTORS = [
    [0, 1, 0, 1, 1, 0, 1, 0],
    [1, 1, 0, 0, 1, 0, 0, 1],
    [1, 0, 1, 0, 1, 1, 0, 0],
]


def _hamming(lhs: list[int], rhs: list[int]) -> int:
    return sum(int(a != b) for a, b in zip(lhs, rhs, strict=True))


@pytest.mark.parametrize(
    ("client_cls", "client_type"),
    [
        pytest.param(PaillierClient, "paillier", id="paillier"),
        pytest.param(PaillierLookupClient, "paillier_lookup", id="paillier_lookup"),
    ],
)
def test_gpu_clients_expose_execution_context_protocol(
    monkeypatch: pytest.MonkeyPatch,
    client_cls: type[PaillierClient] | type[PaillierLookupClient],
    client_type: str,
) -> None:
    monkeypatch.setenv("DEVICE", "gpu")
    if not client_cls.has_gpu():
        pytest.skip(f"{client_type} GPU backend is unavailable on this machine")

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

    clone = client_cls(
        embed_len=_EMBED_LEN,
        key_len=_KEY_LEN,
        skip_key_gen=True,
    )
    clone.load_stringified_keys(pk, sk)
    if isinstance(clone, PaillierLookupClient):
        clone.load_config(config_dict, precomputed_tables=ctx.dump_tables())
    else:
        clone.load_config(config_dict)

    clone_cipher = clone.encrypt_vec_one(_VECTORS[0])
    clone_encoded = clone.encode_hamming_server(clone_cipher, clone.encrypt_vec_one(_VECTORS[1]))

    assert json.loads(clone.stringify_pk()) == json.loads(pk)
    assert json.loads(clone.stringify_sk()) == json.loads(sk)
    assert json.loads(clone.stringify_config()) == config_dict
    assert clone.decode_hamming_client_one(clone_encoded) == _hamming(_VECTORS[0], _VECTORS[1])


def test_lookup_exec_context_hash_matches_across_cpu_and_gpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEVICE", "cpu")
    cpu_ctx = ExecutionContext.create(
        passphrase=_PASSPHRASE,
        homomorphic_client_type="paillier_lookup",
        embedding_length=_EMBED_LEN,
        key_len=_KEY_LEN,
    )

    monkeypatch.setenv("DEVICE", "gpu")
    if not PaillierLookupClient.has_gpu():
        pytest.skip("paillier_lookup GPU backend is unavailable on this machine")

    restored_gpu = ExecutionContext._from_serialized_exec_context(
        json.loads(cpu_ctx.serialize_exec_context()),
        passphrase=_PASSPHRASE,
    )

    assert cpu_ctx.hash() == restored_gpu.hash()
    assert restored_gpu.device == "gpu"
    assert type(restored_gpu.homomorphic).__name__ == "PaillierLookupClient"


def test_paillier_lookup_gpu_pickle_roundtrip(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEVICE", "gpu")
    if not PaillierLookupClient.has_gpu():
        pytest.skip("paillier_lookup GPU backend is unavailable on this machine")

    client = PaillierLookupClient(embed_len=_EMBED_LEN, key_len=_KEY_LEN)
    restored = pickle.loads(pickle.dumps(client))

    encoded = restored.encode_hamming_server(
        restored.encrypt_vec_one(_VECTORS[0]),
        restored.encrypt_vec_one(_VECTORS[1]),
    )
    assert restored.decode_hamming_client_one(encoded) == _hamming(_VECTORS[0], _VECTORS[1])
