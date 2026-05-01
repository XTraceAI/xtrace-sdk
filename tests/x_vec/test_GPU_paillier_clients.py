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
    client_cls: type[PaillierClient] | type[PaillierLookupClient],
    client_type: str,
) -> None:
    if not client_cls.has_gpu():
        pytest.skip(f"{client_type} GPU backend is unavailable on this machine")

    ctx = ExecutionContext.create(
        passphrase=_PASSPHRASE,
        homomorphic_client_type=client_type,
        embedding_length=_EMBED_LEN,
        key_len=_KEY_LEN,
        device="gpu",
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
        device="gpu",
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


@pytest.mark.parametrize(
    ("client_cls", "client_type"),
    [
        pytest.param(PaillierClient, "paillier", id="paillier"),
        pytest.param(PaillierLookupClient, "paillier_lookup", id="paillier_lookup"),
    ],
)
@pytest.mark.parametrize(
    ("source_device", "target_device"),
    [
        pytest.param("cpu", "gpu", id="cpu_to_gpu"),
        pytest.param("gpu", "cpu", id="gpu_to_cpu"),
    ],
)
def test_cross_device_interop(
    client_cls: type[PaillierClient] | type[PaillierLookupClient],
    client_type: str,
    source_device: str,
    target_device: str,
) -> None:
    """A context created on one device and loaded on the other must:

    1. Hash equal (keys are portable; device must not affect identity).
    2. Round-trip: ciphertexts from either side decode correctly on the other.
    """
    if not client_cls.has_gpu():
        pytest.skip(f"{client_type} GPU backend is unavailable on this machine")

    source_ctx = ExecutionContext.create(
        passphrase=_PASSPHRASE,
        homomorphic_client_type=client_type,
        embedding_length=_EMBED_LEN,
        key_len=_KEY_LEN,
        device=source_device,
    )
    target_ctx = ExecutionContext._from_serialized_exec_context(
        json.loads(source_ctx.serialize_exec_context()),
        passphrase=_PASSPHRASE,
        device=target_device,
    )

    assert source_ctx.device == source_device
    assert target_ctx.device == target_device
    assert source_ctx.hash() == target_ctx.hash()

    src = source_ctx.homomorphic
    tgt = target_ctx.homomorphic
    expected = _hamming(_VECTORS[0], _VECTORS[1])

    # Encrypt + server-encode on source, decode on target.
    src_a = src.encrypt_vec_one(_VECTORS[0])
    src_b = src.encrypt_vec_one(_VECTORS[1])
    encoded_on_source = src.encode_hamming_server(src_a, src_b)
    assert tgt.decode_hamming_client_one(encoded_on_source) == expected

    # Encrypt on source, server-encode on target, decode on target.
    encoded_on_target = tgt.encode_hamming_server(src_a, src_b)
    assert tgt.decode_hamming_client_one(encoded_on_target) == expected

    # Reverse direction: encrypt on target, server-encode on source, decode on source.
    tgt_a = tgt.encrypt_vec_one(_VECTORS[0])
    tgt_b = tgt.encrypt_vec_one(_VECTORS[1])
    encoded_back = src.encode_hamming_server(tgt_a, tgt_b)
    assert src.decode_hamming_client_one(encoded_back) == expected


def test_paillier_lookup_gpu_pickle_roundtrip() -> None:
    if not PaillierLookupClient.has_gpu():
        pytest.skip("paillier_lookup GPU backend is unavailable on this machine")

    client = PaillierLookupClient(embed_len=_EMBED_LEN, key_len=_KEY_LEN, device="gpu")
    restored = pickle.loads(pickle.dumps(client))

    encoded = restored.encode_hamming_server(
        restored.encrypt_vec_one(_VECTORS[0]),
        restored.encrypt_vec_one(_VECTORS[1]),
    )
    assert restored.decode_hamming_client_one(encoded) == _hamming(_VECTORS[0], _VECTORS[1])
