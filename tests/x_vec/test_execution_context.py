import json
import tempfile
from unittest.mock import MagicMock
import os

import pytest
from xtrace_sdk.x_vec.crypto.key_provider import (
    AWSKMSKeyProvider,
    PassphraseKeyProvider,
)
from xtrace_sdk.x_vec.utils.execution_context import ExecutionContext


_PASSPHRASE = "test-exec-ctx-passphrase"
_EMBED_LEN = 64
_KEY_LEN = 512


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def ctx() -> ExecutionContext:
    return ExecutionContext.create(
        passphrase=_PASSPHRASE,
        homomorphic_client_type="paillier",
        embedding_length=_EMBED_LEN,
        key_len=_KEY_LEN,
    )


def _mock_kms_client() -> MagicMock:
    """Build a mock boto3 KMS client that simulates GenerateDataKey and Decrypt."""
    kms = MagicMock()
    dek = os.urandom(32)
    edek = b"mock-encrypted-" + dek  # fake wrapped key
    kms.generate_data_key.return_value = {"Plaintext": dek, "CiphertextBlob": edek}
    kms.decrypt.return_value = {"Plaintext": dek}
    kms._dek = dek
    kms._edek = edek
    return kms


# ---------------------------------------------------------------------------
# Creation
# ---------------------------------------------------------------------------

def test_create_paillier(ctx: ExecutionContext) -> None:
    assert ctx.embed_len() == _EMBED_LEN
    assert ctx.key_len() == _KEY_LEN
    assert isinstance(ctx.id, str) and len(ctx.id) == 64  # SHA-256 hex


def test_create_paillier_lookup() -> None:
    ctx = ExecutionContext.create(
        passphrase=_PASSPHRASE,
        homomorphic_client_type="paillier_lookup",
        embedding_length=_EMBED_LEN,
        key_len=_KEY_LEN,
    )
    assert ctx.embed_len() == _EMBED_LEN


def test_create_unsupported_type_raises() -> None:
    with pytest.raises(ValueError, match="Unsupported"):
        ExecutionContext.create(
            passphrase=_PASSPHRASE,
            homomorphic_client_type="rsa",
            embedding_length=_EMBED_LEN,
            key_len=_KEY_LEN,
        )


def test_create_embed_len_ge_key_len_raises() -> None:
    with pytest.raises(ValueError, match="embedding_length"):
        ExecutionContext.create(
            passphrase=_PASSPHRASE,
            homomorphic_client_type="paillier",
            embedding_length=1024,
            key_len=512,
        )


def test_create_no_passphrase_no_provider_raises() -> None:
    with pytest.raises(ValueError, match="Either key_provider or passphrase"):
        ExecutionContext.create(
            homomorphic_client_type="paillier",
            embedding_length=_EMBED_LEN,
            key_len=_KEY_LEN,
        )


# ---------------------------------------------------------------------------
# Hash / equality
# ---------------------------------------------------------------------------

def test_hash_deterministic(ctx: ExecutionContext) -> None:
    assert ctx.hash() == ctx.hash()


def test_equality_same_keys(ctx: ExecutionContext) -> None:
    assert ctx == ctx


def test_inequality_different_keys() -> None:
    a = ExecutionContext.create(_PASSPHRASE, "paillier", _EMBED_LEN, _KEY_LEN)
    b = ExecutionContext.create(_PASSPHRASE, "paillier", _EMBED_LEN, _KEY_LEN)
    assert a != b


# ---------------------------------------------------------------------------
# Serialization (passphrase)
# ---------------------------------------------------------------------------

def test_to_dict_enc_has_encrypted_sk(ctx: ExecutionContext) -> None:
    d = ctx.to_dict_enc()
    assert "sk" in d and "pk" in d and "type" in d and "config" in d
    assert d["key_provider"] == "passphrase"
    assert "wrapped_key" in d
    plain_sk = ctx.homomorphic.stringify_sk()
    assert d["sk"] != plain_sk


def test_serialize_and_deserialize(ctx: ExecutionContext) -> None:
    serialized = ctx.serialize_exec_context()
    json_obj = json.loads(serialized)
    restored = ExecutionContext._from_serialized_exec_context(json_obj, passphrase=_PASSPHRASE)
    assert restored.hash() == ctx.hash()


def test_save_and_load_from_disk(ctx: ExecutionContext) -> None:
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name
    ctx.save_to_disk(path)
    restored = ExecutionContext.load_from_disk(_PASSPHRASE, path)
    assert restored.hash() == ctx.hash()


def test_save_to_disk_via_create() -> None:
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name
    ctx = ExecutionContext.create(
        passphrase=_PASSPHRASE,
        homomorphic_client_type="paillier",
        embedding_length=_EMBED_LEN,
        key_len=_KEY_LEN,
        path=path,
    )
    restored = ExecutionContext.load_from_disk(_PASSPHRASE, path)
    assert restored.hash() == ctx.hash()


def test_custom_salt(ctx: ExecutionContext) -> None:
    salt = b"my-custom-salt-value"
    ctx_salted = ExecutionContext.create(
        passphrase=_PASSPHRASE,
        homomorphic_client_type="paillier",
        embedding_length=_EMBED_LEN,
        key_len=_KEY_LEN,
        salt=salt,
    )
    assert ctx.to_dict_enc()["sk"] != ctx_salted.to_dict_enc()["sk"]


def test_custom_salt_roundtrip() -> None:
    salt = b"roundtrip-salt"
    ctx = ExecutionContext.create(
        passphrase=_PASSPHRASE,
        homomorphic_client_type="paillier",
        embedding_length=_EMBED_LEN,
        key_len=_KEY_LEN,
        salt=salt,
    )
    serialized = ctx.serialize_exec_context()
    json_obj = json.loads(serialized)
    # wrapped_key in the serialized dict now carries the salt, so passphrase alone suffices
    restored = ExecutionContext._from_serialized_exec_context(json_obj, passphrase=_PASSPHRASE)
    assert restored.hash() == ctx.hash()


def test_custom_salt_disk_roundtrip() -> None:
    salt = b"disk-salt"
    ctx = ExecutionContext.create(
        passphrase=_PASSPHRASE,
        homomorphic_client_type="paillier",
        embedding_length=_EMBED_LEN,
        key_len=_KEY_LEN,
        salt=salt,
    )
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name
    ctx.save_to_disk(path)
    restored = ExecutionContext.load_from_disk(_PASSPHRASE, path)
    assert restored.hash() == ctx.hash()



def test_wrong_passphrase_fails_deserialize(ctx: ExecutionContext) -> None:
    serialized = ctx.serialize_exec_context()
    json_obj = json.loads(serialized)
    with pytest.raises(ValueError):
        ExecutionContext._from_serialized_exec_context(json_obj, passphrase="wrong-passphrase")


def test_wrong_salt_fails_deserialize() -> None:
    salt = b"correct-salt"
    ctx = ExecutionContext.create(
        passphrase=_PASSPHRASE,
        homomorphic_client_type="paillier",
        embedding_length=_EMBED_LEN,
        key_len=_KEY_LEN,
        salt=salt,
    )
    serialized = ctx.serialize_exec_context()
    json_obj = json.loads(serialized)
    # Override with a wrong-salt provider to bypass the stored wrapped_key
    wrong_provider = PassphraseKeyProvider(_PASSPHRASE, salt=b"wrong-salt")
    with pytest.raises(ValueError):
        ExecutionContext._from_serialized_exec_context(json_obj, key_provider=wrong_provider)


def test_context_id_override() -> None:
    ctx = ExecutionContext.create(
        passphrase=_PASSPHRASE,
        homomorphic_client_type="paillier",
        embedding_length=_EMBED_LEN,
        key_len=_KEY_LEN,
    )
    serialized = ctx.serialize_exec_context()
    json_obj = json.loads(serialized)
    restored = ExecutionContext._from_serialized_exec_context(
        json_obj, passphrase=_PASSPHRASE, context_id="custom-id-123"
    )
    assert restored.id == "custom-id-123"


# ---------------------------------------------------------------------------
# KeyProvider: PassphraseKeyProvider
# ---------------------------------------------------------------------------

def test_passphrase_provider_explicit() -> None:
    provider = PassphraseKeyProvider(_PASSPHRASE)
    ctx = ExecutionContext.create(
        key_provider=provider,
        homomorphic_client_type="paillier",
        embedding_length=_EMBED_LEN,
        key_len=_KEY_LEN,
    )
    assert ctx.embed_len() == _EMBED_LEN
    assert ctx.key_provider.provider_id() == "passphrase"


def test_passphrase_provider_roundtrip() -> None:
    provider = PassphraseKeyProvider(_PASSPHRASE, salt=b"explicit-provider-salt")
    ctx = ExecutionContext.create(
        key_provider=provider,
        homomorphic_client_type="paillier",
        embedding_length=_EMBED_LEN,
        key_len=_KEY_LEN,
    )
    serialized = ctx.serialize_exec_context()
    json_obj = json.loads(serialized)
    restored = ExecutionContext._from_serialized_exec_context(json_obj, passphrase=_PASSPHRASE)
    assert restored.hash() == ctx.hash()


# ---------------------------------------------------------------------------
# KeyProvider: AWSKMSKeyProvider (mocked)
# ---------------------------------------------------------------------------

def test_aws_kms_provider_create() -> None:
    kms = _mock_kms_client()
    provider = AWSKMSKeyProvider.create(kms, "alias/test-key")
    assert provider.provider_id() == "aws_kms"
    assert len(provider.get_key()) == 32
    kms.generate_data_key.assert_called_once_with(KeyId="alias/test-key", KeySpec="AES_256")


def test_aws_kms_provider_wrap_unwrap() -> None:
    kms = _mock_kms_client()
    provider = AWSKMSKeyProvider.create(kms, "alias/test-key")
    wrapped = provider.wrap_key()
    restored = AWSKMSKeyProvider.from_wrapped(wrapped, kms_client=kms, key_id="alias/test-key")
    assert restored.get_key() == provider.get_key()


def test_aws_kms_exec_context_roundtrip() -> None:
    kms = _mock_kms_client()
    provider = AWSKMSKeyProvider.create(kms, "alias/test-key")
    ctx = ExecutionContext.create(
        key_provider=provider,
        homomorphic_client_type="paillier",
        embedding_length=_EMBED_LEN,
        key_len=_KEY_LEN,
    )
    serialized = ctx.serialize_exec_context()
    json_obj = json.loads(serialized)
    assert json_obj["key_provider"] == "aws_kms"

    # Reconstruct the provider from the stored EDEK
    import base64
    edek = base64.b64decode(json_obj["wrapped_key"])
    restored_provider = AWSKMSKeyProvider.from_wrapped(edek, kms_client=kms, key_id="alias/test-key")
    restored = ExecutionContext._from_serialized_exec_context(json_obj, key_provider=restored_provider)
    assert restored.hash() == ctx.hash()


def test_aws_kms_disk_roundtrip() -> None:
    kms = _mock_kms_client()
    provider = AWSKMSKeyProvider.create(kms, "alias/test-key")
    ctx = ExecutionContext.create(
        key_provider=provider,
        homomorphic_client_type="paillier",
        embedding_length=_EMBED_LEN,
        key_len=_KEY_LEN,
    )
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name
    ctx.save_to_disk(path)

    # On load, caller must reconstruct the KMS provider from the stored EDEK
    with open(path) as f:
        json_obj = json.loads(f.read())
    import base64
    edek = base64.b64decode(json_obj["wrapped_key"])
    restored_provider = AWSKMSKeyProvider.from_wrapped(edek, kms_client=kms, key_id="alias/test-key")
    restored = ExecutionContext.load_from_disk(path=path, key_provider=restored_provider)
    assert restored.hash() == ctx.hash()
