import os

import pytest
from xtrace_sdk.x_vec.crypto.encryption.aes import AESClient


@pytest.fixture
def aes_client() -> AESClient:
    return AESClient(os.urandom(32))


def test_encrypt_decrypt_roundtrip(aes_client: AESClient) -> None:
    plaintext = "Hello, world!"
    ciphertext = aes_client.encrypt(plaintext)
    assert aes_client.decrypt(ciphertext) == plaintext


def test_encrypt_decrypt_bytes(aes_client: AESClient) -> None:
    raw = b"binary data \x00\x01\x02"
    ciphertext = aes_client.encrypt(raw)
    assert aes_client.decrypt(ciphertext) == raw.decode("utf-8")


def test_encrypt_produces_different_ciphertexts(aes_client: AESClient) -> None:
    plaintext = "same input"
    ct1 = aes_client.encrypt(plaintext)
    ct2 = aes_client.encrypt(plaintext)
    assert ct1 != ct2  # random nonce means different ciphertext each time


def test_wrong_key_fails() -> None:
    client_a = AESClient(os.urandom(32))
    client_b = AESClient(os.urandom(32))
    ciphertext = client_a.encrypt("secret")
    with pytest.raises(ValueError):
        client_b.decrypt(ciphertext)


def test_invalid_key_length_raises() -> None:
    with pytest.raises(ValueError, match="32-byte key"):
        AESClient(b"too-short")


def test_empty_string(aes_client: AESClient) -> None:
    ciphertext = aes_client.encrypt("")
    assert aes_client.decrypt(ciphertext) == ""


def test_unicode(aes_client: AESClient) -> None:
    plaintext = "日本語テスト 🔐"
    ciphertext = aes_client.encrypt(plaintext)
    assert aes_client.decrypt(ciphertext) == plaintext


def test_long_plaintext(aes_client: AESClient) -> None:
    plaintext = "A" * 100_000
    ciphertext = aes_client.encrypt(plaintext)
    assert aes_client.decrypt(ciphertext) == plaintext
