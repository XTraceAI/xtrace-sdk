"""Key provider abstractions for AES key management.

Defines a ``KeyProvider`` protocol and concrete implementations:

- ``PassphraseKeyProvider``: derives a 256-bit AES key from a passphrase using scrypt.
- ``AWSKMSKeyProvider``: envelope encryption via AWS KMS — the data encryption key (DEK)
  is generated or unwrapped by KMS and never persisted in plaintext.
"""

from __future__ import annotations

import base64
import os
from typing import Any, Protocol, cast

from Crypto.Protocol.KDF import scrypt as _scrypt_raw

_SCRYPT_N = 2**14
_SCRYPT_R = 8
_SCRYPT_P = 1
_KEY_LEN = 32
_DEFAULT_SALT = b"xtrace-aes-gcm-v1"


class KeyProvider(Protocol):
    """Protocol for objects that supply a 256-bit AES key and can wrap/unwrap it for storage."""

    def get_key(self) -> bytes:
        """Return the raw 256-bit AES key."""
        ...

    def wrap_key(self) -> bytes:
        """Return an opaque blob that can be stored alongside ciphertext to recover the key later."""
        ...

    @classmethod
    def from_wrapped(cls, wrapped: bytes, **kwargs: Any) -> KeyProvider:
        """Reconstruct a provider from a blob previously returned by :meth:`wrap_key`."""
        ...

    def provider_id(self) -> str:
        """Return a short identifier for the provider type (used in serialization)."""
        ...


class PassphraseKeyProvider:
    """Derives a 256-bit AES key from a passphrase using scrypt.

    This is the default key provider and preserves backwards-compatible behavior.
    The passphrase is never stored — only the derived key is kept in memory.
    """

    def __init__(self, passphrase: str, salt: bytes | None = None) -> None:
        self._salt = salt or _DEFAULT_SALT
        _pw: Any = passphrase.encode("utf-8")
        _sl: Any = self._salt
        self._key = cast(bytes, _scrypt_raw(
            _pw, _sl, key_len=_KEY_LEN, N=_SCRYPT_N, r=_SCRYPT_R, p=_SCRYPT_P,
        ))

    def get_key(self) -> bytes:
        return self._key

    def wrap_key(self) -> bytes:
        # For passphrase provider, the "wrapped key" is just the salt.
        # The actual key is re-derived from the passphrase on load.
        return self._salt

    @classmethod
    def from_wrapped(cls, wrapped: bytes, **kwargs: Any) -> PassphraseKeyProvider:
        """Re-derive the key from the passphrase and stored salt.

        :param wrapped: The salt bytes returned by :meth:`wrap_key`.
        :param passphrase: The original passphrase (passed via kwargs).
        """
        passphrase = kwargs["passphrase"]
        return cls(passphrase, salt=wrapped)

    def provider_id(self) -> str:
        return "passphrase"


class AWSKMSKeyProvider:
    """Envelope encryption via AWS KMS.

    On creation, generates a random 256-bit DEK and wraps it with KMS.
    On load, unwraps a previously wrapped DEK using KMS.

    Requires ``boto3`` at runtime. The KMS key must grant the caller
    ``kms:Encrypt`` and ``kms:Decrypt`` permissions.

    :param kms_client: A ``boto3`` KMS client (``boto3.client("kms")``).
    :param key_id: KMS key ID, ARN, or alias (e.g. ``"alias/xtrace"``).
    """

    def __init__(self, kms_client: object, key_id: str) -> None:
        self._kms = kms_client
        self._key_id = key_id
        self._key: bytes | None = None
        self._edek: bytes | None = None

    @classmethod
    def create(cls, kms_client: object, key_id: str) -> AWSKMSKeyProvider:
        """Generate a fresh DEK via KMS ``GenerateDataKey``.

        :param kms_client: A ``boto3`` KMS client.
        :param key_id: KMS key ID, ARN, or alias.
        """
        provider = cls(kms_client, key_id)
        resp = kms_client.generate_data_key(KeyId=key_id, KeySpec="AES_256")  # type: ignore[attr-defined]
        provider._key = resp["Plaintext"]
        provider._edek = resp["CiphertextBlob"]
        return provider

    @classmethod
    def from_wrapped(cls, wrapped: bytes, **kwargs: Any) -> AWSKMSKeyProvider:
        """Unwrap a previously stored EDEK using KMS ``Decrypt``.

        :param wrapped: The EDEK bytes returned by :meth:`wrap_key`.
        :param kms_client: A ``boto3`` KMS client (passed via kwargs).
        :param key_id: KMS key ID, ARN, or alias (passed via kwargs).
        """
        kms_client = kwargs["kms_client"]
        key_id = kwargs["key_id"]
        provider = cls(kms_client, key_id)
        resp = kms_client.decrypt(CiphertextBlob=wrapped, KeyId=key_id)
        provider._key = resp["Plaintext"]
        provider._edek = wrapped
        return provider

    def get_key(self) -> bytes:
        if self._key is None:
            raise RuntimeError("Key not initialized. Use AWSKMSKeyProvider.create() or AWSKMSKeyProvider.from_wrapped().")
        return self._key

    def wrap_key(self) -> bytes:
        if self._edek is None:
            raise RuntimeError("No wrapped key available. Use AWSKMSKeyProvider.create() first.")
        return self._edek

    def provider_id(self) -> str:
        return "aws_kms"
