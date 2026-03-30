import base64
import hashlib

from Crypto.Cipher import AES
from Crypto.Util.Padding import unpad


def _legacy_cbc_decrypt(passphrase: str, enc: bytes) -> str:
    """Decrypt ciphertext produced by the pre-v0.2 AES-CBC implementation.

    Old wire format: base64( IV[16] || CBC_ciphertext )
    Old key derivation: SHA-256(passphrase)

    Use this in migration scripts to re-encrypt data with the current AES-GCM format.
    """
    key = hashlib.sha256(passphrase.encode("utf-8")).digest()
    raw = base64.b64decode(enc)
    iv = raw[:AES.block_size]
    cipher = AES.new(key, AES.MODE_CBC, iv)
    return unpad(cipher.decrypt(raw[AES.block_size:]), AES.block_size, "pkcs7").decode("utf-8")


class AESClient:
    """AES-GCM encryption client.

    Accepts a raw 256-bit key (typically supplied by a :class:`KeyProvider`).
    Encryption uses AES-256-GCM which provides both confidentiality and authenticity.
    A random 16-byte nonce per operation ensures semantic security.
    """

    def __init__(self, key: bytes) -> None:
        if len(key) != 32:
            raise ValueError(f"AES-256 requires a 32-byte key, got {len(key)}")
        self._key = key

    def encrypt(self, raw: str | bytes) -> bytes:
        """AES-GCM encrypt a plaintext.

        :param raw: Plaintext string or bytes to encrypt.
        :type raw: str | bytes
        :return: Base64-encoded bytes containing nonce + tag + ciphertext.
        :rtype: bytes
        """
        if isinstance(raw, str):
            raw_bytes = raw.encode("utf-8")
        else:
            raw_bytes = raw

        cipher = AES.new(self._key, AES.MODE_GCM)
        ciphertext, tag = cipher.encrypt_and_digest(raw_bytes)
        # Format: nonce (16) + tag (16) + ciphertext
        return base64.b64encode(bytes(cipher.nonce) + tag + ciphertext)

    def decrypt(self, enc: bytes) -> str:
        """AES-GCM decrypt a ciphertext.

        :param enc: Base64-encoded ciphertext produced by :meth:`encrypt`.
        :type enc: bytes
        :return: Decrypted plaintext string.
        :rtype: str
        :raises ValueError: If the ciphertext is corrupted or the key is wrong.
        """
        data = base64.b64decode(enc)
        nonce = data[:16]
        tag = data[16:32]
        ciphertext = data[32:]

        cipher = AES.new(self._key, AES.MODE_GCM, nonce=nonce)
        plaintext = cipher.decrypt_and_verify(ciphertext, tag)
        return plaintext.decode("utf-8")
