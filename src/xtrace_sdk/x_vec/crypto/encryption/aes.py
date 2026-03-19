import base64
import hashlib

from Crypto import Random
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad


class AESClient:
    """This class wraps Crypto AES encryption lib."""

    def __init__(self, key: str):
        self.bs = AES.block_size
        self.key = hashlib.sha256(key.encode()).digest()

    def encrypt(self, raw: str | bytes) -> bytes:
        """AES encrypt a plaintext in bytes

        :param raw: plaintext to be encrypted with self.key
        :type raw: bytes
        :return: cipher in bytes
        :rtype: bytes
        """

        if isinstance(raw, str):
            raw_bytes = pad(raw.encode("utf8"), self.bs, "pkcs7")
        else:
            raw_bytes = raw
        
        iv = Random.new().read(AES.block_size)
        cipher = AES.new(self.key, AES.MODE_CBC, iv)
        return base64.b64encode(iv + cipher.encrypt(raw_bytes))

    def decrypt(self, enc: bytes) -> str:
        """AES decrypt a cipher

        :param enc: cipher in bytes
        :type enc: bytes
        :return: plaintext
        :rtype: string
        """
        enc = base64.b64decode(enc)
        iv = enc[: AES.block_size]
        cipher = AES.new(self.key, AES.MODE_CBC, iv)
        return unpad(cipher.decrypt(enc[AES.block_size :]), self.bs, "pkcs7").decode("utf8")
