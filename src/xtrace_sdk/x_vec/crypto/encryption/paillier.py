import secrets

import gmpy2
from Crypto.Util import number

from xtrace_sdk.x_vec.crypto.encryption.homomorphic_base import HomomorphicBase
from xtrace_sdk.x_vec.utils.xtrace_types import PaillierKeyPair, PaillierPublicKey


class Paillier(HomomorphicBase[gmpy2.mpz, gmpy2.mpz, PaillierKeyPair, PaillierPublicKey]):
    """
    A self implemented paillier encryption scheme
    """

    def __init__(self, keys: PaillierKeyPair):
        """constructor

        :param keys: Paillier Pk SK pair returned by key_gen
        :type keys: PaillierKeyPair
        """
        self.keys = keys 

    @staticmethod
    def key_gen(key_len: int) -> PaillierKeyPair:
        """Key generation routine for the Paillier crypto scheme. 

        :param key_len: the number of bits in PK SK
        :type key_len: int
        :return: Paillier PK, SK pair
        :rtype: PaillierKeyPair
        """

        p = gmpy2.mpz(number.getPrime(key_len))

        q = gmpy2.mpz(number.getPrime(key_len))

        n = p * q
        phi = (p - 1) * (q - 1)
        g = 1 + n

        return {
            "sk" : {"phi": phi, "inv": gmpy2.invert(phi, n)},
            "pk": {"g": g, "n": n, "n_squared": n * n}
        }

    @staticmethod
    def generate_random_r(pk:PaillierPublicKey) -> gmpy2.mpz:
        """Helper function for encryption required by Paillier scheme

        :param pk: the public key
        :type pk: PaillierPublicKey
        :return: a random r to be used for encryption.
        :rtype: gmpy2.mpz
        """
        n = int(pk['n'])
        while True:
            r = gmpy2.mpz(secrets.randbelow(n - 1) + 1)
            if gmpy2.gcd(r, pk['n']) == 1:
                break
        return r

    @staticmethod
    def encrypt(plaintext: gmpy2.mpz, pk: PaillierPublicKey) -> gmpy2.mpz:
        """Encryption with PK

        :param plaintext: the plaintext to be encrypted. 
        :type plaintext: gmpy2.mpz
        :param pk: the public key to be used for encryption
        :type pk: PaillierPublicKey
        :return: Paillier cipher 
        :rtype: gmpy2.mpz
        """
        if not (0 <= plaintext < pk["n"]):
            raise ValueError("plaintext must be in range [0, n)")
        g = pk["g"]
        n = pk["n"]
        n_squared = pk["n_squared"]
        r = Paillier.generate_random_r(pk)
        if gmpy2.gcd(r, n) != 1:
            raise ValueError("random r is not coprime with n")
        a = gmpy2.powmod(g, plaintext, n_squared)
        b = gmpy2.powmod(r, n, n_squared)
        return (a * b) % n_squared

    @staticmethod
    def decrypt(ciphertext: gmpy2.mpz, keys: PaillierKeyPair) -> gmpy2.mpz:
        """Decrypts a cihpher with paillier key pair and returns the plaintext

        :param ciphertext: cipher to be decrypted
        :type ciphertext: gmpy2.mpz
        :param keys: pk,sk pair where sk is used to encrypt ct.
        :type keys: PaillierKeyPair
        :return: plaintext
        :rtype: gmpy2.mpz
        """
        if not (0 <= ciphertext < keys['pk']['n_squared']):
            raise ValueError("ciphertext must be in range [0, n^2)")
        phi = keys['sk']["phi"]
        n = keys['pk']['n']
        n_squared = keys['pk']['n_squared']
        x = gmpy2.powmod(ciphertext, phi, n_squared) -1 
        inv = keys['sk']["inv"]

        return ((x // n) * inv) % n

    @staticmethod
    def add(ciphertext1: gmpy2.mpz, ciphertext2: gmpy2.mpz, pk: PaillierPublicKey) -> gmpy2.mpz:
        """Homomorphic addition of Paillier crypto system.

        :param ciphertext1: one the ciphers to be added
        :type ciphertext1: int
        :param ciphertext2:  one the ciphers to be added
        :type ciphertext2: int
        :param pk: Pailier public key
        :type pk: PaillierPublicKey
        :return: ciphertext3 such that decrypt(ct3,sk) = decrypt(ct1,sk) + decrypt(ct2,sk)
        :rtype: gmpy2.mpz
        """
        return (ciphertext1 * ciphertext2) % (pk['n_squared'])

    @staticmethod
    def multiply(ciphertext1: gmpy2.mpz, ciphertext2: gmpy2.mpz, pk: PaillierPublicKey) -> gmpy2.mpz:
        """Not supported for Paillier.
        """
        raise NotImplementedError("Paillier is not homomorphic under multiplication")

    @staticmethod
    def xor(ciphertext1: gmpy2.mpz, ciphertext2: gmpy2.mpz, pk: PaillierPublicKey) -> gmpy2.mpz:
        """Not supported for Paillier.
        """
        raise NotImplementedError("Paillier is not homomorphic under xor")