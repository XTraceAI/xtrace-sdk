import math
import secrets

from Crypto.Util import number

from xtrace_sdk.x_vec.crypto.encryption.homomorphic_base import HomomorphicBase
from xtrace_sdk.x_vec.utils.xtrace_types import GoldwasserMicaliKeyPair, GoldwasserMicaliPublicKey

BIT_STRING_LENGTH = 1024  # Adjustable constant for bit sequence length

class GoldwasserMicali(HomomorphicBase[list[int], list[int], GoldwasserMicaliKeyPair, GoldwasserMicaliPublicKey]):
    """
    A self implemented Goldwasser Micali encryption scheme
    """

    def __init__(self, keys: GoldwasserMicaliKeyPair):
        """constructor

        :param keys: GoldwasserMicali Pk SK pair returned by key_gen
        :type keys: GoldwasserMicaliKeyPair
        """
        self.keys = keys 

    @staticmethod
    def key_gen(key_len: int) -> GoldwasserMicaliKeyPair:
        """Key generation routine for the GoldwasserMicali crypto scheme. 

        :param key_len: the number of bits in PK SK
        :type key_len: int
        :return: GoldwasserMicali PK, SK pair
        :rtype: GoldwasserMicaliKeyPair
        """

        """Generate a prime number congruent to 3 mod 4. These are Blum primes"""
        p = 0
        while p % 4 != 3:
            p = number.getPrime(key_len)

        q = 0
        while q % 4 != 3:
            q = number.getPrime(key_len)

        n = p * q
        x = n - 1

        return {
            "sk" : {"p": p, "q":q},
            "pk": {"x": x, "n": n}
        }

    @staticmethod
    def generate_random_r(pk:GoldwasserMicaliPublicKey) -> int:
        """Helper function for encryption required by GoldwasserMicali scheme

        :param pk: the public key
        :type pk: GoldwasserMicaliPublicKey
        :return: a random r to be used for encryption.
        :rtype: int
        """
        while True:
            r = secrets.randbelow(pk['n']) + 1
            if math.gcd(r, pk['n']) == 1:
                break
        return r

    @staticmethod
    def encrypt(plaintext: list, pk: GoldwasserMicaliPublicKey) -> list:
        """Encryption with PK

        :param plaintext: the plaintext to be encrypted. 
        :type plaintext: int
        :param pk: the public key to be used for encryption
        :type pk: GoldwasserMicaliPublicKey
        :return: Goldwasser Micali cipher 
        :rtype: int
        """
        x = pk["x"]
        n = pk["n"]
        ciphertext = []
        for bit in plaintext:
            while True:
                y = secrets.randbelow(n - 1) + 1
                if number.GCD(y, n) == 1:
                    break
            c = (pow(y, 2, n) * pow(x, bit, n)) % n
            ciphertext.append(c)
        return ciphertext

    @staticmethod
    def decrypt(ciphertext: list, keys: GoldwasserMicaliKeyPair) -> list:
        """Decrypts a cihpher with Goldwasser Micali key pair and returns the plaintext

        :param ciphertext: cipher to be decrypted
        :type ciphertext: int
        :param keys: pk,sk pair where sk is used to encrypt ct.
        :type keys: GoldwasserMicaliKeyPair
        :return: plaintext
        :rtype: int
        """

        p = keys['sk']['p']
        q = keys['sk']['q']
        bits = []
        
        for c in ciphertext:
            # Check quadratic residuosity modulo p and q
            res_p = pow(c % p, (p-1)//2, p)
            res_q = pow(c % q, (q-1)//2, q)
            bits.append(0 if (res_p == 1 and res_q == 1) else 1)
        return bits


    @staticmethod
    def add(ciphertext1: list, ciphertext2: list, pk: GoldwasserMicaliPublicKey) -> list:
        """Not supported for GoldwasserMicali.
        """
        raise NotImplementedError("Goldwasser Micali is not homomorphic under addition")


    @staticmethod
    def multiply(ciphertext1: list, ciphertext2: list, pk: GoldwasserMicaliPublicKey) -> list:
        """Not supported for GoldwasserMicali.
        """
        raise NotImplementedError("Goldwasser Micali is not homomorphic under multiplication")

    @staticmethod
    def xor(ciphertext1: list, ciphertext2: list, pk: GoldwasserMicaliPublicKey) -> list:
        """Homomorphic xor of GoldwasserMicali crypto system.

        :param ciphertext1: one the ciphers to be xor'ed
        :type ciphertext1: int
        :param ciphertext2:  one the ciphers to be xor'ed
        :type ciphertext2: int
        :param pk: Pailier public key
        :type pk: GoldwasserMicaliPublicKey
        :return: ciphertext3 such that decrypt(ct3,sk) = decrypt(ct1,sk) [xor] decrypt(ct2,sk)
        :rtype: int
        """
        
        n = pk['n']
        if len(ciphertext1) != len(ciphertext2):
            raise ValueError("Ciphertexts must be of the same length")
            
        return [(c1 * c2) % n for c1, c2 in zip(ciphertext1, ciphertext2, strict=False)]
        

    @staticmethod
    def L(x: int, n:int) -> int:
        """Helper method for decryption
        """
        return int((x - 1) // n)
