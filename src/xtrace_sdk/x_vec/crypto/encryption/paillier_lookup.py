import random
from time import perf_counter
from typing import Optional

import gmpy2
from Crypto.Util import number
from gmpy2 import mpz

from xtrace_sdk.x_vec.crypto.encryption.homomorphic_base import HomomorphicBase
from xtrace_sdk.x_vec.utils.xtrace_types import PaillierLookupKeyPair, PaillierLookupPublicKey

# Constants for Paillier Lookup Tables
MSG_BITS = 8  # Message chunk size
NOISE_TABLE_SIZE = 2**8  # Precomputed noise entries
NOISE_MULTIPLES = 14  # Number of noise factors to multiply

# Imports necessary for obtaining DSA parameters (generating alpha)
# from cryptography.hazmat.primitives.asymmetric import dsa
# import secrets

class Paillier_Lookup(HomomorphicBase[int, int, PaillierLookupKeyPair, PaillierLookupPublicKey]):
    """
    A self implemented paillier encryption scheme
    """


    def __init__(self, keys: PaillierLookupKeyPair):
        """constructor

        :param keys: Paillier Pk SK pair returned by key_gen
        :type keys: PaillierLookupKeyPair
        """
        self.keys = keys

    @staticmethod
    def key_gen(key_len: int, alpha_len:int) -> PaillierLookupKeyPair:
        """Key generation routine for the Paillier crypto scheme. 

        :param key_len: the number of bits in PK SK
        :type key_len: int
        :return: Paillier PK, SK pair
        :rtype: PaillierKeyPair
        """

        # --- pick q-bit split close to requested alpha_len ---
        q_bits_bound = min(key_len, alpha_len)
        q_bits_1 = q_bits_bound // 2
        q_bits_2 = q_bits_bound - q_bits_1
        
        # --- two independent DSA-style groups ---
        p1, q1, g1 = Paillier_Lookup.gen_dsa_params_custom(key_len, q_bits_1)
        p2, q2, g2 = Paillier_Lookup.gen_dsa_params_custom(key_len, q_bits_2)
        
        # Paillier primes
        p = p1
        q = p2
        n = p * q
        n_sq = n * n
        
        # Lift g1, g2 to p^2, q^2 so their orders gain factors p and q
        g1_p2 = Paillier_Lookup.lift_to_p2_with_p_component(g1 % p, p)      # ord = q1 * p in Z_{p^2}*
        g2_q2 = Paillier_Lookup.lift_to_p2_with_p_component(g2 % q, q)      # ord = q2 * q in Z_{q^2}*
        
        # CRT to n^2
        g, _ = Paillier_Lookup.crt_pair(g1_p2, p * p, g2_q2, q * q)
        
        # alpha = lcm(q1, q2) (q1 * q2 should work as well)
        a = gmpy2.lcm(q1, q2)
        
        # (sanity) ensure ord(g) | n*a and n-part is present
        if gmpy2.powmod(g, n * a, n_sq) != 1 or gmpy2.powmod(g, n, n_sq) == 1:
            # Regenerate if keygen fails, extremely improbable
            return Paillier_Lookup.key_gen(key_len, alpha_len)
        
        phi = (p - 1) * (q - 1)
        g_n = gmpy2.powmod(g, n, n_sq)
        g_a_inv = gmpy2.powmod(Paillier_Lookup.L(gmpy2.powmod(g, a, n_sq), n), -1, n)
        
        # precompute g table
        g_table = Paillier_Lookup.precompute_g_table(g, n, key_len)
        
        # We are not using a precomputed noise table anymore, pass an empty list for API compatibility.
        noise_table = Paillier_Lookup.precompute_noise_table(g, n)
        
        return {
            "sk": {"phi": phi, "a": a, "g_a_inv": g_a_inv},
            "pk": {"g": g, "n": n, "n_squared": n_sq, "g_n": g_n},
            "g_table": g_table,
            "noise_table": noise_table,  # kept for signature compatibility; not used yet
            "key_len": key_len,
            "message_chunks": (key_len * 2 + MSG_BITS - 1) // MSG_BITS
        }


    @staticmethod
    def precompute_g_table(g: mpz, n: mpz, key_len: int, msg_bits: int=MSG_BITS) -> dict[int, list[int]]:
        start_time = perf_counter()
        g_table = {}
        max_j = 2**msg_bits
        for i in range(0, (key_len * 2 + MSG_BITS - 1) // MSG_BITS):  # Split into key_length // MSG_BIT chunks (e.g., 8-bit each)
            base = gmpy2.powmod(g, 2**(msg_bits * i), n**2)
            g_table[i] = [gmpy2.powmod(base, j, n**2) for j in range(max_j)]
        end_time = perf_counter()
        # print(f"Precomputation of g_table took {end_time - start_time:.6f} seconds")
        # print(f"size of g_table: {sys.getsizeof(g_table)} bytes")
        return g_table

    @staticmethod
    def precompute_noise_table(g: mpz, n: mpz, size: int=NOISE_TABLE_SIZE) -> list[int]:
        start_time = perf_counter()
        n_sq = n**2
        g_n = gmpy2.powmod(g, n, n_sq)
        noise_table = [gmpy2.powmod(g_n, Paillier_Lookup.generate_random_r(n), n_sq) for _ in range(size)]
        # noise_table = [1 + Paillier_Lookup.generate_random_r({"n": n}) * g % n_sq for _ in range(size)]
        end_time = perf_counter()
        # print(f"Precomputation of noise_table took {end_time - start_time:.6f} seconds")
        # print(f"size of noise_table: {sys.getsizeof(noise_table)} bytes")
        return noise_table

    @staticmethod
    def dump_tables(keys: PaillierLookupKeyPair) -> dict:
        """Dump precomputed tables to simple python types (dict/list of ints) for serialization"""
        g_table = keys.get("g_table", {})
        noise_table = keys.get("noise_table", [])
        
        # Convert g_table: dict[int, list[mpz]] -> dict[int, list[int/hex_str]]
        # keys are int, values are lists of mpz. Convert mpz to int for json/pickle compatibility
        dumped_g = {k: [int(x) for x in v] for k, v in g_table.items()}
        
        # Convert noise_table: list[mpz] -> list[int]
        dumped_noise = [int(x) for x in noise_table]
        
        return {
            "g_table": dumped_g,
            "noise_table": dumped_noise
        }

    @staticmethod
    def load_tables(data: dict) -> tuple[dict[int, list[mpz]], list[mpz]]:
        """Load tables from simple python types"""
        # Convert back to mpz
        g_table_raw = data.get("g_table", {})
        noise_table_raw = data.get("noise_table", [])
        
        g_table = {int(k): [mpz(x) for x in v] for k, v in g_table_raw.items()}
        noise_table = [mpz(x) for x in noise_table_raw]
        
        return g_table, noise_table

    @staticmethod
    def gen_dsa_params_custom(p_bits: int, q_bits: int) -> tuple[mpz, mpz, mpz]:
        """1024-bit DSA => q is 160-bit (FIPS 186-4).

        :param p_bits: The number of bits for the keys
        :type p_bits: int
        :param q_bits: The number of bits in alpha
        :type q_bits: int
        :return: mpz(P), mpz(Q), mpz(G).
        :rtype: int
        """
        if q_bits >= p_bits:
            raise ValueError("alpha_len too large: q_bits must be < p_bits")

        # q: exact q_bits prime
        q = mpz(number.getPrime(q_bits))

        # choose k so that p = k*q + 1 has exactly p_bits bits
        two_pow_p_1 = mpz(1) << (p_bits - 1)
        two_pow_p   = mpz(1) << p_bits
        k_lo = two_pow_p_1 // q
        k_hi = (two_pow_p - 1) // q  # inclusive upper bound

        # ensure even k so that p is odd (since q is odd)
        if k_lo % 2 == 1:
            k_lo += 1

        while True:
            # pick even k uniformly in [k_lo, k_hi]
            span = (k_hi - k_lo) // 2 + 1
            t = mpz(number.getRandomRange(0, span))
            k = k_lo + 2 * t
            p = k * q + 1
            if Paillier_Lookup.bitlen(p) != p_bits:
                continue
            if gmpy2.is_prime(p, 40):  # Miller–Rabin rounds
                break

        # generator of order q
        e = (p - 1) // q
        while True:
            h = mpz(number.getRandomRange(2, p - 1))
            g = gmpy2.powmod(h, e, p)
            if g != 1:
                break

        return p, q, g
    
    @staticmethod
    def generate_random_r(n: mpz) -> int:
        """Helper function for encryption required by Paillier scheme

        :param n: the RSA modulus from the public key
        :type n: mpz
        :return: a random r to be used for encryption.
        :rtype: int
        """
        while True:
            r = gmpy2.mpz(random.randint(0, n))
            if gmpy2.gcd(r, n) == 1:
                break
        return r

    @staticmethod
    def encrypt(
        plaintext: int, pk: PaillierLookupPublicKey, g_table: dict[int, list[mpz]] | None= None, noise_table : list[mpz] | None= None, message_chunks :int=0
    ) -> int:
        """Encryption with PK

        :param plaintext: the plaintext to be encrypted. 
        :type plaintext: int
        :param pk: the public key to be used for encryption
        :type pk: PaillierPublicKey
        :return: Paillier cipher 
        :rtype: int
        """

        if g_table is None:
            g_table = {}
        if noise_table is None:
            noise_table = [mpz(1)]
        assert plaintext < pk["n"] and plaintext >= 0, "plaintext must be in range [0,n)"
        g = pk["g"]
        n = pk["n"]
        n_sq = pk["n_squared"]
        r = Paillier_Lookup.generate_random_r(pk['n'])
        assert gmpy2.gcd(r, n) == 1
        
        m_split = [(plaintext >> (MSG_BITS*i)) & (2**MSG_BITS -1) for i in range(message_chunks)]
        g_m = gmpy2.mpz(1)

        for i in range(message_chunks):
            g_m = (g_m * g_table[i][m_split[i]]) % n_sq     
        # assert g_m == pow(g, plaintext, n * n)
        
        # Compute noise: multiply random entries from noise_table
        
        noise = gmpy2.mpz(1)
        for _ in range(NOISE_MULTIPLES):
            noise = (noise * random.choice(noise_table)) % n_sq
        
        return (g_m * noise) % n_sq

        # return (pow(g, plaintext, n * n) * pow(r, n, n * n)) % (n * n)

    @staticmethod
    def decrypt(ciphertext: int, keys: PaillierLookupKeyPair) -> int:
        """Decrypts a cihpher with paillier key pair and returns the plaintext

        :param ciphertext: cipher to be decrypted
        :type ciphertext: int
        :param keys: pk,sk pair where sk is used to encrypt ct.
        :type keys: PaillierKeyPair
        :return: plaintext
        :rtype: int
        """
        assert ciphertext < keys['pk']['n_squared'] and ciphertext >=0 , "ciphertext must be in range [0,n^2)"
        phi = keys['sk']["phi"]
        a = keys['sk']["a"]
        g = keys['pk']['g']
        n = keys['pk']['n']
        n_sq = keys['pk']['n_squared']

        c_a = Paillier_Lookup.L(gmpy2.powmod(ciphertext, a, n_sq), n)
        g_a_inv = keys['sk']["g_a_inv"] # gmpy2.powmod(Paillier_Lookup.L(gmpy2.powmod(g, a, n_sq), n), -1, n)

        return (c_a * g_a_inv) % n


    @staticmethod
    def add(ciphertext1: int, ciphertext2: int, pk: PaillierLookupPublicKey) -> int:
        """Homomorphic addition of Paillier crypto system.

        :param ciphertext1: one the ciphers to be added
        :type ciphertext1: int
        :param ciphertext2:  one the ciphers to be added
        :type ciphertext2: int
        :param pk: Pailier public key
        :type pk: PaillierPublicKey
        :return: ciphertext3 such that decrypt(ct3,sk) = decrypt(ct1,sk) + decrypt(ct2,sk)
        :rtype: int
        """
        return (ciphertext1 * ciphertext2) % (pk['n_squared'])

    @staticmethod
    def multiply(ciphertext1: int, ciphertext2: int, pk: PaillierLookupPublicKey) -> int:
        """Not supported for Paillier.
        """
        raise NotImplementedError("Paillier is not homomorphic under multiplication")

    @staticmethod
    def xor(ciphertext1: int, ciphertext2: int, pk: PaillierLookupPublicKey) -> int:
        """Not supported for Paillier.
        """
        raise NotImplementedError("Paillier is not homomorphic under xor")

    @staticmethod
    def L(x: int, n:int) -> int:
        """Helper method for decryption
        """
        return gmpy2.mpz((x - 1) // n)

    @staticmethod
    def bitlen(x: mpz) -> int:
        """Helper method to get bit length of gmpy mpz integers
        """
        if x <= 0:
            return 0
        return int(gmpy2.num_digits(x, 2))

    @staticmethod
    def crt_pair(a1: mpz, m1: mpz, a2: mpz, m2: mpz) -> tuple[mpz, mpz]:
        """Helper method for Chinsese Remainder Theorem
        """
        M = m1 * m2
        inv = gmpy2.invert(m1 % m2, m2)
        t = ((a2 - a1) % m2) * inv % m2
        return (a1 + m1 * t) % M, M

    @staticmethod
    def lift_to_p2_with_p_component(g_mod_p: mpz, p: mpz) -> mpz:
        """Multiply by (1+p) to inject an order-p component in Z_{p^2}*
        """
        return (g_mod_p * (1 + p)) % (p * p)
