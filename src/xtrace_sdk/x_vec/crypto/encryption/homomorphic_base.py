from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar

# Inputs (parameters) -> contravariant
T_CT  = TypeVar("T_CT")   # ciphertext for decrypt
T_PT = TypeVar("T_PT")  # lhs ciphertext for add/mul/xor
T_KEY = TypeVar("T_KEY")  # key for decrypt
T_KP  = TypeVar("T_KP")       # keypair type returned by key_gen


# # Parameter spec for key_gen (lets each subclass have its own args/kwargs)
# P = ParamSpec("P")

class HomomorphicBase(Generic[T_CT, T_PT, T_KP, T_KEY], ABC):
    """The Interface which all homomorphic encryption primitives should implement.

    :param ABC: Abstract class
    :type ABC: abc.ABC
    """

    def __init__(self) -> None:
        pass

    @staticmethod
    @abstractmethod
    def key_gen(*args: Any, **kwargs: Any) -> T_KP:
        """Key generation routine."""
        pass

    @staticmethod
    @abstractmethod
    def encrypt(pt: T_PT, key: T_KEY) -> T_PT:
        """Encrypts a plain text with key"""
        pass

    @staticmethod
    @abstractmethod
    def decrypt(ct: T_CT, key: T_KP) -> T_CT:
        """decrypts a cipher"""
        pass

    @staticmethod
    @abstractmethod
    def add(ct1: T_CT, ct2: T_CT, key: T_KEY) -> T_CT:
        """homomorphically add two ciphers and return ct3 such that decrypt(ct3)= decrypt(ct1) + decrypt(ct2)"""
        pass

    @staticmethod
    @abstractmethod
    def multiply(ct1: T_CT, ct2: T_CT, key: T_KEY) -> T_CT:
        """homomorphically multiply two ciphers and return ct3 such that decrypt(ct3)= decrypt(ct1) * decrypt(ct2)"""
        pass

    @staticmethod
    @abstractmethod
    def xor(ct1: T_CT, ct2: T_CT, key: T_KEY) -> T_CT:
        """homomorphically xor two ciphers and return ct3 such that decrypt(ct3)= decrypt(ct1) ^ decrypt(ct2)"""
        pass
