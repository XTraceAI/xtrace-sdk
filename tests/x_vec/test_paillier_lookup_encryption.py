import random

import pytest
from xtrace_sdk.x_vec.crypto.encryption import paillier_lookup
from xtrace_sdk.x_vec.utils.xtrace_types import PaillierKeyPair

random.seed(42)

@pytest.fixture
def key_pair(key_len:int, alpha_len:int) -> PaillierKeyPair:
    """Fixture to generate a key pair for testing."""
    keys = paillier_lookup.PaillierLookup.key_gen(key_len, alpha_len)
    return keys


@pytest.fixture
def random_numbers(key_len:int, num_runs:int) -> list:
    """Fixture to generate a list of random numbers for testing."""
    return [random.randint(1, 2**key_len) for _ in range(num_runs)]


def test_encrypt_and_decrypt(key_pair: PaillierKeyPair, random_numbers: list) -> None:
    """Test the encryption and decryption process."""
    for num in random_numbers:
        ciphertext = paillier_lookup.PaillierLookup.encrypt(num, key_pair['pk'], key_pair['g_table'], key_pair['noise_table'], key_pair['message_chunks'])
        decrypted_num = paillier_lookup.PaillierLookup.decrypt(ciphertext, key_pair)
        assert decrypted_num == num, f"Decrypted number {decrypted_num} does not match original number {num}."


def test_addition_of_ciphers(key_pair: PaillierKeyPair, random_numbers: list, key_len:int) -> None:
    """Test the addition of encrypted numbers."""
    encrypted_numbers = [paillier_lookup.PaillierLookup.encrypt(num, key_pair['pk'], key_pair['g_table'], key_pair['noise_table'], key_pair['message_chunks']) for num in random_numbers]

    # Add the encrypted numbers

    for i in range(len(encrypted_numbers)):
        random_offset = random.randint(1, 2**key_len)
        enc_offset = paillier_lookup.PaillierLookup.encrypt(random_offset, key_pair['pk'], key_pair['g_table'], key_pair['noise_table'], key_pair['message_chunks'])
        encrypted_sum = paillier_lookup.PaillierLookup.add(enc_offset, encrypted_numbers[i], key_pair['pk'])
        # Decrypt the sum
        decrypted_sum = paillier_lookup.PaillierLookup.decrypt(encrypted_sum, key_pair)
        # Check if the decrypted sum matches the sum of original numbers
        assert decrypted_sum == random_numbers[i]+random_offset, f"Decrypted sum {decrypted_sum} does not match expected sum {random_numbers[i]+random_offset}."