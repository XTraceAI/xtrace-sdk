import pytest
from xtrace_sdk.x_vec.crypto.encryption import paillier
from xtrace_sdk.x_vec.utils.xtrace_types import PaillierKeyPair
import random


@pytest.fixture
def key_pair(key_len:int) -> PaillierKeyPair:
    """Fixture to generate a key pair for testing."""
    keys = paillier.Paillier.key_gen(key_len)
    return keys

@pytest.fixture
def random_numbers(key_len:int, num_runs:int) -> list:
    """Fixture to generate a list of random numbers for testing."""
    return [random.randint(1, 2**key_len) for _ in range(num_runs)]

def test_encrypt_and_decrypt(key_pair: PaillierKeyPair, random_numbers: list) -> None:
    """Test the encryption and decryption process."""
    for num in random_numbers:
        ciphertext = paillier.Paillier.encrypt(num, key_pair['pk'])
        decrypted_num = paillier.Paillier.decrypt(ciphertext, key_pair)
        assert decrypted_num == num, f"Decrypted number {decrypted_num} does not match original number {num}."

def test_addition_of_ciphers(key_pair: PaillierKeyPair, random_numbers: list) -> None:
    """Test the addition of encrypted numbers."""
    encrypted_numbers = [paillier.Paillier.encrypt(num, key_pair['pk']) for num in random_numbers]
    
    # Add the encrypted numbers

    for i in range(len(encrypted_numbers)):
        random_offset = random.randint(1, 2**1024)
        enc_offset = paillier.Paillier.encrypt(random_offset, key_pair['pk'])
        encrypted_sum = paillier.Paillier.add(enc_offset, encrypted_numbers[i], key_pair['pk'])
        # Decrypt the sum
        decrypted_sum = paillier.Paillier.decrypt(encrypted_sum, key_pair)
    
        # Check if the decrypted sum matches the sum of original numbers
        assert decrypted_sum == random_numbers[i]+random_offset, f"Decrypted sum {decrypted_sum} does not match expected sum {random_numbers[i]+random_offset}."