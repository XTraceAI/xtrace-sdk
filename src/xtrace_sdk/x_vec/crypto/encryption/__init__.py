"""Encryption primitives used by the XTrace Vec.

Includes:

- ``aes``: AES encryption for chunk content
- ``paillier``: Paillier homomorphic encryption
- ``paillier_lookup``: Paillier variant with precomputed lookup tables
- ``goldwasser_micali``: Goldwasser-Micali homomorphic encryption (experimental)
- ``homomorphic_base``: base class for homomorphic encryption schemes
"""