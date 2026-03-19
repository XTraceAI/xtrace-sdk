import json
import os

import gmpy2

from typing import Optional, Union

from xtrace_sdk.x_vec.crypto.encryption.paillier_lookup import Paillier_Lookup
from xtrace_sdk.x_vec.crypto.hamming_client_base import HammingClientBase
from xtrace_sdk.x_vec.utils.xtrace_types import PaillierEncryptedNumber, PaillierLookupKeyPair

DEVICE = os.getenv("DEVICE", "cpu")



class PaillierLookupGPU:
    """Wrapper for the C++ GPU implementation of the Paillier-Lookup cryptography system.

    This class is not instantiated directly — use :class:`PaillierLookupClient` which dispatches
    to this backend automatically when ``DEVICE=gpu``.
    """

    def __init__(self,embed_len:int=512, key_len:int=1024, alpha_len:int=50, skip_key_gen:bool=False) -> None:
        try:
            #lazy import here to avoid import error when GPU backend is not available
            from .paillier_GPU_lookup_client import PaillierGPULookupClient
        except Exception as e:
            raise ImportError(
                "GPU backend unavailable. Expected Paillier GPU .so under src/crypto/ "
                "Please refer to the documentation for compilation instructions."
            ) from e
        self.gpu_client = PaillierGPULookupClient(embed_len=embed_len, key_len=key_len, alpha_len=alpha_len ,skip_key_gen=skip_key_gen)

    def encrypt(self, embds: list[list[int]]) -> list[PaillierEncryptedNumber]:
        """Encrypt a batch of binary embedding vectors on GPU.

        :param embds: List of binary vectors with values in {0, 1}.
        :type embds: list[list[int]]
        :return: List of encrypted vectors.
        :rtype: list[PaillierEncryptedNumber]
        """
        # Fast path using raw bytes integration. Eliminates heavy python string operations.
        bytes_batch = self.gpu_client.encrypt_bytes(embds)
        res = []
        for bytes_list in bytes_batch:
            res.append([int.from_bytes(b, byteorder='little') for b in bytes_list])
        return res

    def decode_hamming_client(self, cipher: list[list[int | bytes]]) -> list[int]:
        """Decrypt a batch of encrypted Hamming distances on GPU.

        :param cipher: Batch of encrypted Hamming encodings from the server.
        :type cipher: list[list[int]]
        :return: List of plain-text Hamming distances.
        :rtype: list[int]
        """
        chunk_byte_len = (self.gpu_client.key_len * 4) // 8
        
        bytes_batch = []
        for c in cipher:
            # c is a list of chunks (either int or bytes)
            record_bytes = []
            for h in c:
                if isinstance(h, bytes):
                    # Ensure bytes are exactly chunk_byte_len.
                    # Server might send variable length if it's using compact integer representation.
                    if len(h) != chunk_byte_len:
                        h = int.from_bytes(h, byteorder='little').to_bytes(chunk_byte_len, byteorder='little')
                    record_bytes.append(h)
                else:
                    record_bytes.append(int(h).to_bytes(chunk_byte_len, byteorder='little'))
            bytes_batch.append(record_bytes)
            
        return self.gpu_client.decode_hamming_client_bytes(bytes_batch)
    
    def stringify_pk(self) -> str:
        """Return a JSON string representation of the public key."""
        return self.gpu_client.stringify_pk()

    def stringify_sk(self) -> str:
        """Return a JSON string representation of the secret key."""
        return self.gpu_client.stringify_sk()

    def stringify_config(self) -> str:
        """Return a JSON string representation of the encryption configuration."""
        return self.gpu_client.stringify_config()

    def load_stringified_keys(self, pk: str, sk: str) -> None:
        """Load public and secret keys from their JSON string representations.

        :param pk: JSON-encoded public key produced by :meth:`stringify_pk`.
        :param sk: JSON-encoded secret key produced by :meth:`stringify_sk`.
        """
        self.gpu_client.load_stringified_keys(pk, sk)

    def load_config(self, config: dict, precomputed_tables: dict | None = None, tables_bytes: dict | None = None) -> None:
        """Load encryption configuration and optionally restore precomputed tables.

        :param config: Configuration dict (``embed_len``, ``key_len``, etc.).
        :type config: dict
        :param precomputed_tables: Optional dict of precomputed ``g_table`` / ``noise_table``
            to skip recomputation.
        :param tables_bytes: Optional dict of the same tables in raw binary format (faster to load).
        """
        if tables_bytes:
            self.gpu_client.load_config_bytes(config, tables=tables_bytes)
        elif precomputed_tables:
            self.gpu_client.load_config(config, tables=precomputed_tables)
        else:
            self.gpu_client.load_config(config)

    def dump_tables(self) -> dict:
        """Dump precomputed tables as a dict of Python-native types for caching."""
        return self.gpu_client.dump_tables()

    def dump_tables_bytes(self) -> dict:
        """Dump precomputed tables in compact raw binary format for caching."""
        return self.gpu_client.dump_tables_bytes()



class PaillierLookupCPU:
    """This is an implementation of paillier cryptography system optimized for caculating hamming distance between
    two binary vectors.
    """

    def __init__(self,embed_len:int=512, key_len:int=1024, alpha_len:int=50, skip_key_gen:bool=False) -> None:
        """constructor

        :param key_len: the length (in bit) of the public key and secret key generated for Paillier Cryptosystem, defaults to 1024
        :type key_len: int, optional
        :param embed_len: the length of the embedding vector this paillier client supports, defaults to 512
        :type embed_len: int, optional
        :param key_path: the path to the keys, defaults to None
        :type key_path: str, optional
        """
        self.alpha_len = alpha_len
        self.key_len = key_len
        self.chunk_len = key_len * 2  
        if embed_len > key_len:
            self.chunk_num = 2*embed_len // self.chunk_len + int((2 *embed_len) % self.chunk_len != 0)
        else:
            self.chunk_num = 1

        self.keys: PaillierLookupKeyPair | None
        if not skip_key_gen:
            self.keys = Paillier_Lookup.key_gen(key_len, alpha_len=self.alpha_len)
        else:
            self.keys = None

        self.embed_len = embed_len



    def stringify_pk(self) -> str:
        """stringify public key for networking/storage purpose

        :return: stringified public key
        :rtype: str
        """
        assert self.keys is not None, "Keys not initialized"
        pk = self.keys['pk']
        return json.dumps({'g': int(pk['g']), 'n': int(pk['n']), 'n_squared': int(pk['n_squared']), 'g_n': int(pk['g_n'])})

    def stringify_sk(self) -> str:
        """stringify secret key for networking/storage purpose

        :return: stringified secret key
        :rtype: str
        """
        assert self.keys is not None, "Keys not initialized"
        sk = self.keys['sk']
        return json.dumps({'phi': int(sk['phi']), 'a': int(sk['a']), 'g_a_inv': int(sk['g_a_inv'])})

    def stringify_config(self) -> str:
        """stringify crypto context for networking/storage purpose

        :return: stringified crypto context
        :rtype: str
        """
        assert self.keys is not None, "Keys not initialized"
        str_config = {}
        str_config['message_chunks'] = int(self.keys['message_chunks'])
        str_config['alpha_len'] = self.alpha_len
        str_config['embed_len'] = self.embed_len
        str_config['key_len'] = self.key_len
        return json.dumps(str_config)
    
    def dump_tables(self) -> dict:
        """Dump g_table and noise_table for caching"""
        assert self.keys is not None, "Keys not initialized"
        return Paillier_Lookup.dump_tables(self.keys)

    def load_stringified_keys(self, pk: str, sk: str) -> None:
        """load stringified keys

        :param pk: stringified public key
        :type pk: str
        :param sk: stringified secret key
        :type sk: str
        """
        pk_data = json.loads(pk)
        sk_data = json.loads(sk)
        self.keys = {
            'pk': {
                'g': gmpy2.mpz(pk_data['g']),
                'n': gmpy2.mpz(pk_data['n']),
                'n_squared': gmpy2.mpz(pk_data['n_squared']),
                'g_n': gmpy2.mpz(pk_data['g_n']),
            },
            'sk': {
                'phi': gmpy2.mpz(sk_data['phi']),
                'a': gmpy2.mpz(sk_data['a']),
                'g_a_inv': gmpy2.mpz(sk_data['g_a_inv']),
            },
            'g_table': {},
            'noise_table': [],
            'key_len': 0,
            'message_chunks': 0,
        }

    def load_config(self, config: dict, precomputed_tables: dict | None = None) -> None:
        """load crypto context from a json string

        :param context: the json string containing crypto context
        :type context: str
        :param precomputed_tables: optional dict containing 'g_table' and 'noise_table' to skip recomputation
        """
        assert self.keys is not None, "Keys not initialized"
        g = self.keys['pk']['g']
        n = self.keys['pk']['n']
        key_len = config['key_len']
        
        if precomputed_tables:
            g_table, noise_table = Paillier_Lookup.load_tables(precomputed_tables)
            self.keys['g_table'] = g_table
            self.keys['noise_table'] = noise_table
        else:
            self.keys['g_table'] = Paillier_Lookup.precompute_g_table(g, n, key_len)
            self.keys['noise_table'] = Paillier_Lookup.precompute_noise_table(g, n)
            
        self.keys['message_chunks'] = int(config['message_chunks'])

    def id2power(self,id_:int) -> tuple[int, int]:
        """Helper: return chunk index and corresponding power of 2 for a given position in the padded array.

        :param id_: Index of an entry in the embedding vector.
        :type id_: int
        """
        return id_ // self.chunk_len, self.chunk_len - 1 - id_%self.chunk_len 
    
    def encrypt(self,embd:list[int]) -> PaillierEncryptedNumber:
        """This function implements the encryption scheme on embedding vectors that needs to be run on client side.

        :param embd: the embedding vector to be encrypted
        :type embd: iterable[0,1]
        :return: return a list of length chunk_num contaning PaillierEncryptedNumber
        :rtype: PaillierEncryptedNumber
        """
        assert self.keys is not None, "Keys not initialized"
        assert len(embd) == self.embed_len
        padded_embd = []

        for i in range(self.embed_len):
            assert embd[i] in [0,1], "Embedding vector must be binary"
            padded_embd += ['0',str(embd[i])]

        int_repr =  [int("".join(padded_embd[i*self.chunk_len : (i+1)*self.chunk_len]),2) for i in range(self.chunk_num)]
        return [Paillier_Lookup.encrypt(i,self.keys['pk'],self.keys['g_table'], self.keys['noise_table'], self.keys['message_chunks']) for i in int_repr]

    def decode_hamming_client(self, cipher: list[int | bytes]) -> int:
        """Given a PaillierEncryptedNumber returned from server, calculate the hamming distance encoded
        """
        assert self.keys is not None, "Keys not initialized"
        # cipher is a list of chunks
        de_c = []
        for c in cipher:
            if isinstance(c, bytes):
                c = int.from_bytes(c, byteorder='little')
            de_c.append(Paillier_Lookup.decrypt(c, self.keys))

        bin_c_truncated: list[str] = [f"{d:b}" for d in de_c]
        bin_c_str = ""
        for s in bin_c_truncated:
            if len(s) != self.chunk_len and len(s) != 0:
                s = "0" * (self.chunk_len-len(s)) + s
            bin_c_str += s
        ham = 0
        for i in range(1,len(bin_c_str),2):
            ham += int(bin_c_str[i])
        return ham
    

class PaillierLookupClient(HammingClientBase):
    """Paillier-Lookup homomorphic encryption client optimised for computing encrypted Hamming
    distances between binary embedding vectors.

    Uses precomputed lookup tables to accelerate encryption, making it significantly faster than
    :class:`PaillierClient` for large collections. Dispatches to CPU or GPU backend via the
    ``DEVICE`` environment variable (``cpu`` by default).
    """

    def __init__(self,embed_len:int=512, key_len:int=1024, alpha_len:int=50, skip_key_gen:bool=False) -> None:
        """constructor

        :param key_len: the length (in bit) of the public key and secret key generated for Paillier Cryptosystem, defaults to 1024
        :type key_len: int, optional
        :param embed_len: the length of the embedding vector this paillier client supports, defaults to 512
        :type embed_len: int, optional
        :param key_path: the path to the keys, defaults to None
        :type key_path: str, optional
        """
        # Store attributes directly for backward compatibility
        self.alpha_len = alpha_len
        self.key_len = key_len
        self.embed_len = embed_len
        self.chunk_len = key_len * 2
        if embed_len > key_len:
            self.chunk_num = 2*embed_len // self.chunk_len + int((2 *embed_len) % self.chunk_len != 0)
        else:
            self.chunk_num = 1
        
        self.device = DEVICE
        self.client: PaillierLookupGPU | PaillierLookupCPU
        if self.device == "gpu":
            if not self.has_gpu():
                raise RuntimeError("GPU requested via DEVICE=gpu, but GPU backend is unavailable.")
            self.client = PaillierLookupGPU(embed_len=embed_len, key_len=key_len, alpha_len=alpha_len, skip_key_gen=skip_key_gen)
        else:
            self.client = PaillierLookupCPU(embed_len=embed_len, key_len=key_len, alpha_len=alpha_len, skip_key_gen=skip_key_gen)
        
    @staticmethod
    def has_gpu() -> bool:
        """Return ``True`` if the GPU backend extension is available and loadable.

        :rtype: bool
        """
        try:
            from xtrace_sdk.x_vec.crypto.paillier_lookup_client import PaillierLookupGPU
            PaillierLookupGPU(skip_key_gen=True)
            return True
        except Exception:
            return False

    def encrypt_vec_one(self, embd: list[int]) -> PaillierEncryptedNumber:
        """Encrypt a single binary embedding vector.

        :param embd: Binary vector of length ``embed_len`` with values in {0, 1}.
        :type embd: list[int]
        :return: Encrypted vector.
        :rtype: PaillierEncryptedNumber
        """
        if isinstance(self.client, PaillierLookupCPU):
            return self.client.encrypt(embd)
        else:
            return self.client.encrypt([embd])[0]

    def encrypt_vec_batch(self, embds: list[list[int]]) -> list[PaillierEncryptedNumber]:
        """Encrypt a batch of binary embedding vectors.

        :param embds: List of binary vectors, each of length ``embed_len`` with values in {0, 1}.
        :type embds: list[list[int]]
        :return: List of encrypted vectors.
        :rtype: list[PaillierEncryptedNumber]
        """
        if isinstance(self.client, PaillierLookupCPU):
            return [self.client.encrypt(embd) for embd in embds]
        else:
            return self.client.encrypt(embds)

    def decode_hamming_client_one(self, cipher: list[int | bytes]) -> int:
        """Decrypt a single encrypted Hamming distance returned by the XTrace server.

        :param cipher: Encrypted Hamming encoding.
        :type cipher: PaillierEncryptedNumber
        :return: Plain-text Hamming distance.
        :rtype: int
        """
        if isinstance(self.client, PaillierLookupCPU):
            return self.client.decode_hamming_client(cipher)
        else:
            return self.client.decode_hamming_client([cipher])[0]

    def decode_hamming_client_batch(self, ciphers: list[list[int | bytes]]) -> list[int]:
        """Decrypt a batch of encrypted Hamming distances returned by the XTrace server.

        :param ciphers: List of encrypted Hamming encodings.
        :type ciphers: list[PaillierEncryptedNumber]
        :return: List of plain-text Hamming distances.
        :rtype: list[int]
        """
        if isinstance(self.client, PaillierLookupCPU):
            return [self.client.decode_hamming_client(cipher) for cipher in ciphers]
        else:
            return self.client.decode_hamming_client(ciphers)

    def stringify_pk(self) -> str:
        """Return a JSON string representation of the public key."""
        return self.client.stringify_pk()

    def stringify_sk(self) -> str:
        """Return a JSON string representation of the secret key."""
        return self.client.stringify_sk()

    def stringify_config(self) -> str:
        """Return a JSON string representation of the encryption configuration."""
        return self.client.stringify_config()

    def load_stringified_keys(self, pk: str, sk: str) -> None:
        """Load public and secret keys from their JSON string representations.

        :param pk: JSON-encoded public key produced by :meth:`stringify_pk`.
        :param sk: JSON-encoded secret key produced by :meth:`stringify_sk`.
        """
        self.client.load_stringified_keys(pk, sk)

    def load_config(self, config: dict, precomputed_tables: dict | None = None) -> None:
        """Load encryption configuration and optionally restore precomputed tables.

        :param config: Configuration dict (``embed_len``, ``key_len``, ``alpha_len``, etc.).
        :type config: dict
        :param precomputed_tables: Optional dict of precomputed ``g_table`` / ``noise_table``
            to skip recomputation on the CPU backend.
        """
        if self.device == "cpu":
            self.client.load_config(config, precomputed_tables=precomputed_tables)
        else:
             # GPU client might not support explicit table loading yet, fall back to standard
            self.client.load_config(config)

    def __getstate__(self) -> dict:
        """Support for pickling.
        We serialize the configuration, keys, and for CPU, the precomputed tables.
        """
        state = {
            "device": self.device,
            "embed_len": self.embed_len,
            "key_len": self.key_len,
            "alpha_len": self.alpha_len,
            "pk": self.stringify_pk(),
            "sk": self.stringify_sk(),
            "config": json.loads(self.stringify_config())
        }
        
        # Always try to fetch and standardise tables into raw py::bytes representation.
        # This provides a 60-80% reduction in serialization size compared to list of hex strings.
        try:
            if hasattr(self.client, "dump_tables_bytes"):
                raw_bytes = self.client.dump_tables_bytes()
                if raw_bytes:
                    state["tables_bytes"] = raw_bytes
            else:
                raw_tables = self.client.dump_tables()
                if raw_tables:
                    if self.device == "cpu":
                        # CPU dump_tables returns:
                        # g_table: dict[int, list[int]], noise_table: list[int]
                        # Convert to flat lists of hex strings
                        flat_g = []
                        # Ensure we iterate in order of the chunk indices
                        for chunk_idx in sorted(raw_tables.get("g_table", {}).keys()):
                            for val in raw_tables["g_table"][chunk_idx]:
                                flat_g.append(format(val, "x"))
                                
                        flat_noise = [format(val, "x") for val in raw_tables.get("noise_table", [])]

                        state["tables"] = {
                            "g_table": flat_g,
                            "noise_table": flat_noise
                        }
                    elif self.device == "gpu":
                        # GPU dump_tables already returns flat lists of hex strings
                        state["tables"] = raw_tables
        except Exception:
            pass
            
        return state

    def __setstate__(self, state: dict) -> None:
        """Restore from pickle"""
        self.device = DEVICE
        self.embed_len = state["embed_len"]
        self.key_len = state["key_len"]
        self.alpha_len = state["alpha_len"]
        self.chunk_len = self.key_len * 2
        
        if self.embed_len > self.key_len:
            self.chunk_num = 2*self.embed_len // self.chunk_len + int((2 *self.embed_len) % self.chunk_len != 0)
        else:
            self.chunk_num = 1
            
        # Re-initialize the inner client
        if self.device == "cpu":
            self.client = PaillierLookupCPU(
                embed_len=self.embed_len, 
                key_len=self.key_len, 
                alpha_len=self.alpha_len, 
                skip_key_gen=True
            )
        elif self.device == "gpu":
            self.client = PaillierLookupGPU(
                embed_len=self.embed_len, 
                key_len=self.key_len, 
                alpha_len=self.alpha_len, 
                skip_key_gen=True
            )

        # Load keys
        self.client.load_stringified_keys(state["pk"], state["sk"])
        
        # Load tables if available
        if "tables_bytes" in state:
             # Fast path: raw binary bytes support
             tables_bytes = state["tables_bytes"]
             # Check if device supports binary byte loading natively
             if hasattr(self.client, "load_config") and "tables_bytes" in self.client.load_config.__code__.co_varnames and isinstance(self.client, PaillierLookupGPU):
                 self.client.load_config(state["config"], tables_bytes=tables_bytes)
             else:
                 # It indicates a mismatch where the current client can only accept legacy tables but cache has bytes.
                 # Recompute for fallback.
                 self.client.load_config(state["config"])
        elif "tables" in state:
            tables = state["tables"]
            if self.device == "cpu":
                # CPU load_config expects raw python int formats:
                # g_table: dict[str/int, list[int]], noise_table: list[int]
                # We need to un-flatten the hex strings back into this format
                from xtrace_sdk.x_vec.crypto.encryption.paillier_lookup import PAILLIER_MSG_TABLE_SIZE  # type: ignore[attr-defined]
                
                flat_g = tables.get("g_table", [])
                g_dict = {}
                # Recover dict structure by chunking the flat list
                for i in range(0, len(flat_g), PAILLIER_MSG_TABLE_SIZE):
                    chunk_idx = i // PAILLIER_MSG_TABLE_SIZE
                    g_dict[str(chunk_idx)] = [int(hex_val, 16) for hex_val in flat_g[i : i + PAILLIER_MSG_TABLE_SIZE]]
                
                noise_list = [int(hex_val, 16) for hex_val in tables.get("noise_table", [])]
                
                cpu_tables = {
                    "g_table": g_dict,
                    "noise_table": noise_list
                }
                self.client.load_config(state["config"], precomputed_tables=cpu_tables)
            else:
                # GPU client natively expects the flat hex lists (legacy format)
                self.client.load_config(state["config"], precomputed_tables=tables)
        else:
             self.client.load_config(state["config"])

    def dump_tables(self) -> dict:
        # Both CPU and GPU clients now support dump_tables
        try:
             return self.client.dump_tables()
        except AttributeError:
             return {}