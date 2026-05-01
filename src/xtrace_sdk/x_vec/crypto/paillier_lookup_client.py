import json
from typing import Any

import gmpy2

from xtrace_sdk.x_vec.crypto.device import DeviceMode, resolve_device
from xtrace_sdk.x_vec.crypto.encryption.paillier_lookup import PaillierLookup
from xtrace_sdk.x_vec.crypto.hamming_client_base import HammingClientBase
from xtrace_sdk.x_vec.utils.xtrace_types import EncryptedVector, PaillierEncryptedNumber, PaillierLookupKeyPair


def _load_gpu_backend() -> type[Any]:
    try:
        from xtrace_sdk.x_vec.crypto.paillier_lookup_gpu_ext import PaillierGPULookupClient
    except ImportError as e:
        raise ImportError(
            "GPU extension not built. Run `./build_gpu_binaries.sh` "
            "(requires Docker, NVIDIA driver >= 550)."
        ) from e
    return PaillierGPULookupClient



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
            self.keys = PaillierLookup.key_gen(key_len, alpha_len=self.alpha_len)
        else:
            self.keys = None

        self.embed_len = embed_len



    def stringify_pk(self) -> str:
        """stringify public key for networking/storage purpose

        :return: stringified public key
        :rtype: str
        """
        if self.keys is None:
            raise RuntimeError("Keys not initialized")
        pk = self.keys['pk']
        return json.dumps({'g': int(pk['g']), 'n': int(pk['n']), 'n_squared': int(pk['n_squared']), 'g_n': int(pk['g_n'])})

    def stringify_sk(self) -> str:
        """stringify secret key for networking/storage purpose

        :return: stringified secret key
        :rtype: str
        """
        if self.keys is None:
            raise RuntimeError("Keys not initialized")
        sk = self.keys['sk']
        return json.dumps({'phi': int(sk['phi']), 'a': int(sk['a']), 'g_a_inv': int(sk['g_a_inv'])})

    def stringify_config(self) -> str:
        """stringify crypto context for networking/storage purpose

        :return: stringified crypto context
        :rtype: str
        """
        if self.keys is None:
            raise RuntimeError("Keys not initialized")
        str_config = {}
        str_config['message_chunks'] = int(self.keys['message_chunks'])
        str_config['alpha_len'] = self.alpha_len
        str_config['embed_len'] = self.embed_len
        str_config['key_len'] = self.key_len
        return json.dumps(str_config)
    
    def dump_tables(self) -> dict:
        """Dump g_table and noise_table for caching"""
        if self.keys is None:
            raise RuntimeError("Keys not initialized")
        return PaillierLookup.dump_tables(self.keys)

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
        if self.keys is None:
            raise RuntimeError("Keys not initialized")
        g = self.keys['pk']['g']
        n = self.keys['pk']['n']
        key_len = config['key_len']
        
        if precomputed_tables:
            g_table, noise_table = PaillierLookup.load_tables(precomputed_tables)
            self.keys['g_table'] = g_table
            self.keys['noise_table'] = noise_table
        else:
            self.keys['g_table'] = PaillierLookup.precompute_g_table(g, n, key_len)
            self.keys['noise_table'] = PaillierLookup.precompute_noise_table(g, n)
            
        self.alpha_len = config["alpha_len"]
        self.embed_len = config["embed_len"]
        self.key_len = key_len
        self.chunk_len = self.key_len * 2
        if self.embed_len > self.key_len:
            self.chunk_num = 2 * self.embed_len // self.chunk_len + int((2 * self.embed_len) % self.chunk_len != 0)
        else:
            self.chunk_num = 1
        self.keys["message_chunks"] = int(config["message_chunks"])

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
        if self.keys is None:
            raise RuntimeError("Keys not initialized")
        if len(embd) != self.embed_len:
            raise ValueError(f"Embedding length {len(embd)} does not match expected {self.embed_len}")
        padded_embd = []

        for i in range(self.embed_len):
            if embd[i] not in (0, 1):
                raise ValueError(f"Embedding vector must be binary, got {embd[i]} at index {i}")
            padded_embd += ['0',str(embd[i])]

        int_repr =  [int("".join(padded_embd[i*self.chunk_len : (i+1)*self.chunk_len]),2) for i in range(self.chunk_num)]
        return [PaillierLookup.encrypt(i,self.keys['pk'],self.keys['g_table'], self.keys['noise_table'], self.keys['message_chunks']) for i in int_repr]

    def decode_hamming_client(self, cipher: list[int | bytes]) -> int:
        """Given a PaillierEncryptedNumber returned from server, calculate the hamming distance encoded
        """
        if self.keys is None:
            raise RuntimeError("Keys not initialized")
        # cipher is a list of chunks
        de_c = []
        for c in cipher:
            if isinstance(c, bytes):
                c = int.from_bytes(c, byteorder='little')
            de_c.append(PaillierLookup.decrypt(c, self.keys))

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
    """Paillier-Lookup client that dispatches to CPU or GPU based on the ``device`` kwarg.

    With ``device="auto"`` (the default), the GPU extension is probed at
    construction time and used if available, otherwise the CPU implementation
    runs. ``"cpu"`` and ``"gpu"`` force a backend; ``"gpu"`` raises if the
    extension cannot be loaded.
    """

    def __init__(
        self,
        embed_len: int = 512,
        key_len: int = 1024,
        alpha_len: int = 50,
        skip_key_gen: bool = False,
        device: DeviceMode = "auto",
    ) -> None:
        self.alpha_len = alpha_len
        self.key_len = key_len
        self.embed_len = embed_len
        self.chunk_len = key_len * 2
        if embed_len > key_len:
            self.chunk_num = 2 * embed_len // self.chunk_len + int((2 * embed_len) % self.chunk_len != 0)
        else:
            self.chunk_num = 1

        self.device, gpu_cls = resolve_device(device, _load_gpu_backend)
        if self.device == "gpu" and gpu_cls is not None:
            self.client: Any = gpu_cls(
                embed_len=embed_len,
                key_len=key_len,
                alpha_len=alpha_len,
                skip_key_gen=skip_key_gen,
            )
        else:
            self.client = PaillierLookupCPU(
                embed_len=embed_len,
                key_len=key_len,
                alpha_len=alpha_len,
                skip_key_gen=skip_key_gen,
            )

    @staticmethod
    def has_gpu() -> bool:
        try:
            _load_gpu_backend()(skip_key_gen=True)
            return True
        except Exception:
            return False

    @staticmethod
    def _cipher_to_int(value: Any) -> int:
        if isinstance(value, bytes):
            return int.from_bytes(value, byteorder="little")
        return int(value)

    def encrypt_vec_one(self, embd: list[int]) -> EncryptedVector:
        if isinstance(self.client, PaillierLookupCPU):
            return self.client.encrypt(embd)
        return [self._cipher_to_int(cipher) for cipher in self.client.encrypt([embd])[0]]

    def encrypt_vec_batch(self, embds: list[list[int]]) -> list[EncryptedVector]:
        if isinstance(self.client, PaillierLookupCPU):
            return [self.client.encrypt(embd) for embd in embds]
        return [[self._cipher_to_int(cipher) for cipher in row] for row in self.client.encrypt(embds)]

    def decode_hamming_client_one(self, cipher: list[int | bytes]) -> int:
        if isinstance(self.client, PaillierLookupCPU):
            return self.client.decode_hamming_client(cipher)
        normalized = [self._cipher_to_int(value) for value in cipher]
        return int(self.client.decode_hamming_client([normalized])[0])

    def decode_hamming_client_batch(self, ciphers: list[list[int | bytes]]) -> list[int]:
        if isinstance(self.client, PaillierLookupCPU):
            return [self.client.decode_hamming_client(cipher) for cipher in ciphers]
        normalized = [[self._cipher_to_int(value) for value in cipher] for cipher in ciphers]
        return [int(value) for value in self.client.decode_hamming_client(normalized)]

    def stringify_pk(self) -> str:
        return self.client.stringify_pk()

    def stringify_sk(self) -> str:
        return self.client.stringify_sk()

    def stringify_config(self) -> str:
        return self.client.stringify_config()

    def load_stringified_keys(self, pk: str, sk: str) -> None:
        self.client.load_stringified_keys(pk, sk)

    def load_config(self, config: dict, precomputed_tables: dict | None = None) -> None:
        sanitized_config = {k: v for k, v in config.items() if k != "device"}
        if isinstance(self.client, PaillierLookupCPU):
            self.client.load_config(sanitized_config, precomputed_tables=precomputed_tables)
        elif precomputed_tables is None:
            self.client.load_config(sanitized_config)
        else:
            self.client.load_config(sanitized_config, precomputed_tables)
        self.embed_len = config["embed_len"]
        self.key_len = config["key_len"]
        self.alpha_len = config.get("alpha_len", self.alpha_len)
        self.chunk_len = self.key_len * 2
        if self.embed_len > self.key_len:
            self.chunk_num = 2 * self.embed_len // self.chunk_len + int((2 * self.embed_len) % self.chunk_len != 0)
        else:
            self.chunk_num = 1

    def encode_hamming_server(self, ct1: list[int | bytes], ct2: list[int | bytes]) -> EncryptedVector:
        if len(ct1) != len(ct2):
            raise ValueError("ct1 and ct2 must have the same length")
        if isinstance(self.client, PaillierLookupCPU):
            if self.client.keys is None:
                raise RuntimeError("Keys not initialized")
            return [
                int(PaillierLookup.add(self._cipher_to_int(a), self._cipher_to_int(b), self.client.keys["pk"]))
                for a, b in zip(ct1, ct2, strict=True)
            ]

        result = type(self.client).encode_hamming_server(
            [self._cipher_to_int(cipher) for cipher in ct1],
            [self._cipher_to_int(cipher) for cipher in ct2],
            json.loads(self.stringify_pk()),
        )
        return [self._cipher_to_int(cipher) for cipher in result]

    def dump_tables(self) -> dict:
        dump_fn = getattr(self.client, "dump_tables", None)
        return dump_fn() if dump_fn is not None else {}

    def __getstate__(self) -> dict:
        state = {
            "embed_len": self.embed_len,
            "key_len": self.key_len,
            "alpha_len": self.alpha_len,
            "pk": self.stringify_pk(),
            "sk": self.stringify_sk(),
            "config": json.loads(self.stringify_config()),
        }
        tables = self.dump_tables()
        if tables:
            state["tables"] = tables
        return state

    def __setstate__(self, state: dict) -> None:
        self.embed_len = state["embed_len"]
        self.key_len = state["key_len"]
        self.alpha_len = state["alpha_len"]
        self.chunk_len = self.key_len * 2
        if self.embed_len > self.key_len:
            self.chunk_num = 2 * self.embed_len // self.chunk_len + int((2 * self.embed_len) % self.chunk_len != 0)
        else:
            self.chunk_num = 1

        self.device, gpu_cls = resolve_device("auto", _load_gpu_backend)
        if self.device == "gpu" and gpu_cls is not None:
            self.client = gpu_cls(
                embed_len=self.embed_len,
                key_len=self.key_len,
                alpha_len=self.alpha_len,
                skip_key_gen=True,
            )
        else:
            self.client = PaillierLookupCPU(
                embed_len=self.embed_len,
                key_len=self.key_len,
                alpha_len=self.alpha_len,
                skip_key_gen=True,
            )

        self.client.load_stringified_keys(state["pk"], state["sk"])
        self.load_config(state["config"], precomputed_tables=state.get("tables"))
