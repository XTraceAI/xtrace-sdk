import base64
import hashlib
import json
import logging
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

_log = logging.getLogger(__name__)

from xtrace_sdk.x_vec.crypto.encryption.aes import AESClient  # noqa: E402
from xtrace_sdk.x_vec.crypto.key_provider import (  # noqa: E402
    AWSKMSKeyProvider,
    KeyProvider,
    PassphraseKeyProvider,
)
from xtrace_sdk.x_vec.crypto.paillier_client import PaillierClient  # noqa: E402
from xtrace_sdk.x_vec.crypto.paillier_lookup_client import PaillierLookupClient  # noqa: E402
from xtrace_sdk.x_vec.utils.settings import SUPPORTED_HOMOMORPHIC_CLIENTS  # noqa: E402

if TYPE_CHECKING:
    from xtrace_sdk.integrations.xtrace import XTraceIntegration


@runtime_checkable
class HomomorphicClient(Protocol):
    """Protocol describing all methods used by ExecutionContext on a homomorphic client."""

    def encrypt_vec_one(self, embd: list[int]) -> list[int]: ...
    def encrypt_vec_batch(self, embds: list[list[int]]) -> list[list[int]]: ...
    def decode_hamming_client_one(self, cipher: list[int | bytes]) -> int: ...
    def decode_hamming_client_batch(self, ciphers: list[list[int | bytes]]) -> list[int]: ...
    def stringify_sk(self) -> str: ...
    def stringify_pk(self) -> str: ...
    def stringify_config(self) -> str: ...
    def load_stringified_keys(self, pk: str, sk: str) -> None: ...


def _resolve_key_provider(
    key_provider: KeyProvider | None,
    passphrase: str | None,
    salt: bytes | None,
) -> KeyProvider:
    """Return a concrete KeyProvider, preferring an explicit provider over passphrase."""
    if key_provider is not None:
        return key_provider
    if passphrase is not None:
        return PassphraseKeyProvider(passphrase, salt=salt)
    raise ValueError("Either key_provider or passphrase must be supplied.")


class ExecutionContext:
    """Bundles a homomorphic encryption client and an AES key under a single key-provider-protected object.

    An ``ExecutionContext`` is the root secret for a XTrace deployment. It holds:

    - A homomorphic client (``PaillierClient`` or ``PaillierLookupClient``) whose secret key is
      used to decrypt Hamming distances returned by the XTrace server.
    - An AES key supplied by a :class:`KeyProvider`, used to encrypt chunk content before upload.

    The secret key is **never transmitted in plaintext** — it is AES-encrypted with the
    key provider's key before any remote storage.

    :param homomorphic_client: An initialised ``PaillierClient`` or ``PaillierLookupClient``.
    :param key_provider: A :class:`KeyProvider` that supplies the AES encryption key.
    :param context_id: Optional deterministic ID. If omitted, one is derived from a SHA-256
        hash of the public key and configuration.
    """

    def __init__(self, homomorphic_client: HomomorphicClient, key_provider: KeyProvider, context_id: str | None = None) -> None:
        self.homomorphic = homomorphic_client
        self.key_provider = key_provider
        self.aes = AESClient(key_provider.get_key())
        if context_id is None:
            self.id = self.hash() #slow, but only need to do once.
        else:
            self.id = context_id

    @classmethod
    def create(
        cls,
        passphrase: str | None = None,
        homomorphic_client_type: str = "paillier",
        embedding_length: int = 512,
        key_len: int = 1024,
        salt: bytes | None = None,
        path: str | None = None,
        key_provider: KeyProvider | None = None,
    ) -> "ExecutionContext":
        """Create a new execution context and optionally save it to disk.

        Supply either ``key_provider`` or ``passphrase`` (with optional ``salt``).
        If both are given, ``key_provider`` takes precedence.

        :param passphrase: Secret passphrase used to derive the AES encryption key and protect
            the homomorphic secret key at rest.
        :param homomorphic_client_type: ``"paillier"`` or ``"paillier_lookup"``.
        :param embedding_length: Dimension of the binary embedding vectors (must match the model).
        :param key_len: RSA modulus size in bits (minimum ``1024``).
        :param salt: Optional salt bytes for passphrase-based key derivation.
        :param path: If provided, persist the context to this file path via :meth:`save_to_disk`.
        :param key_provider: Explicit :class:`KeyProvider` instance (e.g. :class:`AWSKMSKeyProvider`).
        :return: Initialised ``ExecutionContext``.
        :raises ValueError: If ``homomorphic_client_type`` is not recognised or
            ``embedding_length >= key_len``.
        """
        provider = _resolve_key_provider(key_provider, passphrase, salt)

        if homomorphic_client_type.lower() in ("paillier", "paillier_lookup") and embedding_length >= key_len:
            raise ValueError(
                f"embedding_length ({embedding_length}) must be strictly less than key_len ({key_len}). "
                f"The Paillier-Lookup scheme requires embed_len < key_len to guarantee the padded "
                f"plaintext fits within the RSA modulus."
            )
        homomorphic_client: PaillierClient | PaillierLookupClient
        if homomorphic_client_type.lower() == "paillier":
            homomorphic_client = PaillierClient(embed_len=embedding_length, key_len=key_len)
        elif homomorphic_client_type.lower() == "paillier_lookup":
            homomorphic_client = PaillierLookupClient(embed_len=embedding_length, key_len=key_len)
        else:
            raise ValueError(f"Unsupported homomorphic client type: {homomorphic_client_type}")
        ctx = cls(homomorphic_client, provider)
        if path:
            ctx.save_to_disk(path)
        return ctx

    @property
    def device(self) -> str:
        """Active compute backend: ``"cpu"`` or ``"gpu"``."""
        try:
            return getattr(self.homomorphic, "device", "unknown")
        except Exception:
            return "unknown"

    def to_dict_enc(self) -> dict:
        """Return a serialisable dict with the secret key AES-encrypted under the key provider."""
        d = {
            "sk": self.aes.encrypt(self.homomorphic.stringify_sk()).decode('utf-8'),
            "pk": self.homomorphic.stringify_pk(),
            "type": type(self.homomorphic).__name__,
            "config": self._config_with_device(),
            "key_provider": self.key_provider.provider_id(),
            "wrapped_key": base64.b64encode(self.key_provider.wrap_key()).decode('utf-8'),
        }
        return d

    def to_dict_plain(self) -> dict:
        """Return a serialisable dict with the secret key in plaintext. Do not persist or transmit."""
        return {
            "sk": self.homomorphic.stringify_sk(),
            "pk": self.homomorphic.stringify_pk(),
            "type": type(self.homomorphic).__name__,
            "config": self._config_with_device()
        }

    def embed_len(self) -> int:
        """Embedding vector dimension this context was configured for."""
        cfg = json.loads(self.homomorphic.stringify_config())
        return cfg.get("embed_len",0)

    def key_len(self) -> int:
        """RSA modulus size in bits used for key generation."""
        cfg = json.loads(self.homomorphic.stringify_config())
        return cfg.get("key_len",0)

    def __str__(self) -> str:
        return json.dumps(self.to_dict_enc(),indent=2)

    def hash(self) -> str:
        """Compute a deterministic SHA-256 fingerprint of this context's cryptographic identity.

        The ``device`` field is excluded so that CPU and GPU contexts sharing the same keys compare
        as equal.

        :return: Hex-encoded SHA-256 digest.
        :rtype: str
        """
        _log.debug("Computing execution context hash...")
        data = self.to_dict_plain()
        data['config'] = self.homomorphic.stringify_config()
        str_data = json.dumps(data)
        hash_obj = hashlib.sha256(str_data.encode('utf-8'))
        return hash_obj.hexdigest()

    def __hash__(self) -> int:
        return int(self.hash(), 16)

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, ExecutionContext):
            return False
        return self.hash() == other.hash()

    def _config_with_device(self) -> str:
        cfg = json.loads(self.homomorphic.stringify_config())
        return json.dumps(cfg)

    def serialize_exec_context(self) -> str:
        """Serialise the execution context to a JSON string suitable for storage or transmission.

        The secret key is AES-encrypted under the key provider before inclusion.

        :return: JSON string representing the encrypted execution context.
        :rtype: str
        :raises ValueError: If the homomorphic client type is not supported.
        """
        homomorphic_type = type(self.homomorphic).__name__

        if homomorphic_type not in SUPPORTED_HOMOMORPHIC_CLIENTS:
            raise ValueError(f"Unsupported homomorphic client type: {homomorphic_type}")

        exec_context = self.to_dict_enc()

        return json.dumps(exec_context)

    @classmethod
    def _from_serialized_exec_context(
        cls,
        json_obj: dict,
        passphrase: str | None = None,
        key_provider: KeyProvider | None = None,
        context_id: str | None = None,
    ) -> "ExecutionContext":
        """Reconstruct an ``ExecutionContext`` from a previously serialised dict.

        Supply either ``key_provider`` or ``passphrase``. For passphrase-based contexts the
        salt is read from the stored ``wrapped_key`` field automatically.

        :param json_obj: Dict produced by :meth:`to_dict_enc`.
        :param passphrase: Passphrase for passphrase-based contexts.
        :param key_provider: Explicit :class:`KeyProvider` to use for decryption.
        :param context_id: Optional context ID to attach; if ``None`` one is recomputed.
        :return: Restored ``ExecutionContext``.
        :raises ValueError: If the stored homomorphic client type is not supported.
        """
        # Reconstruct the key provider
        if key_provider is not None:
            provider = key_provider
        elif json_obj.get("key_provider") == "passphrase":
            if passphrase is None:
                raise ValueError("passphrase is required to load a passphrase-based context.")
            wrapped = base64.b64decode(json_obj["wrapped_key"])
            provider = PassphraseKeyProvider.from_wrapped(wrapped, passphrase=passphrase)
        else:
            raise ValueError("Either key_provider or passphrase must be supplied.")

        aes_client = AESClient(provider.get_key())
        sk = aes_client.decrypt(json_obj["sk"].encode('utf-8'))
        config = json.loads(json_obj["config"])
        if json_obj["type"] not in SUPPORTED_HOMOMORPHIC_CLIENTS:
            raise ValueError(f"Unsupported homomorphic client type: {json_obj['type']}")

        concrete_client: PaillierClient | PaillierLookupClient
        if json_obj["type"] == "PaillierLookupClient":
            concrete_client = PaillierLookupClient(embed_len=config["embed_len"], key_len=config["key_len"], alpha_len=config["alpha_len"], skip_key_gen=True)
        elif json_obj["type"] == "PaillierClient":
            concrete_client = PaillierClient(embed_len=config["embed_len"], key_len=config["key_len"], skip_key_gen=True)

        concrete_client.load_stringified_keys(json_obj['pk'], sk)


        # Inject tables if available to skip recomputation
        precomputed_tables = json_obj.get("tables")
        if isinstance(concrete_client, PaillierLookupClient):
            concrete_client.load_config(config, precomputed_tables=precomputed_tables)
        else:
            concrete_client.load_config(config)

        return cls(concrete_client, provider, context_id=context_id)

    def dump_tables(self) -> dict:
        """Dump precomputed encryption tables (Paillier-Lookup only) for caching.

        :return: Dict containing ``g_table`` and ``noise_table``, or an empty dict if the
            underlying client does not support table export.
        :rtype: dict
        """
        dump_fn = getattr(self.homomorphic, "dump_tables", None)
        return dump_fn() if dump_fn is not None else {}

    def save_to_disk(self, path: str) -> None:
        """Persist the execution context to a local file.

        The secret key is AES-encrypted before writing. The passphrase/key is not stored.

        :param path: File path to write to.
        :type path: str
        """
        exec_context = self.to_dict_enc()
        with open(path, 'w') as f:
            json.dump(exec_context, f)


    @classmethod
    def load_from_disk(
        cls,
        passphrase: str | None = None,
        path: str = "",
        key_provider: KeyProvider | None = None,
    ) -> "ExecutionContext":
        """Load an ``ExecutionContext`` from a file previously saved with :meth:`save_to_disk`.

        :param passphrase: Passphrase for passphrase-based contexts.
        :param path: File path to read from.
        :param salt: Optional salt for passphrase-based key derivation.
        :param key_provider: Explicit :class:`KeyProvider` (e.g. :class:`AWSKMSKeyProvider`).
        :return: Restored ``ExecutionContext``.
        """
        with open(path) as f:
            exec_context = f.read()

        json_obj = json.loads(exec_context)
        return cls._from_serialized_exec_context(json_obj, passphrase=passphrase, key_provider=key_provider)

    async def save_to_remote(self, integration: "XTraceIntegration") -> str:
        """Upload the execution context to XTrace remote storage.

        The secret key is AES-encrypted under the key provider before upload — XTrace never sees
        the plaintext secret key or the passphrase.

        :param integration: Authenticated :class:`~xtrace_sdk.integrations.xtrace.XTraceIntegration` instance.
        :return: The ``context_id`` assigned by the server.
        :rtype: str
        """
        return await integration.store_exec_context(self.to_dict_enc(), self.id)

    @classmethod
    async def load_from_remote(
        cls,
        passphrase: str | None = None,
        context_id: str = "",
        integration: "XTraceIntegration | None" = None,
        key_provider: KeyProvider | None = None,
    ) -> "ExecutionContext":
        """Fetch and decrypt an ``ExecutionContext`` from XTrace remote storage.

        :param passphrase: Passphrase for passphrase-based contexts.
        :param context_id: ID returned when the context was originally saved.
        :param integration: Authenticated :class:`~xtrace_sdk.integrations.xtrace.XTraceIntegration` instance.
        :param salt: Optional salt for passphrase-based key derivation.
        :param key_provider: Explicit :class:`KeyProvider` (e.g. :class:`AWSKMSKeyProvider`).
        :return: Restored ``ExecutionContext``.
        """
        if integration is None:
            raise ValueError("integration is required.")
        serial_ctx = await integration.get_serialized_exec_context(context_id)
        
        return cls._from_serialized_exec_context(serial_ctx, key_provider=key_provider, passphrase=passphrase, context_id=str(context_id))
