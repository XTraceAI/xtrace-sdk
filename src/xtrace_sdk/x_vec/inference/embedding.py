import os
from typing import Any

import aiohttp
import numpy as np


class Embedding:
    """
    This class provides an interface to generate embeddings using different providers.
    Supported providers include "ollama", "openai", and "sentence_transformer".
    It also includes methods to convert float embeddings to binary format.
    """

    def __init__(self, provider: str, model_name: str, dim: int):
        """
        Constructor of the Embedding class with a specific provider and model.

        param provider: The embedding service provider. Supported values are "ollama", "openai", and "sentence_transformer".
        type provider: str
        param model_name: The name of the embedding model to use.
        type model_name: str
        param dim: The dimension of the embedding vectors.
        type dim: int
        raises ValueError: If the provider is not supported.
        """
        URL_MAP = {
            "ollama": "http://localhost:11434/api/embeddings",
            "openai": "https://api.openai.com/v1/embeddings",
            "sentence_transformer": None,
        }
        self.dim = dim

        if provider not in URL_MAP:
            raise ValueError(f"Unsupported provider: {provider}")
        self.provider = provider

        if provider == 'sentence_transformer':
            import sentence_transformers
            self.model = sentence_transformers.SentenceTransformer(model_name, truncate_dim=dim)
        self.url = URL_MAP.get(provider)

        self.model_name = model_name

    def __hash__(self) -> int:
        return hash((self.provider, self.model_name, self.dim))

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Embedding):
            return NotImplemented
        return (self.url, self.model_name, self.dim) == (other.url, other.model_name, other.dim)

    async def embed(self, text: str) -> np.ndarray:
        """Generates an embedding for the given text using the specified provider.

        :param text: The input text to be embedded.
        :type text: str
        :return: A numpy array representing the embedding of the input text.
        :rtype: np.ndarray
        :raises ValueError: If the embedding dimension does not match the expected dimension.
        """
        float_embd: np.ndarray

        if self.provider == "ollama":
            assert self.url is not None
            async with aiohttp.ClientSession() as session, \
                    session.post(self.url, json={"model": self.model_name, "prompt": text}) as resp:
                resp.raise_for_status()
                data = await resp.json()
                float_embd = np.asarray(data.get("embedding", []))

        elif self.provider == "openai":
            assert self.url is not None
            async with aiohttp.ClientSession() as session, \
                    session.post(
                        self.url,
                        headers={"Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}"},
                        json={"model": self.model_name, "input": text, "encoding_format": "float"},
                    ) as resp:
                resp.raise_for_status()
                data = await resp.json()
                float_embd = np.asarray(data.get("data", [])[0].get("embedding", []))

        elif self.provider == "sentence_transformer":
            float_embd = np.asarray(self.model.encode(text).reshape((1, self.dim))[0])

        if float_embd.shape[0] != self.dim:
            raise ValueError(f"Expected embedding dimension {self.dim}, but got {float_embd.shape[0]}")
        return float_embd

    @staticmethod
    def float_2_bin(float_array: np.ndarray | list[float]) -> np.ndarray:
        """Convert a list of floats to a list of binary integers, naive implementation, preserves dimension

        :param float_array: A numpy array or list of floats to be converted.
        :type float_array: np.ndarray or list[float]
        :return: A numpy array of binary integers (0s and 1s).
        :rtype: np.ndarray
        """
        if isinstance(float_array, list):
            float_array = np.asarray(float_array)
        return (float_array > 0).astype(np.int8)

    async def bin_embed(self, text: str) -> np.ndarray:
        """Generates a binary embedding for the given text.
        :param text: The input text to be embedded.
        :type text: str
        :return: A numpy array representing the binary embedding of the input text.
        :rtype: np.ndarray
        """
        float_embedding = await self.embed(text)
        return Embedding.float_2_bin(float_embedding)
