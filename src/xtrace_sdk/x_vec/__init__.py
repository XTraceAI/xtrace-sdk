"""XTrace Vec for Python.

This package provides the XTrace Vec for Python, which allows you to
augment your Python AI applications using a privacy-preserving vector
database. The SDK provides a set of tools and libraries to help you
integrate with the XTrace platform and equip your AI applications with
the ability to store, retrieve, and manage memory as needed in a secure
and efficient manner.
"""

from xtrace_sdk.x_vec.crypto.key_provider import (
    AWSKMSKeyProvider,
    KeyProvider,
    PassphraseKeyProvider,
)
from xtrace_sdk.x_vec.data_loaders.loader import DataLoader
from xtrace_sdk.x_vec.inference.embedding import Embedding
from xtrace_sdk.x_vec.inference.llm import InferenceClient
from xtrace_sdk.x_vec.retrievers.retriever import Retriever
from xtrace_sdk.x_vec.utils.execution_context import ExecutionContext

__all__ = [
    "AWSKMSKeyProvider",
    "DataLoader",
    "Embedding",
    "ExecutionContext",
    "InferenceClient",
    "KeyProvider",
    "PassphraseKeyProvider",
    "Retriever",
]