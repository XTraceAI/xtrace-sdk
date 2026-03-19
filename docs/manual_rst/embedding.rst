Embedding Models
=================

``Embedding`` converts text into binary vectors for encrypted storage and search.
The ``embed_len`` dimension must match the value set on your ``ExecutionContext``.

Supported providers:

- **Ollama** — local, no API key required
- **Sentence Transformers** — local, models downloaded from Hugging Face (requires ``[embedding]`` extra)
- **OpenAI** — cloud-based

For end-to-end privacy, run Ollama or Sentence Transformers locally. OpenAI can be
used when privacy is not a concern.

The ``INFERENCE_API_KEY`` environment variable is read automatically when ``api_key`` is
not passed explicitly.


Ollama
------

.. code-block:: python

   from xtrace_sdk.x_vec.inference.embedding import Embedding

   embed = Embedding("ollama", "mxbai-embed-large", 1024)
   vector = await embed.bin_embed("some text")

For Ollama setup instructions, see https://ollama.com/docs/installation.


Sentence Transformers
---------------------

Models are downloaded from Hugging Face on first use.
See https://www.sbert.net/docs/pretrained_models.html for available models.
Requires ``pip install "xtrace-ai-sdk[embedding]"``.

.. code-block:: python

   from xtrace_sdk.x_vec.inference.embedding import Embedding

   embed = Embedding("sentence_transformer", "mixedbread-ai/mxbai-embed-large-v1", 512)
   vector = await embed.bin_embed("some text")


OpenAI
------

Set your OpenAI API key via the ``INFERENCE_API_KEY`` environment variable or pass it
directly as ``api_key``.

.. code-block:: python

   from xtrace_sdk.x_vec.inference.embedding import Embedding

   embed = Embedding("openai", "text-embedding-3-small", 1536)
   vector = await embed.bin_embed("some text")


Bring Your Own Vectors
-----------------------

If you already have float vectors from another source, convert them to the binary format
XTrace expects using ``Embedding.float_2_bin``. The length of the resulting list must match
``embed_len`` on your homomorphic client.

.. code-block:: python

   from xtrace_sdk.x_vec.inference.embedding import Embedding

   your_vector = [0.1, -0.2, 0.3, ...]   # list of floats, length = embed_len
   binary_vector = Embedding.float_2_bin(your_vector)
