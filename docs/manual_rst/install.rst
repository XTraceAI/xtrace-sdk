============
Installation
============

Requires Python 3.11 or later.

Base install
------------

The base package includes the full x-vec SDK — encrypted vector storage, retrieval,
execution context management, and the Ollama and OpenAI embedding providers:

.. code-block:: bash

    pip install xtrace-ai-sdk


Optional extras
---------------

``[embedding]`` — Sentence Transformers
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Adds local embedding support via `Sentence Transformers <https://www.sbert.net>`_.
Models are downloaded from Hugging Face on first use.

Ollama and any OpenAI API-compatible embedding service (including OpenAI, Redpill,
and Anthropic) are supported out of the box in the base install — no extra flag needed.
Only install this extra if you want to use Sentence Transformers directly:

.. code-block:: bash

    pip install "xtrace-ai-sdk[embedding]"


``[cli]`` — Interactive CLI
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Adds the ``xtrace`` command-line interface — an interactive shell for managing
knowledge bases, loading data, and running queries without writing Python:

.. code-block:: bash

    pip install "xtrace-ai-sdk[cli]"

See :doc:`cli_quickstart` for setup and usage.


Combined
^^^^^^^^

Extras can be combined:

.. code-block:: bash

    pip install "xtrace-ai-sdk[embedding,cli]"


Install from source
-------------------

.. code-block:: bash

    git clone https://github.com/XTraceAI/xtrace-vec-sdk.git
    cd xtrace-vec-sdk
    pip install -e .

To include extras:

.. code-block:: bash

    pip install -e ".[embedding,cli]"
