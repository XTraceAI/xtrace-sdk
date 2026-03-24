.. XTrace Vec documentation master file, created by
   sphinx-quickstart on Fri Feb 14 20:28:30 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

XTrace SDK documentation
========================

What is XTrace?
---------------

Traditional vector databases require you to hand your data to a third party in plaintext. XTrace
is different: your document content is AES-encrypted and your embedding vectors are
homomorphically encrypted **before they leave your machine**. The server stores and searches
over ciphertexts — computing nearest-neighbor Hamming distances directly on encrypted
vectors — without ever seeing the underlying data. When results come back, you decrypt them
locally.

This means you get semantic search with the same privacy guarantees as if the data never
left your laptop.

The SDK has two modules: **x-vec** for encrypted vector search, and **x-mem** for
encrypted agent memory (coming soon). Both share the same credentials, execution context,
and XTrace integration.


Getting started
---------------

.. note::

   **Create a free account** at `app.xtrace.ai <https://app.xtrace.ai>`_ to get your
   API key and org ID. The free tier is rate-limited but fully functional.

Choose your path:

- **Python SDK** — :doc:`manual_rst/install` then :doc:`manual_rst/quickstart` for a
  concept-first walkthrough with full code examples.
- **CLI** — :doc:`manual_rst/install` then :doc:`manual_rst/cli_quickstart` for a
  terminal-first workflow that gets you querying in four commands.


.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   manual_rst/install
   manual_rst/quickstart

.. toctree::
   :maxdepth: 2
   :caption: x-vec

   manual_rst/embedding
   manual_rst/xtrace
   manual_rst/inference
   manual_rst/metadata

.. toctree::
   :maxdepth: 2
   :caption: x-mem

   manual_rst/x_mem

.. toctree::
   :maxdepth: 2
   :caption: CLI

   manual_rst/cli_quickstart
   manual_rst/cli_commands

.. toctree::
   :maxdepth: 2
   :caption: Reference

   manual_rst/configuration
   manual_rst/changelog
   manual_rst/contribute

.. toctree::
   :maxdepth: 1
   :caption: API Reference




Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
