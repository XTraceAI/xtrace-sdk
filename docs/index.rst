.. XTrace Vec documentation master file, created by
   sphinx-quickstart on Fri Feb 14 20:28:30 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

XTrace SDK documentation
========================

XTrace is an encrypted vector and memory database. Your document content and embedding
vectors are encrypted on the client before storage; the server computes nearest-neighbor
search on ciphertexts and never sees the underlying data.

The SDK has two modules: **x-vec** for encrypted vector search, and **x-mem** for
encrypted agent memory (coming soon). Both share the same credentials, execution context,
and XTrace integration.

This documentation is organized for three audiences:

- **Potential users** — start with the `README <https://github.com/XTraceAI/xtrace-vec-sdk>`_
  for a brief description of what XTrace does.
- **New users** — start with :doc:`manual_rst/install` then
  :doc:`manual_rst/quickstart` for a concept-first walkthrough.
- **Experienced users** — jump to the API reference or the specific guide you need.


.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   manual_rst/install
   manual_rst/quickstart

.. toctree::
   :maxdepth: 2
   :caption: x-vec

   manual_rst/xtrace
   manual_rst/inference
   manual_rst/embedding
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
