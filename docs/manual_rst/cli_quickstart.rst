XTrace CLI Quick Start
=================================

This section walks you through setting up your environment and running the init workflow for a first time setup. The CLI is the quickest way to create an execution context, configure your embedding model, and save the required settings to a .env file so the SDK and tools work together out of the box.

Install dependencies
--------------------------------------------------
Create a virtual environment for sdk dependencies and install the CLI:

.. code-block:: bash

    python -m venv .venv
    source .venv/bin/activate
    pip install -e ".[cli]"

Using the CLI
--------------------------------------------------
Commands are invoked directly from the terminal:

.. code-block:: bash

    xtrace <command> [ARGS]

Run ``xtrace --help`` to view all available commands:

.. code-block:: bash

    xtrace --help

You can also start an interactive shell with tab completion and command history:

.. code-block:: bash

    xtrace shell

Inside the shell, omit the ``xtrace`` prefix (e.g. ``xvec load ...`` instead of ``xtrace xvec load ...``).

Initialize SDK
--------------------------------------------------
Run ``init`` to configure the SDK:

.. code-block:: bash

    xtrace init

The init command configures your local SDK repo in the following ways:

- connects your XTrace account via your ``API_KEY`` and ``ORG_ID``
- creates or loads a local ``Execution Context``, which is a unique fingerprint from which to interact with your stored data
- loads an ``Embedding Model``, which embeds data for storage and retrieval in a vector space

You only need to run ``init`` once to persist your configuration, with information automatically being stored in a .env file and a ``/data`` directory.

Key concepts
--------------------------------------------------

**Execution context** — your private cryptographic state. It holds a Paillier key pair
(for encrypting vectors) and an AES key (for encrypting content), all locked by a
passphrase you choose during ``init``. Every chunk you store and every query you run uses
this context. Losing the passphrase means losing the ability to decrypt your data.

**Embedding model** — converts text into binary vectors for encrypted storage and search.
The model you select during ``init`` must be the same one used for both uploading and
querying. Changing models later requires re-uploading your data.

**Knowledge base** — a namespace on XTrace where your encrypted chunks live. Create one
with ``xtrace kb create`` before loading data.

For a deeper look at how the encryption works, see the :doc:`quickstart` (Python SDK tutorial).

Your first query
--------------------------------------------------

After running ``init``, four commands take you from an empty knowledge base to search results:

.. code-block:: bash

    # 1. Create a knowledge base (note the KB ID in the output)
    xtrace kb create my-first-kb

    # 2. Load documents from a local folder
    xtrace xvec load ./my-docs/ <KB_ID>

    # 3. Search
    xtrace xvec retrieve <KB_ID> "your query here"

    # 4. (Optional) Search with LLM synthesis
    xtrace xvec retrieve <KB_ID> "your query" --inference openai --model gpt-4o

Command groups
--------------------------------------------------
Commands are organized by submodule. All subgroups and shared commands are accessible from the same ``xtrace`` CLI entry point.

**Shared**

+---------------+---------------------------------------------------------------+
| ``init``      | Initialize the SDK (credentials, execution context, model).   |
+---------------+---------------------------------------------------------------+
| ``version``   | Print the installed SDK version.                              |
+---------------+---------------------------------------------------------------+
| ``shell``     | Start an interactive CLI shell.                               |
+---------------+---------------------------------------------------------------+

**Knowledge base admin** — ``xtrace kb <command>``

+-------------------+----------------------------------------------------------+
| ``create``        | Create a new knowledge base.                             |
+-------------------+----------------------------------------------------------+
| ``delete``        | Delete one or more knowledge bases by ID.                |
+-------------------+----------------------------------------------------------+
| ``list``          | List knowledge bases available to your API key.          |
+-------------------+----------------------------------------------------------+
| ``describe``      | Describe one or more knowledge bases by ID.              |
+-------------------+----------------------------------------------------------+

KB admin commands require ``ADMIN_KEY`` input, entered once per shell session.
To save the key to your ``.env`` for implicit admin access, run ``xtrace init --admin``.

**x-vec** — ``xtrace xvec <command>``

+---------------------+--------------------------------------------------------+
| ``load``            | Load data from a folder into a knowledge base.         |
+---------------------+--------------------------------------------------------+
| ``retrieve``        | Retrieve from a knowledge base using a text query.     |
+---------------------+--------------------------------------------------------+
| ``query``           | Alias of ``retrieve``. Same signature and behavior.    |
+---------------------+--------------------------------------------------------+
| ``head``            | Preview vectors in a knowledge base.                   |
+---------------------+--------------------------------------------------------+
| ``fetch``           | Fetch specific vectors by ID from a knowledge base.    |
+---------------------+--------------------------------------------------------+
| ``upsert``          | Upsert a single text chunk into a knowledge base.      |
+---------------------+--------------------------------------------------------+
| ``upsert-file``     | Upsert chunks from a single file into a knowledge base.|
+---------------------+--------------------------------------------------------+

**x-mem** — ``xtrace xmem <command>``

x-mem commands are coming soon.

For full usage details for each command, run ``xtrace xvec <command> --help`` or see :doc:`cli_commands`.
