XTrace CLI Quick Start
=================================

This section walks you through setting up your environment, launching the interactive XTrace SDK CLI shell, and running the init workflow for a first time setup. The CLI is the quickest way to create an execution context, configure your embedding model, and save the required settings to a .env file so the SDK and tools work together out of the box.

Install dependencies
--------------------------------------------------
Create a virtual environment for sdk dependencies and install the CLI:

.. code-block:: bash

    python -m venv .venv
    source .venv/bin/activate
    pip install -e ".[cli]"

Start shell
--------------------------------------------------
Enter a CLI shell with

.. code-block:: bash

    xtrace

and view a list of available commands with

.. code-block::

    > help

Initialize SDK
--------------------------------------------------
Inside the CLI shell, run ``init`` to configure the SDK:

.. code-block::

    > init

The init command configures your local SDK repo in the following ways:

- connects your XTrace account via your ``API_KEY`` and ``ORG_ID``
- creates or loads a local ``Execution Context``, which is a unique fingerprint from which to interact with your stored data
- loads a local ``Embedding Model``, which embeds data for storage and retrieval in a vector space

You only need to run ``init`` once to persist your configuration, with information automatically being stored in a .env file and a ``/data`` directory.

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
| ``create-kb``     | Create a new knowledge base.                             |
+-------------------+----------------------------------------------------------+
| ``delete-kb``     | Delete one or more knowledge bases by ID.                |
+-------------------+----------------------------------------------------------+
| ``list-kbs``      | List knowledge bases available to your API key.          |
+-------------------+----------------------------------------------------------+
| ``describe-kb``   | Describe one or more knowledge bases by ID.              |
+-------------------+----------------------------------------------------------+

KB admin commands require ``ADMIN_KEY`` input, entered once per ``xtrace`` session.
To configure your repo with implicit ``ADMIN_KEY`` privileges, run ``init --admin``.

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
