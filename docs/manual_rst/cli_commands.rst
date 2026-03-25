CLI Command Reference
=================================

This section provides usage details for all CLI commands. See :doc:`cli_quickstart` for installation and first-time setup.

Commands are grouped by submodule and invoked as:

.. code-block:: bash

    xtrace <group> <command> [ARGS]       # subgroup command
    xtrace <command> [ARGS]               # top-level command

Inside the interactive shell (``xtrace shell``), omit the ``xtrace`` prefix:

.. code-block:: bash

    > xvec load /path/to/data/ KB_ID
    > kb create-kb my-kb

Shared commands
--------------------------------------------------

``init``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    xtrace init [--env-file {path/to/env-file}] [--admin] [--inference] [--help]

``init`` sets up your local SDK repo by connecting your XTrace credentials, creating or loading an ``Execution Context``, and loading an ``Embedding Model``. Must be run once before any data commands. ``--env-file`` / ``-f`` sets a custom path for the generated ``.env`` file (default: ``.env``). ``--admin`` saves your admin key to ``.env`` for implicit admin access. ``--inference`` saves an inference API key.

``version``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    xtrace version

Prints the installed SDK version.

``shell``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    xtrace shell

Starts an interactive CLI shell with tab completion and command history. Inside the shell, run commands without the ``xtrace`` prefix.

Knowledge base admin — ``xtrace kb``
--------------------------------------------------

KB commands require ``ADMIN_KEY`` input, entered once per session. To avoid repeated prompts, run ``init --admin`` to save the key to your ``.env``.

``create-kb``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    xtrace kb create-kb {NAME} [-d "{description}"] [-p {permission}] [--json] [-a {API_KEY}] [--help]

Creates a knowledge base named ``NAME``. Use ``-d`` for an optional description (wrap in quotes). Control access with ``-p {permission}``: ``read``, ``write``, ``delete``, or ``none`` (default: ``write``). The permission applies to the API key in your ``.env`` unless overridden with ``-a {API_KEY}``. ``--json`` returns the raw API response.

``delete-kb``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    xtrace kb delete-kb {KB_ID...} [--json] [--help]

Permanently deletes one or more space-separated knowledge bases. Prompts for confirmation. ``--json`` outputs raw JSON results.

``list-kbs``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    xtrace kb list-kbs [--all] [--json] [-a {api_key}] [--help]

Lists all knowledge bases accessible to your current API key. By default, only KBs with explicit permissions are shown. ``--all`` also shows KBs with no permissions (displayed as ``NONE``). ``--json`` returns raw JSON including numeric ``permissionLabel`` values. Override the API key with ``-a {api_key}``.

``describe-kb``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    xtrace kb describe-kb {KB_ID ...} [--json] [-a {api_key}] [--help]

Prints details for one or more knowledge bases. ``--json`` returns raw JSON (list when multiple IDs are given). Override the API key with ``-a {api_key}``.

x-vec commands — ``xtrace xvec``
--------------------------------------------------

``load``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    xtrace xvec load {/path/to/dir/} {KB_ID} [-f {file-types,...}] [--help]

Loads data from a directory into a knowledge base with id ``KB_ID``, processing files of type ``.txt``, ``.md``, ``.json``, and ``.csv``. To filter by file type, use ``-f`` with a comma-separated list (e.g. ``txt,json``).

``retrieve``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    xtrace xvec retrieve {KB_ID} {"query"} [-k {integer}] [--inference {provider} --model {"model"}] [--json] [-a {api_key}] [--help]

Returns the ``k`` most similar vectors to the query (default ``k=3``). ``--inference {provider} --model {"model"}`` runs an LLM over the retrieved context (requires an inference key from ``init --inference``). ``--json`` returns raw chunk data. Override the API key with ``-a {api_key}``.

``query``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    xtrace xvec query {KB_ID} "{query}" [-k {integer}] [--inference {provider} --model {"model"}] [--json] [-a {api_key}] [--help]

Alias of ``retrieve``. Same signature and behavior.

``head``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    xtrace xvec head {KB_ID} [--all] [--fullChunks] [--json] [-a {api_key}] [--help]

Previews vectors in a knowledge base. Shows up to 25 vectors with truncated content by default. ``--all`` shows every vector. ``--fullChunks`` disables content truncation. ``--json`` returns a raw JSON array. Override the API key with ``-a {api_key}``.

``fetch``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    xtrace xvec fetch {KB_ID} {VECTOR_ID...} [--fullChunks] [--json] [-a {api_key}] [--help]

Fetches one or more vectors by ID. ``--fullChunks`` disables content truncation. ``--json`` returns a raw JSON list of ``{id, content}`` objects. Override the API key with ``-a {api_key}``.

``upsert``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    xtrace xvec upsert {KB_ID} "{text}" [--help]

Inserts a single text chunk into a knowledge base. Wrap ``text`` in quotes if it contains spaces.

``upsert-file``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    xtrace xvec upsert-file {/path/to/file} {KB_ID} [--help]

Inserts and chunks the contents of a single file into a knowledge base. Supported types: ``.txt``, ``.md``, ``.json``, ``.csv``. For multiple files, use ``load``.

x-mem commands — ``xtrace xmem``
--------------------------------------------------

x-mem commands are coming soon.
