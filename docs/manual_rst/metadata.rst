Metadata Filtering
==================

Metadata filters can be applied during nearest-neighbor search or used standalone via
``meta_search`` / ``meta_search_paginated``.


Privacy Notice
--------------

.. warning::

   Metadata fields (``tag1``–``tag5``, ``facets``) are **stored in plaintext** and are not
   encrypted by default. Chunk content and embedding vectors are end-to-end encrypted —
   metadata is the only part of a chunk that the XTrace server can read.

   If metadata privacy is a requirement, consider the following mitigation until native support
   is available:

   * **Store only opaque identifiers** in metadata tags (e.g. a hashed or randomly assigned
     ``user_id`` rather than a readable name), and keep the mapping in your own system.
   * **Restrict filters to equality checks** (``$eq`` / ``$in``) on those opaque values.
     Range operators (``$gt``, ``$lte``, ``$begins_with``, etc.) leak ordering information and
     should be avoided when the tag value itself is sensitive.

   XTrace plans to support encrypted metadata indexes natively in a future release.


Metadata Fields
---------------

Each chunk has five indexed scalar tags and one multi-value field:

.. list-table::
   :header-rows: 1

   * - Field
     - Type
     - Recommended Use
   * - ``tag1``
     - String
     - High-cardinality identifier (e.g. ``user_id``, ``org_id``)
   * - ``tag2``
     - String
     - Collection / project / knowledge base
   * - ``tag3``
     - Zero-padded number string
     - Numeric ranges (e.g. score, price, count)
   * - ``tag4``
     - ISO 8601 date string
     - Temporal ranges (e.g. ``created_at``)
   * - ``tag5``
     - String
     - Source or namespace
   * - ``facets``
     - List of strings
     - Labels, categories, ad-hoc metadata

.. warning::

   All tags are compared as **strings**. For correct range ordering, numeric values must be
   zero-padded to a fixed width and dates must use ISO 8601 UTC.

   .. code-block:: python

       # Numeric — zero-pad so "0000000010" > "0000000002"
       "tag3": "0000000010"

       # Date — ISO 8601 UTC
       "tag4": "2024-01-01T00:00:00Z"


Supported Operators
--------------------

Scalar tags (``tag1``–``tag5``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1

   * - Operator
     - Description
   * - ``"value"`` (raw string)
     - Exact match
   * - ``{"$eq": "value"}``
     - Exact match
   * - ``{"$ne": "value"}``
     - Not equal
   * - ``{"$gt": "v"}`` / ``{"$gte": "v"}``
     - Greater than / greater than or equal
   * - ``{"$lt": "v"}`` / ``{"$lte": "v"}``
     - Less than / less than or equal
   * - ``{"$gte": "a", "$lte": "b"}``
     - Range (between)
   * - ``{"$begins_with": "prefix"}``
     - Prefix match
   * - ``{"$in": ["a", "b"]}``
     - Value is in list
   * - ``{"$nin": ["a", "b"]}``
     - Value is not in list
   * - ``{"$contains": "substr"}``
     - Substring match
   * - ``{"$exists": True}``
     - Field is present and non-empty

``facets`` (multi-value)
^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1

   * - Operator
     - Description
   * - ``{"$subset": ["a", "b"]}``
     - Contains all provided tokens
   * - ``{"$any": ["a", "b"]}``
     - Contains at least one provided token
   * - ``{"$none": ["a", "b"]}``
     - Contains none of the provided tokens
   * - ``{"$contains": "token"}``
     - Contains the given token
   * - ``{"$size": N}``
     - Facets list has exactly N tokens


Query Examples
---------------

**Nearest-neighbor search with a filter:**

.. code-block:: python

    ids = await retriever.nn_search_for_ids(
        query_vector,
        k=5,
        kb_id="your_kb_id",
        meta_filter={"tag1": "org_772", "tag2": "invoices"},
    )

**Range filter on a numeric tag:**

.. code-block:: python

    meta_filter = {
        "tag1": "org_772",
        "tag3": {"$gt": "000000100000"},
    }

**Date range with facet refinement:**

.. code-block:: python

    meta_filter = {
        "tag4": {
            "$gte": "2024-01-01T00:00:00Z",
            "$lte": "2024-03-31T23:59:59Z",
        },
        "facets": {"$contains": "finance"},
    }

**Subset facet filter:**

.. code-block:: python

    meta_filter = {
        "tag1": "org_772",
        "facets": {"$subset": ["finance", "tax_audit"]},
    }

**Standalone metadata search:**

.. code-block:: python

    results = await xtrace.meta_search(
        kb_id="your_kb_id",
        meta_filter={"tag1": "org_772", "tag2": "invoices"},
        context_id=execution_context.id,
    )

**Paginated metadata search:**

.. code-block:: python

    page = await xtrace.meta_search_paginated(
        kb_id="your_kb_id",
        context_id=execution_context.id,
        meta_filter={"tag1": "org_772", "tag2": "invoices"},
        limit=20,
        offset=0,
        return_content=True,   # include encrypted chunk_content in results
    )
