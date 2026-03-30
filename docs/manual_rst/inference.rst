LLM Inference
===============

``InferenceClient`` runs an LLM over retrieved context — useful for RAG pipelines where
you want a synthesized answer rather than raw chunk results.

Supported providers:

- **OpenAI**
- **Anthropic (Claude)** — via OpenAI-compatible API
- **Redpill** — private inference in TEE GPUs
- **Ollama** — fully local

For end-to-end privacy, run Ollama locally. If that is not feasible, Redpill provides
private inference via GPU Trusted Execution Environments (TEE). If inference privacy
is not a concern, OpenAI or Anthropic can be used.

The ``INFERENCE_API_KEY`` environment variable is read automatically when ``api_key`` is
not passed explicitly.


OpenAI
------

.. code-block:: python

   from xtrace_sdk.x_vec.inference.llm import InferenceClient

   inference = InferenceClient(inference_provider="openai", model_name="gpt-4o", api_key="your_api_key")
   inference.query("How many r's are in the word strawberry?")

For supported models, refer to the `OpenAI documentation <https://platform.openai.com/docs/models>`_.


Anthropic (Claude)
------------------

Uses Anthropic's OpenAI-compatible API endpoint.

.. code-block:: python

   from xtrace_sdk.x_vec.inference.llm import InferenceClient

   inference = InferenceClient(inference_provider="claude", model_name="claude-sonnet-4-6", api_key="your_api_key")
   inference.query("How many r's are in the word strawberry?")

For supported models, refer to the `Anthropic documentation <https://docs.anthropic.com/en/docs/about-claude/models>`_.


Redpill
-------

Redpill provides private inference with models running in TEE (Trusted Execution Environment)
GPUs, ensuring your queries remain secure during inference. This is an ideal middle ground when
you need privacy protection but cannot run models locally.

Key features:

- **Private inference**: Models run in TEE GPU environments
- **Unified API**: Access to 200+ AI models through a single API
- **Cost-effective**: Transparent per-token pricing

.. code-block:: python

   from xtrace_sdk.x_vec.inference.llm import InferenceClient

   inference = InferenceClient(inference_provider="redpill", model_name="deepseek/deepseek-v3-0324", api_key="your_api_key")
   inference.query("How many r's are in the word strawberry?")

Create an API key at https://redpill.ai/. For the full model list and pricing, see
https://docs.redpill.ai/.


Ollama
------

Ollama runs entirely locally, providing the strongest inference privacy guarantee.

.. code-block:: python

   from xtrace_sdk.x_vec.inference.llm import InferenceClient

   inference = InferenceClient(inference_provider="ollama", model_name="llama3.3", api_key="ollama")
   inference.query("How many r's are in the word strawberry?")

For Ollama setup instructions, see https://ollama.com/docs/installation.
