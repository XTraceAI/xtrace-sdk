from typing import Callable

from openai import OpenAI


def _default_prompt_template(context: str, query: str) -> str:
    return (
        f"DOCUMENT:\n{context}\n"
        f"QUESTION:\n{query}\n"
        f"INSTRUCTIONS:\nAnswer the user's QUESTION using the DOCUMENT text above. "
        f"Keep your answer grounded in the facts of the DOCUMENT."
    )


class InferenceClient:
    """A client wrapper for interacting with inference models via OpenAI API."""

    def __init__(
        self,
        inference_provider: str,
        model_name: str,
        api_key: str | None = None,
        base_url: str | None = None,
        prompt_template: Callable[[str, str], str] | None = None,
    ) -> None:
        """Initialize the InferenceClient.

        :param inference_provider: The inference provider to use, or ``"custom"`` with a ``base_url``.
        :type inference_provider: str
        :param model_name: The name of the model to use.
        :type model_name: str
        :param api_key: The API key to use for authentication, defaults to None
        :type api_key: str, optional
        :param base_url: Custom OpenAI-compatible API base URL. Required when ``inference_provider`` is ``"custom"``.
        :type base_url: str, optional
        :param prompt_template: A function that takes a context and query and returns a prompt string, defaults to None
        :type prompt_template: Callable[[str, str], str], optional
        """

        URL_MAP: dict[str, str | None] = {
            "openai": "https://api.openai.com/v1",
            "ollama": "http://localhost:11434/v1",
            "custom": None,
        }

        if inference_provider not in URL_MAP:
            raise ValueError(f"Unsupported inference provider: {inference_provider}. Use 'custom' with a base_url for other providers.")

        resolved_url = base_url or URL_MAP[inference_provider]
        if resolved_url is None:
            raise ValueError("base_url is required when inference_provider is 'custom'")

        self.client = OpenAI(base_url=resolved_url, api_key=api_key)
        self.model_name = model_name
        self.prompt_template = prompt_template or _default_prompt_template

    def query(self, query: str, context: str | None = None, stream: bool = False) -> str:
        """Query the inference model.

        :param query: The query string to send to the model.
        :type query: str
        :param context: Optional context string to provide additional information to the model.
        :type context: str, optional
        :param stream: If True, stream the response incrementally.
        :type stream: bool, optional
        :return: The model's response text.
        :rtype: str
        """
        res = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": self.prompt_template(context, query) if context else query}],
            stream=stream,
        )

        if not stream:
            return res.choices[0].message.content or ""

        response = ""
        for chunk in res:
            if len(chunk.choices) == 0:
                continue
            if chunk.choices[0].delta.content is not None:
                response += chunk.choices[0].delta.content
        return response
