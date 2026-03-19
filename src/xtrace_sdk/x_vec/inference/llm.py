# mypy: ignore-errors
from typing import Any, Callable, Optional

from openai import OpenAI


class InferenceClient:
    """A client wrapper for interacting with inference models via OpenAI API
    """

    def __init__(self, inference_provider:str, model_name:str, api_key:str | None=None, prompt_template: Callable[[str, str], str] | None = None):
        """Initialize the InferenceClient.

        :param inference_provider: The inference provider to use.
        :type inference_provider: str
        :param model_name: The name of the model to use.
        :type model_name: str
        :param api_key: The API key to use for authentication, defaults to None
        :type api_key: str, optional
        :param prompt_template: A function that takes a context and query and returns a prompt string, defaults to None
        :type prompt_template: Optional[Callable[[str, str], str]], optional
        """

        URL_MAP = {
            "openai": "https://api.openai.com/v1",
            "redpill": "https://api.redpill.ai/v1",
            "claude": "https://api.anthropic.com/v1",
            "ollama": "http://localhost:11434/v1",
        } # Map of inference providers to their API URLs, Add more providers as needed

        if inference_provider not in URL_MAP:
            raise ValueError(f"Unsupported inference provider: {inference_provider}")

        self.client = OpenAI(base_url=URL_MAP[inference_provider], api_key=api_key)
        self.model_name = model_name

        if not prompt_template:
            self.prompt_template = lambda context,query: f"DOCUMENT:\n{context}\nQUESTION:\n{query}\nINSTRUCTIONS:\n Answer the users QUESTION using the DOCUMENT text above.Keep your answer ground in the facts of the DOCUMENT."

    def query(self, query: Any, context:Any | None=None,stream:bool=False, print_response:bool=True) -> Any:
        """Query the inference model.

        :param query: The query string to send to the model.
        :type query: str
        :param context: Optional context string to provide additional information to the model.
        :type context: str, optional
        """
        res = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": self.prompt_template(context, query) if context else query}],
            stream=stream,
        )

        if not stream:
            response = res.choices[0].message.content
            if print_response:
                print(response)
            return response

        response = ""

        for chunk in res:
            if len(chunk.choices) == 0:
                continue
            if chunk.choices[0].delta.content is not None:
                if print_response:
                    # Print the content of the chunk
                    print(chunk.choices[0].delta.content, end="")
                response += chunk.choices[0].delta.content
                # Optionally, you can return the content as a string or yield it
        return response
