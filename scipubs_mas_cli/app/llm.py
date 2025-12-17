from __future__ import annotations

from functools import lru_cache

from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage

from .config import BASE_URL, API_KEY, MODEL_NAME


@lru_cache(maxsize=8)
def get_chat_model(temperature: float = 0.1) -> ChatOpenAI:
    """Return (and cache) a ChatOpenAI instance for the given temperature.

    Note: We cache a small number of model instances because creating a new
    client object on every LLM call is needless overhead. Temperatures in this
    project are effectively a small discrete set (e.g., 0.1/0.2/0.3).
    """
    return ChatOpenAI(
        base_url=BASE_URL,
        api_key=API_KEY,
        model=MODEL_NAME,
        temperature=temperature,
    )


def call_llm(messages: list[BaseMessage], temperature: float = 0.1) -> str:
    """Utility to call the chat model and get string content."""
    llm = get_chat_model(temperature=temperature)
    response = llm.invoke(messages)
    # For standard OpenAI-compatible models, content is just a string
    return str(response.content)
