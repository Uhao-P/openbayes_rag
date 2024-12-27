import os
from functools import lru_cache
from typing import List, Dict, Callable, Any

import numpy as np
from openai import (
    AsyncOpenAI,
    APIConnectionError,
    RateLimitError,
    Timeout,
)
from pydantic import BaseModel, Field
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from .base import BaseKVStorage
from .utils import (
    compute_args_hash,
    wrap_embedding_func_with_attrs,
    quantize_embedding,
    get_best_cached_response,
)



os.environ["TOKENIZERS_PARALLELISM"] = "false"


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((RateLimitError, APIConnectionError, Timeout)),
)
async def openai_complete_if_cache(
    model,
    prompt,
    system_prompt=None,
    history_messages=[],
    base_url=None,
    api_key=None,
    **kwargs,
) -> str:
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key

    openai_async_client = (
        AsyncOpenAI() if base_url is None else AsyncOpenAI(base_url=base_url)
    )
    hashing_kv: BaseKVStorage = kwargs.pop("hashing_kv", None)
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})

    if hashing_kv is not None:
        # Calculate args_hash only when using cache
        args_hash = compute_args_hash(model, messages)

        # Get embedding cache configuration
        embedding_cache_config = hashing_kv.global_config.get(
            "embedding_cache_config", {"enabled": False, "similarity_threshold": 0.95}
        )
        is_embedding_cache_enabled = embedding_cache_config["enabled"]
        if is_embedding_cache_enabled:
            # Use embedding cache
            embedding_model_func = hashing_kv.global_config["embedding_func"]["func"]
            current_embedding = await embedding_model_func([prompt])
            quantized, min_val, max_val = quantize_embedding(current_embedding[0])
            best_cached_response = await get_best_cached_response(
                hashing_kv,
                current_embedding[0],
                similarity_threshold=embedding_cache_config["similarity_threshold"],
            )
            if best_cached_response is not None:
                return best_cached_response
        else:
            # Use regular cache
            if_cache_return = await hashing_kv.get_by_id(args_hash)
            if if_cache_return is not None:
                return if_cache_return["return"]

    if "response_format" in kwargs:
        response = await openai_async_client.beta.chat.completions.parse(
            model=model, messages=messages, **kwargs
        )
    else:
        response = await openai_async_client.chat.completions.create(
            model=model, messages=messages, **kwargs
        )
    content = response.choices[0].message.content
    if r"\u" in content:
        content = content.encode("utf-8").decode("unicode_escape")

    if hashing_kv is not None:
        await hashing_kv.upsert(
            {
                args_hash: {
                    "return": content,
                    "model": model,
                    "embedding": quantized.tobytes().hex()
                    if is_embedding_cache_enabled
                    else None,
                    "embedding_shape": quantized.shape
                    if is_embedding_cache_enabled
                    else None,
                    "embedding_min": min_val if is_embedding_cache_enabled else None,
                    "embedding_max": max_val if is_embedding_cache_enabled else None,
                    "original_prompt": prompt,
                }
            }
        )
    return content



@wrap_embedding_func_with_attrs(embedding_dim=1536, max_token_size=8192)
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=60),
    retry=retry_if_exception_type((RateLimitError, APIConnectionError, Timeout)),
)
async def openai_embedding(
    texts: list[str],
    model: str = "text-embedding-3-small",
    base_url: str = None,
    api_key: str = None,
) -> np.ndarray:
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key

    openai_async_client = (
        AsyncOpenAI() if base_url is None else AsyncOpenAI(base_url=base_url)
    )
    response = await openai_async_client.embeddings.create(
        model=model, input=texts, encoding_format="float"
    )
    return np.array([dp.embedding for dp in response.data])



class Model(BaseModel):
    """
    This is a Pydantic model class named 'Model' that is used to define a custom language model.

    Attributes:
        gen_func (Callable[[Any], str]): A callable function that generates the response from the language model.
            The function should take any argument and return a string.
        kwargs (Dict[str, Any]): A dictionary that contains the arguments to pass to the callable function.
            This could include parameters such as the model name, API key, etc.

    Example usage:
        Model(gen_func=openai_complete_if_cache, kwargs={"model": "gpt-4", "api_key": os.environ["OPENAI_API_KEY_1"]})

    In this example, 'openai_complete_if_cache' is the callable function that generates the response from the OpenAI model.
    The 'kwargs' dictionary contains the model name and API key to be passed to the function.
    """

    gen_func: Callable[[Any], str] = Field(
        ...,
        description="A function that generates the response from the llm. The response must be a string",
    )
    kwargs: Dict[str, Any] = Field(
        ...,
        description="The arguments to pass to the callable function. Eg. the api key, model name, etc",
    )

    class Config:
        arbitrary_types_allowed = True