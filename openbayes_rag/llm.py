import numpy as np
from openai import (
    AsyncOpenAI,
    APIConnectionError,
    RateLimitError,
)
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
)


global_openai_async_client = None

def get_openai_async_client_instance(api_key=None, base_url=None):
    global global_openai_async_client
    if global_openai_async_client is None:
        if api_key is None or base_url is None:
            global_openai_async_client = AsyncOpenAI()
        else:
            global_openai_async_client = AsyncOpenAI(api_key=api_key, base_url=base_url)
    return global_openai_async_client



@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((RateLimitError, APIConnectionError)),
)
async def openai_complete_if_cache(
    model, prompt, system_prompt=None, history_messages=[], api_key=None, base_url=None, **kwargs
) -> str:
    openai_async_client = get_openai_async_client_instance(api_key=api_key, base_url=base_url)
    hashing_kv: BaseKVStorage = kwargs.pop("hashing_kv", None)
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})
    if hashing_kv is not None:
        args_hash = compute_args_hash(model, messages)
        if_cache_return = await hashing_kv.get_by_id(args_hash)
        if if_cache_return is not None:
            return if_cache_return["return"]

    response = await openai_async_client.chat.completions.create(
        model=model, messages=messages, **kwargs
    )

    if hashing_kv is not None:
        await hashing_kv.upsert(
            {args_hash: {"return": response.choices[0].message.content, "model": model}}
        )
        await hashing_kv.index_done_callback()
    return response.choices[0].message.content



async def gpt_4o_complete(
    prompt, system_prompt=None, history_messages=[], api_key=None, base_url=None, **kwargs
) -> str:
    return await openai_complete_if_cache(
        "gpt-4o",
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        api_key=api_key,
        base_url=base_url,
        **kwargs,
    )


async def gpt_4o_mini_complete(
    prompt, system_prompt=None, history_messages=[], api_key=None, base_url=None, **kwargs
) -> str:
    return await openai_complete_if_cache(
        "gpt-4o-mini",
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        api_key=api_key,
        base_url=base_url,
        **kwargs,
    )


@wrap_embedding_func_with_attrs(embedding_dim=1536, max_token_size=8192)
@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((RateLimitError, APIConnectionError)),
)
async def openai_embedding(texts: list[str], api_key=None, base_url=None) -> np.ndarray:
    openai_async_client = get_openai_async_client_instance(api_key=api_key, base_url=base_url)
    response = await openai_async_client.embeddings.create(
        model="text-embedding-3-small", input=texts, encoding_format="float"
    )
    return np.array([dp.embedding for dp in response.data])

