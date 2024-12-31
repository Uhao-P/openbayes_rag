import pytest
import numpy as np
from unittest.mock import AsyncMock, Mock, patch
from openbayes_rag import llm


api_key = "sk-"
base_url = "https://api-store.xiaosuan.com/v1"

def test_get_openai_async_client_instance():
    with patch("openbayes_rag.llm.AsyncOpenAI") as mock_openai:
        mock_openai.return_value = "CLIENT"
        client = llm.get_openai_async_client_instance(api_key=api_key, base_url=base_url)
    assert client == "CLIENT"


@pytest.fixture
def mock_openai_client():
    with patch("openbayes_rag.llm.get_openai_async_client_instance") as mock_openai:
        mock_client = AsyncMock()
        mock_openai.return_value = mock_client
        yield mock_client



@pytest.mark.asyncio
async def test_openai_gpt4o(mock_openai_client):
    mock_response = AsyncMock()
    mock_response.choices = [Mock(message=Mock(content="1"))]
    messages = [{"role": "system", "content": "3"}, {"role": "user", "content": "2"}]
    mock_openai_client.chat.completions.create.return_value = mock_response

    response = await llm.gpt_4o_complete("2", system_prompt="3", api_key=api_key, base_url=base_url)

    mock_openai_client.chat.completions.create.assert_awaited_once_with(
        model="gpt-4o",
        messages=messages,
    )
    assert response == "1"


@pytest.mark.asyncio
async def test_openai_gpt4omini(mock_openai_client):
    mock_response = AsyncMock()
    mock_response.choices = [Mock(message=Mock(content="1"))]
    messages = [{"role": "system", "content": "3"}, {"role": "user", "content": "2"}]
    mock_openai_client.chat.completions.create.return_value = mock_response

    response = await llm.gpt_4o_mini_complete("2", system_prompt="3", api_key=api_key, base_url=base_url)

    mock_openai_client.chat.completions.create.assert_awaited_once_with(
        model="gpt-4o-mini",
        messages=messages,
    )
    assert response == "1"




@pytest.mark.asyncio
async def test_openai_embedding(mock_openai_client):
    mock_response = AsyncMock()
    mock_response.data = [Mock(embedding=[1, 1, 1])]
    texts = ["Hello world"]
    mock_openai_client.embeddings.create.return_value = mock_response

    response = await llm.openai_embedding(texts, api_key=api_key, base_url=base_url)

    mock_openai_client.embeddings.create.assert_awaited_once_with(
        model="text-embedding-3-small", input=texts, encoding_format="float"
    )
    # print(response)
    assert np.allclose(response, np.array([[1, 1, 1]]))

