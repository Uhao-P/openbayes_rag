import pytest
from unittest.mock import MagicMock, patch
from openbayes_rag.storage import ElasticDBTool  # Make sure to replace 'your_module_name' with the actual name of your module

@pytest.fixture
def elastic_tool():
    with patch('elasticsearch.Elasticsearch') as mock_es:
        # Mock the Elasticsearch client
        tool = ElasticDBTool(host='localhost', port=9200)
        tool.client = MagicMock()
        yield tool

def test_create_index(elastic_tool):
    elastic_tool.create_index("test_index")
    elastic_tool.client.indices.create.assert_called_once_with(index="test_index", body=elastic_tool.index_settings, ignore=400)

def test_bulk_insert(elastic_tool):
    documents = [{"id": "1", "text": "Hello"}]
    elastic_tool.bulk_insert("test_index", documents)
    expected_actions = [{"_index": "test_index", "_id": "1", "_source": {"id": "1", "text": "Hello"}}]
    elastic_tool.client.bulk.assert_called_once_with(body=expected_actions, index="test_index")

def test_search(elastic_tool):
    query = "Hello"
    fields = ["text"]
    elastic_tool.search("test_index", query, fields)
    expected_body = {
        "query": {
            "multi_match": {
                "query": query,
                "fields": fields
            }
        }
    }
    elastic_tool.client.search.assert_called_once_with(index="test_index", body=expected_body)

# Run the tests
if __name__ == "__main__":
    pytest.main()
