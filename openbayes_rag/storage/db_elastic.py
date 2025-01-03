import os
from elasticsearch import Elasticsearch, helpers
from dataclasses import dataclass

@dataclass
class ElasticDBTool:
    host: str = "localhost"
    port: int = 9200
    password: str = None
    index_settings: dict = None
    ca_path: str = None
    
    def __post_init__(self):
        self.client = Elasticsearch(
            [{'host': self.host, 'port': self.port, 'scheme': 'https'}],
            basic_auth = ('elastic', self.password),
            ca_certs= self.ca_path,
            verify_certs=True 
        )
        if self.index_settings is None:
            self.index_settings = {
                "settings": {
                    "number_of_shards": 1,
                    "number_of_replicas": 0
                },
                "mappings": {
                    "properties": {
                        "ent_id": {"type": "keyword"},
                        "chunk_id": {"type": "keyword"},
                        "ent": {"type": "text"},
                        "degree": {"type": "integer"},
                        "vector": {"type": "dense_vector", "dims": 128},
                        "rel_id": {"type": "keyword"},
                        "src_id": {"type": "keyword"},
                        "tg_id": {"type": "keyword"},
                        "rel": {"type": "text"},
                        "doc_id": {"type": "keyword"},
                        "chunk": {"type": "text"},
                        "doc_text": {"type": "text"}
                    }
                }
            }

    def create_index(self, index_name):
        self.client.indices.create(index=index_name, body=self.index_settings, ignore=400)

    
    def bulk_insert(self, index, documents):
        actions = [
            {
                "_index": index,
                "_id": doc["id"],
                "_source": doc
            }
            for doc in documents
        ]
        self.client.bulk(body=actions, index=index)


    def search(self, index_name, query, fields):
        body = {
            "query": {
                "multi_match": {
                    "query": query,
                    "fields": fields
                }
            }
        }
        return self.client.search(index=index_name, body=body)

# Example usage:
if __name__ == "__main__":
    tool = ElasticDBTool()
    tool.create_index("entities")
    tool.create_index("relationships")
    tool.create_index("chunks")
    tool.create_index("documents")

    # Example documents
    entities = [{"ent_id": "1", "chunk_id": ["1"], "ent": "Example Entity", "degree": 1, "vector": [0.1] * 128}]
    relationships = [{"rel_id": "1", "src_id": "1", "tg_id": "2", "chunk_id": ["1"], "rel": "Related", "vector": [0.2] * 128}]
    chunks = [{"chunk_id": "1", "doc_id": "1", "chunk": "Example chunk of text", "vector": [0.3] * 128}]
    documents = [{"doc_id": "1", "doc_text": "Example document text"}]

    tool.bulk_insert("entities", entities)
    tool.bulk_insert("relationships", relationships)
    tool.bulk_insert("chunks", chunks)
    tool.bulk_insert("documents", documents)

    # Search example
    print(tool.search("entities", "Example", ["ent"]))
