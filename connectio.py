import uuid
import json
import logging
from pathlib import Path

from pymilvus import connections, Collection
from sentence_transformers import SentenceTransformer

from pymilvus import (
    FieldSchema, CollectionSchema, DataType,
    Collection, MilvusException, utility
)

class MilvusStorage:
    def create_schema(self):
        schema_fields = [
            FieldSchema(name='chunk_id', dtype=DataType.VARCHAR, is_primary=True, max_length=36),
            FieldSchema(name='document_id', dtype=DataType.VARCHAR, max_length=36),
            FieldSchema(name='chunk_order', dtype=DataType.INT32),
            FieldSchema(name='base_url', dtype=DataType.VARCHAR, max_length=512),
            FieldSchema(name='canonical_url', dtype=DataType.VARCHAR, max_length=512),
            FieldSchema(name='doc_last_modified', dtype=DataType.INT64),
            FieldSchema(name='content_type', dtype=DataType.VARCHAR, max_length=20),
            FieldSchema(name='content_source_type', dtype=DataType.VARCHAR, max_length=50),
            FieldSchema(name='language', dtype=DataType.VARCHAR, max_length=15),
            FieldSchema(name="chunk_text", dtype=DataType.VARCHAR, max_length=15000),
            FieldSchema(name="chunk_text_vector", dtype=DataType.FLOAT_VECTOR, dim=768),
            FieldSchema(name="doc_version", dtype=DataType.VARCHAR, max_length=5),
            FieldSchema(name='is_active', dtype=DataType.BOOL),
        ]
        return schema_fields

    def define_index(self, collection: Collection):
        collection.create_index(
            field_name="chunk_text_vector",
            index_params={
                "index_type": "HNSW",
                "metric_type": "COSINE",
                "params": {"M": 16, "efConstruction": 200}
            }
        )

    def create_collection(self, logger, collection_name: str):
        schema = self.create_schema()
        if utility.has_collection(collection_name):
            logger.info(f"Collection {collection_name} already exists")
            return Collection(name=collection_name)

        logger.info(f"Creating collection: {collection_name}")
        funds_schema = CollectionSchema(fields=schema, description="Markdown Chunks Schema")
        funds_collection = Collection(name=collection_name, schema=funds_schema)
        self.define_index(funds_collection)
        logger.info(f"Collection {collection_name} created successfully")
        return funds_collection

    def insert_chunks(self, collection: Collection, chunks):
        insert_data = [
            [c["chunk_id"] for c in chunks],
            [c["document_id"] for c in chunks],
            [c["chunk_order"] for c in chunks],
            [c["base_url"] for c in chunks],
            [c["canonical_url"] for c in chunks],
            [c["doc_last_modified"] for c in chunks],
            [c["content_type"] for c in chunks],
            [c["content_source_type"] for c in chunks],
            [c["language"] for c in chunks],
            [c["chunk_text"] for c in chunks],
            [c["chunk_text_vector"] for c in chunks],
            [c["doc_version"] for c in chunks],
            [c["is_active"] for c in chunks],
        ]
        return collection.insert(insert_data)

# --- Main pipeline ---
if __name__ == "__main__":
    # 1. Connect
    connections.connect("default", host="localhost", port="19530")

    # 2. Logger
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("milvus_logger")

    # 3. Init Milvus collection
    storage = MilvusStorage()
    collection_name = "markdown_chunks"
    collection = storage.create_collection(logger, collection_name)

    # 4. Load embedding model
    model = SentenceTransformer("all-mpnet-base-v2")  # 768-dim
    if model.get_sentence_embedding_dimension() != 768:
        raise ValueError("Schema requires 768-dim embeddings. Use a 768-dim model.")

    # 5. Pick one JSON file (adjust path)
    json_file = Path("data/chunks/seedfund.startupindia.gov.in_contact_chunks.json")
    with open(json_file, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    # 6. Compute embeddings and update dicts
    texts = [c["chunk_text"] for c in chunks]
    embeddings = model.encode(texts, convert_to_numpy=True).tolist()
    for c, emb in zip(chunks, embeddings):
        c["chunk_text_vector"] = emb

    # 7. Insert into Milvus
    res = storage.insert_chunks(collection, chunks)
    collection.flush()
    logger.info(f"Inserted {res.insert_count} chunks from {json_file}")

    # 8. Verify existence
    print(f"Collection {collection_name} now has {collection.num_entities} entities")
    collection.load()
    results = collection.query(
    expr="",
    output_fields=["chunk_id", "chunk_order", "chunk_text"],
    limit=10)
    for r in results:
        print(r)


