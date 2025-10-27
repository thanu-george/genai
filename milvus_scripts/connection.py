import uuid
import logging
from pathlib import Path

from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from pymilvus import connections, Collection, utility
from pymilvus import FieldSchema, CollectionSchema, DataType


def recursive_chunk(text, chunk_size=1000, chunk_overlap=200):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )
    return splitter.split_text(text)


def load_and_chunk(input_dir: Path):
    all_chunks = []
    count = 1
    for file_path in input_dir.glob("*.md"):
        with open(file_path, "r", encoding="utf-8",errors="replace") as f:
            raw_text = f.read()

        text_chunks = recursive_chunk(raw_text)

        for i, chunk_text in enumerate(text_chunks):

            all_chunks.append({
                "chunk_id": str(uuid.uuid4()),
                "document_id": file_path.stem,
                "chunk_order": i,
                "base_url": "",
                "canonical_url": str(file_path),
                "doc_last_modified": int(file_path.stat().st_mtime),
                "content_type": "text",
                "content_source_type": "markdown",
                "language": "en",
                "chunk_text": chunk_text,
                "chunk_text_vector": [],  # placeholder
                "doc_version": "v1",
                "is_active": True,
            })

        print(f"[{count}] {file_path.name} → {len(text_chunks)} chunks")
        count += 1

    return all_chunks


# ------------------ Milvus ------------------
class MilvusStorage:
    def __init__(self, host="localhost", port="19530"):
        connections.connect("default", host=host, port=port)
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger("milvus_logger")

    def create_schema(self):
        return [
            FieldSchema(name='chunk_id', dtype=DataType.VARCHAR, is_primary=True, max_length=100),
            FieldSchema(name='document_id', dtype=DataType.VARCHAR, max_length=512),
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

    def define_index(self, collection: Collection):
        collection.create_index(
            field_name="chunk_text_vector",
            index_params={
                "index_type": "HNSW",
                "metric_type": "COSINE",
                "params": {"M": 40, "efConstruction": 400}
            }
        )

    def create_collection(self, logger, collection_name: str):
        schema = CollectionSchema(fields=self.create_schema(), description="Markdown Chunks Schema")
        if utility.has_collection(collection_name):
            logger.info(f"Collection {collection_name} already exists")
            return Collection(name=collection_name)

        logger.info(f"Creating collection: {collection_name}")
        collection = Collection(name=collection_name, schema=schema)
        self.define_index(collection)
        logger.info(f"Collection {collection_name} created successfully")
        return collection

    def insert_chunks(self, collection: Collection, chunks):
        rows = []
        for c in chunks:
            rows.append({
            "chunk_id": str(c.get("chunk_id", f"{c['document_id']}|{c['chunk_order']}")),
            "document_id": c.get("document_id", ""),
            "chunk_order": int(c.get("chunk_order", 0)),
            "base_url": c.get("base_url", ""),
            "canonical_url": c.get("canonical_url", ""),
            "doc_last_modified": int(c.get("doc_last_modified", 0)),
            "content_type": c.get("content_type", ""),
            "content_source_type": c.get("content_source_type", ""),
            "language": c.get("language", ""),
            "chunk_text": c.get("chunk_text", ""),
            "chunk_text_vector": c.get("chunk_text_vector", [0.0] * 768),
            "doc_version": c.get("doc_version", "v1"),
            "is_active": bool(c.get("is_active", True)),
        })

        return collection.insert(rows)



# ------------------ Main Ingestion ------------------
if __name__ == "__main__":
    # Connect
    #connections.connect("default", host="localhost", port="19530")

    # Logger
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("milvus_logger")

    # Milvus collection
    storage = MilvusStorage()
    collection_name = "markdown_chunks"
    collection = storage.create_collection(logger, collection_name)

    # Embedding model
    model = SentenceTransformer("all-mpnet-base-v2")
    if model.get_sentence_embedding_dimension() != 768:
        raise ValueError("Schema requires 768-dim embeddings")

    # Load + Chunk directly from markdown
    input_dir = Path(__file__).parent.parent / "data" / "second_clean"
    chunks = load_and_chunk(input_dir)

    # Compute embeddings
    for c in chunks:
        text = c.get("chunk_text", "")
        if text.strip():
            c["chunk_text_vector"] = model.encode(text).tolist()
        else:
            c["chunk_text_vector"] = [0.0] * 768

    # Insert into Milvus
    res = storage.insert_chunks(collection, chunks)
    collection.flush()
    logger.info(f"Inserted {len(chunks)} chunks from {input_dir}")

    # Verify
    collection.load()
    print(f"Collection {collection_name} now has {collection.num_entities} entities")
    results = collection.query(
        expr="",
        output_fields=["chunk_id", "chunk_order", "chunk_text"],
        limit=5
    )
    for r in results:
        print(r)
