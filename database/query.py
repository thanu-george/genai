from pymilvus import connections, Collection
from sentence_transformers import SentenceTransformer

def semantic_search(collection_name, query_text, top_k=5):
    # connect
    connections.connect("default", host="localhost", port="19530")

    # load collection
    collection = Collection(collection_name)
    collection.load()

    # same embedding model
    model = SentenceTransformer("all-mpnet-base-v2")

    # encode query
    query_vector = model.encode(query_text).tolist()

    # search
    results = collection.search(
        data=[query_vector],               # must be a list of vectors
        anns_field="chunk_text_vector",    # field we indexed
        param={"metric_type": "COSINE", "params": {"ef": 128}},
        limit=top_k,
        output_fields=["chunk_id", "document_id", "chunk_order", "chunk_text"]
    )

    # pretty print
    for i, hit in enumerate(results[0]):
        print(f"\nResult {i+1} (score={hit.distance:.4f})")
        print("Document:", hit.entity.get("document_id"))
        print("Chunk ID:", hit.entity.get("chunk_id"))
        print("Chunk text:", hit.entity.get("chunk_text")[:300], "...")

if __name__ == "__main__":
    semantic_search("markdown_chunks", "partnered services", top_k=3)
