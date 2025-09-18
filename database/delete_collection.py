from pymilvus import connections, Collection, utility
connections.connect("default", host="localhost", port="19530")
collection_name = "markdown_chunks"

if utility.has_collection(collection_name):
    utility.drop_collection(collection_name)
    print(f"Collection '{collection_name}' deleted successfully.")
else:
    print(f"Collection '{collection_name}' does not exist.")
