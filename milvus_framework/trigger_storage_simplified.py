from pymilvus import connections, utility

# Connect to Milvus
connections.connect(alias="default", host="localhost", port="19530")

print("✅ Connected:", utility.get_server_version())
