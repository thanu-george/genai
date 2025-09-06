from pymilvus import connections, utility

connections.connect("default", host="localhost", port="19530")

try:
    print("Milvus version:", utility.get_server_version())
    print(utility.list_collections())
except Exception as e:
    print("Error:", e)

