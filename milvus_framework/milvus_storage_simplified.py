import uuid
import logging
from logging import Logger

from pymilvus import (
    FieldSchema, CollectionSchema, DataType,
    Collection, MilvusException,
    connections, utility
)
from typing import List

# Change below import as per your project structure
# from vector_storage.storage_common.exception_handler import handle_milvus_exception

from typing import List, Dict


class MilvusStorage:
    def create_schema(self):
        schema_fields = [
            # Identification details
            FieldSchema(name='chunk_id', dtype=DataType.VARCHAR, description='Chunk Id', is_primary=True, max_length=36),
            FieldSchema(name='document_id', dtype=DataType.VARCHAR, description='Document Id', max_length=36),
            # Ordernal details ##base_url = https://seedfund.startupindia.gov.in,  canonical_url = https://seedfund.startupindia.gov.in/about.html
            FieldSchema(name='chunk_order', dtype=DataType.INT32, description='Chunk order number helps in reconstructing document sequence'),
            # Document tracing
            FieldSchema(name='base_url', dtype=DataType.VARCHAR, description='Base url of the source website', max_length=512),
            FieldSchema(name='canonical_url', dtype=DataType.VARCHAR, description='Current url of the document', max_length=512),
            # Temporal metadata
            FieldSchema(name='crawl_date', dtype=DataType.INT64, description='Date when the document was fetched'),
            FieldSchema(name='doc_last_modified', dtype=DataType.INT64, description='Document Last Modified'),
            # Content Classification
            FieldSchema(name='content_type', dtype=DataType.VARCHAR, description='Type of data whether it is text/image/mixed', max_length=20),
            FieldSchema(name='content_source_type', dtype=DataType.VARCHAR, description='Type of source e.g., webpage, PDF, announcement', max_length=50),
            # Language details
            FieldSchema(name='language', dtype=DataType.VARCHAR, description='Language of the document', max_length=15),
            # Content
            FieldSchema(name="chunk_text", dtype=DataType.VARCHAR, description='Chunk original text', max_length=15000),
            FieldSchema(name="chunk_text_vector", dtype=DataType.FLOAT_VECTOR, description='Chunk embedding vector', dim=768),  # Text embedding dim
            # Version control
            FieldSchema(name="doc_version", dtype=DataType.VARCHAR, max_length=5, description='Version of the document'),
            FieldSchema(name='is_active', dtype=DataType.BOOL, description='Flag to mark if the document is active')
        ]
        return schema_fields

    def define_index(self, collection: CollectionSchema):
        """
        For Vector Fields:
        chunk_text_vector → 1024 dim
        Recommended index type: HNSW or IVF_FLAT
        HNSW → great for high recall, stable for large collections
        IVF_FLAT → good for large-scale datasets with filtering
        
        Indexes for Categorical fields
        Inverted index is generally best for these fields in Milvus and similar vector databases, as it is optimized for fast filtering on categorical/metadata fields.
        Hash index can also work well for very low-cardinality fields (like is_active), but offers no major advantage over inverted for your use case.
        Binary tree is rarely used for categorical fields in modern vector databases; it’s more relevant for range/ordered queries on numeric data.
        """
        collection.create_index(
            field_name="chunk_text_vector",
            index_params={
                "index_type": "HNSW",
                "metric_type": "COSINE",
                "params": {"M": 16, "efConstruction": 200}
            }
        )
        for date_field in ['crawl_date', 'doc_last_modified']:
            collection.create_index(
                field_name=date_field,
                index_params={'index_type': 'INVERTED'}
            )

        for schema_field in ['language', 'is_active']:
            collection.create_index(
                field_name=schema_field,
                index_type="INVERTED"
            )
        
    def create_collection(
            self,
            logger: Logger,
            collection_name: str
        ):
        """
        Create collection schema
        Create collection
        """
        try:
            schema = self.create_schema()
            """Create collection with schema and indexes"""
            if utility.has_collection(collection_name):
                logger.info(f"Collection {collection_name} already exists")
                funds_collection = Collection(name=collection_name)
                return funds_collection
            logger.info(f"Started collection: {collection_name} creation")
            funds_schema = CollectionSchema(
                fields=schema,
                description="List of Schema Fields"
            )
            funds_schema.verify()
            funds_collection = Collection(
                name=collection_name,
                schema=funds_schema
            )
            self.define_index(funds_collection)
            logger.info(f"Collection: {collection_name} created successfully.")
            return funds_collection
        except MilvusException as e:  # Catch specific exceptions if possible
            handle_milvus_exception(e=e, logger=logger)
        except Exception as e:  # Fallback for generic errors
            handle_milvus_exception(e=e, logger=logger)


class MilvusClient:
    def __init__(
            self,
            logger: Logger,
            host: str = "localhost",
            port: str = "19530",
            collection_name: str = "funds_collection"
        ):
        self.host = host
        self.port = port
        self.logger = logger
        connections.connect("default", host=host, port=port)
        # Below code is for dropping the Collection
        # Collection(collection_name).drop()
        self.collection = MilvusStorage().create_collection(
            collection_name=collection_name,
            logger=logger
        )
        self.collection.load()
    
    def get_collection(self):
        return self.collection

    def insert_nodes(
            self,
            logger: Logger,
            nodes: List[Dict]
        ):
        """Insert chunk nodes into Milvus collection"""
        if not self.collection:
            raise ValueError("Collection not initialized")
        
        # Prepare data in column format
        data_dict = {
            "chunk_id": [],
            "document_id": [],
            "chunk_order": [],
            "base_url": [],
            "canonical_url": [],
            "crawl_date": [],
            "doc_last_modified": [],
            "content_type": [],
            "content_source_type": [],
            "language": [],
            "chunk_text": [],
            "chunk_text_vector": [],
            "doc_version": [],
            "is_active": []
        }

        # Validate and prepare nodes
        
        for node in nodes:
            metadata = node.metadata
            if "chunk_text_vector" not in metadata:
                raise ValueError("Node missing embeddings in 'chunk_text_vector' field")
            
            # Required fields validation
            required_fields = [
                'document_id',
                'chunk_id', 'chunk_text', 'chunk_order',
                'lang', 'is_active',
                'crawl_date', 'doc_last_modified'
            ]
            for field in required_fields:
                if field not in metadata:
                    raise ValueError(f"Missing required field: {field}")
            # Populate data dict
            # print(f"metadata: {node.metadata}")
            for field in data_dict.keys():
                data_dict[field].append(metadata.get(field))

        # Convert to list of columns in schema order
        logger.info("Arranging nodes data as defined in schema.")
        insert_data = [data_dict[field.name] for field in self.collection.schema.fields]
        insert_data_rem_null = [
                [v if v is not None else "" for v in sublist]
                for sublist in insert_data
            ]
        # Insert into Milvus
        try:
            logger.info("Data insertion started.")
            res = self.collection.upsert(insert_data_rem_null)
            logger.info(f"Successfully inserted {len(nodes)} chunks")
            return res
        except Exception as e:
            logger.error(f"Insert failed: {str(e)}")
            raise

    # Method for getting search result
    def search(self, query_embeddings: List[float], top_k: int = 5, filters: str = None):
        """Search similar chunks"""
        search_params = {
            "metric_type": "COSINE",
            "offset": 0,
            "ignore_growing": False,
            "params": {"nprobe": 10}
        }

        return self.collection.search(
            data=[query_embeddings],
            anns_field="chunk_text_vector",
            param=search_params,
            limit=top_k,
            expr=filters,
            output_fields=["chunk_text"]
        )
