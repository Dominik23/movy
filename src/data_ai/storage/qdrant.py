# src/data_ai/storage/qdrant.py
from typing import Optional
from uuid import uuid4

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    VectorParams,
    Filter,
    FieldCondition,
    MatchValue,
)

from data_ai.storage.models import Document, Cluster, DocumentStatus, ClusterStatus

VECTOR_SIZE = 768  # nomic-embed-text dimension


class QdrantStore:
    def __init__(self, url: str = "localhost:6333", prefix: str = "data_ai"):
        self.client = QdrantClient(url=url)
        self.docs_collection = f"{prefix}_documents"
        self.clusters_collection = f"{prefix}_clusters"
        self._ensure_collections()

    def _ensure_collections(self) -> None:
        if not self.client.collection_exists(self.docs_collection):
            self.client.create_collection(
                collection_name=self.docs_collection,
                vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
            )

        if not self.client.collection_exists(self.clusters_collection):
            self.client.create_collection(
                collection_name=self.clusters_collection,
                vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
            )

    def upsert_document(self, doc: Document) -> None:
        vector = doc.vector if doc.vector else [0.0] * VECTOR_SIZE
        self.client.upsert(
            collection_name=self.docs_collection,
            points=[
                PointStruct(
                    id=doc.id,
                    vector=vector,
                    payload={
                        "source_path": doc.source_path,
                        "file_type": doc.file_type,
                        "file_size": doc.file_size,
                        "summary": doc.summary,
                        "status": doc.status.value,
                        "cluster_id": doc.cluster_id,
                        "created_at": doc.created_at.isoformat(),
                        "updated_at": doc.updated_at.isoformat(),
                    },
                )
            ],
        )

    def upsert_cluster(self, cluster: Cluster) -> None:
        # Use zero vector for clusters without centroids (e.g., outlier clusters)
        centroid = cluster.centroid if cluster.centroid else [0.0] * VECTOR_SIZE
        self.client.upsert(
            collection_name=self.clusters_collection,
            points=[
                PointStruct(
                    id=cluster.id,
                    vector=centroid,
                    payload={
                        "name": cluster.name,
                        "doc_count": cluster.doc_count,
                        "variance": cluster.variance,
                        "status": cluster.status.value,
                        "parent_cluster": cluster.parent_cluster,
                        "created_at": cluster.created_at.isoformat(),
                        "updated_at": cluster.updated_at.isoformat(),
                    },
                )
            ],
        )

    def get_documents_by_status(self, status: DocumentStatus) -> list[Document]:
        results = self.client.scroll(
            collection_name=self.docs_collection,
            scroll_filter=Filter(
                must=[FieldCondition(key="status", match=MatchValue(value=status.value))]
            ),
            with_vectors=True,
            limit=10000,
        )

        documents = []
        for point in results[0]:
            documents.append(Document(
                id=point.id,
                vector=point.vector,
                **point.payload,
            ))
        return documents

    def get_all_documents(self) -> list[Document]:
        results = self.client.scroll(
            collection_name=self.docs_collection,
            with_vectors=True,
            limit=10000,
        )

        documents = []
        for point in results[0]:
            documents.append(Document(
                id=point.id,
                vector=point.vector,
                **point.payload,
            ))
        return documents

    def get_all_clusters(self) -> list[Cluster]:
        results = self.client.scroll(
            collection_name=self.clusters_collection,
            with_vectors=True,
            limit=1000,
        )

        clusters = []
        for point in results[0]:
            clusters.append(Cluster(
                id=point.id,
                centroid=point.vector,
                **point.payload,
            ))
        return clusters

    def get_documents_by_cluster(self, cluster_id: str) -> list[Document]:
        results = self.client.scroll(
            collection_name=self.docs_collection,
            scroll_filter=Filter(
                must=[FieldCondition(key="cluster_id", match=MatchValue(value=cluster_id))]
            ),
            with_vectors=True,
            limit=10000,
        )

        documents = []
        for point in results[0]:
            documents.append(Document(
                id=point.id,
                vector=point.vector,
                **point.payload,
            ))
        return documents

    def update_document_status(self, doc_id: str, status: DocumentStatus) -> None:
        self.client.set_payload(
            collection_name=self.docs_collection,
            payload={"status": status.value},
            points=[doc_id],
        )

    def update_document_cluster(self, doc_id: str, cluster_id: str) -> None:
        self.client.set_payload(
            collection_name=self.docs_collection,
            payload={"cluster_id": cluster_id, "status": DocumentStatus.CLUSTERED.value},
            points=[doc_id],
        )

    def update_cluster_status(self, cluster_id: str, status: ClusterStatus) -> None:
        self.client.set_payload(
            collection_name=self.clusters_collection,
            payload={"status": status.value},
            points=[cluster_id],
        )

    def update_cluster_name(self, cluster_id: str, name: str) -> None:
        self.client.set_payload(
            collection_name=self.clusters_collection,
            payload={"name": name},
            points=[cluster_id],
        )

    def delete_all_clusters(self) -> None:
        clusters = self.get_all_clusters()
        if clusters:
            self.client.delete(
                collection_name=self.clusters_collection,
                points_selector=[c.id for c in clusters],
            )

        # Reset cluster assignments on documents
        docs = self.get_all_documents()
        for doc in docs:
            if doc.cluster_id:
                self.client.set_payload(
                    collection_name=self.docs_collection,
                    payload={"cluster_id": None, "status": DocumentStatus.EMBEDDED.value},
                    points=[doc.id],
                )

    def reset(self) -> None:
        self.client.delete_collection(self.docs_collection)
        self.client.delete_collection(self.clusters_collection)
        self._ensure_collections()

    def generate_id(self) -> str:
        return str(uuid4())
