# src/data_ai/storage/models.py
from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class DocumentStatus(str, Enum):
    PENDING = "pending"
    EMBEDDED = "embedded"
    CLUSTERED = "clustered"
    APPLIED = "applied"


class ClusterStatus(str, Enum):
    PROPOSED = "proposed"
    APPROVED = "approved"
    APPLIED = "applied"
    OUTLIER = "outlier"


class Document(BaseModel):
    id: str
    source_path: str
    file_type: str
    file_size: int
    summary: str
    status: DocumentStatus = DocumentStatus.PENDING
    cluster_id: Optional[str] = None
    vector: Optional[list[float]] = None
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)


class Cluster(BaseModel):
    id: str
    name: str
    doc_count: int
    variance: float
    centroid: list[float]
    status: ClusterStatus = ClusterStatus.PROPOSED
    parent_cluster: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
