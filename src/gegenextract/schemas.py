from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from datetime import datetime, timezone


class LoggingConfig(BaseModel):
    level: str = "INFO"
    file: Optional[str] = None


class LLMConfig(BaseModel):
    provider: str
    model: str
    temperature: float = 0.0
    extra: Dict[str, Any] = Field(default_factory=dict)


class PersistenceConfig(BaseModel):
    database_url: str


class AppConfig(BaseModel):
    app: Dict[str, Any]
    logging: LoggingConfig
    data: Dict[str, Any]
    llm: LLMConfig
    persistence: PersistenceConfig
    experiments: Optional[Dict[str, Any]] = None
    ingestion: Optional[Dict[str, Any]] = None


class Document(BaseModel):
    id: str
    path: str
    text: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class Page(BaseModel):
    index: int
    text: Optional[str] = None
    ocr: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None


class ExtractionTarget(BaseModel):
    id: str
    fields: Dict[str, Any]


class DatasetSample(BaseModel):
    id: str
    document: Document
    pages: List[Page]
    target: Optional[ExtractionTarget] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class DatasetSplit(BaseModel):
    train: List[str]
    val: List[str]
    test: List[str]


class ExtractionResult(BaseModel):
    document_id: str
    extracted: Dict[str, Any]
    confidence: Optional[float] = None
    raw: Optional[str] = None


class PromptVersion(BaseModel):
    id: str
    template: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = Field(default_factory=dict)


class LLMCallRecord(BaseModel):
    id: str
    model: str
    prompt: str
    response: Optional[str]
    tokens: Optional[int]
    elapsed_seconds: Optional[float]
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ExperimentRun(BaseModel):
    id: str
    experiment_id: str
    config: Dict[str, Any]
    start_time: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    end_time: Optional[datetime] = None
    metrics: Dict[str, float] = Field(default_factory=dict)
    tags: List[str] = Field(default_factory=list)
