"""Pydantic models used for validating LLM output and API requests/responses."""
from typing import List, Literal, Optional

from pydantic import BaseModel, Field

from app import config


class AssistantAnswer(BaseModel):
    """The JSON structure we ask the LLM to return."""
    answer: str
    sources: List[str] = []
    confidence: Literal["high", "medium", "low"] = "medium"
    follow_up_questions: List[str] = []


class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    history: List[Message] = []
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(None, gt=0.0, le=1.0)
    use_cache: bool = True


class ChatResponse(BaseModel):
    answer: str
    sources: List[str] = []
    confidence: str = "medium"
    follow_up_questions: List[str] = []
    model_used: Optional[str] = None
    tools_used: List[str] = []
    cached: bool = False
    degraded: bool = False  # true when we could not use the LLM and returned a fallback answer
    latency_ms: int = 0
    total_tokens: int = 0


class BatchRequest(BaseModel):
    questions: List[str] = Field(..., min_length=1, max_length=config.MAX_BATCH_SIZE)
    use_cache: bool = True


class BatchResponse(BaseModel):
    results: List[ChatResponse]
    total_latency_ms: int


# ---------------- W16: agentic (cross-source verification) ----------------
class AgenticChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    max_iterations: Optional[int] = Field(None, ge=1, le=10)


class AgenticChatResponse(BaseModel):
    answer: str
    sources: List[str] = []
    confidence: str = "medium"
    verification: str = "not_verified"  # sufficient | insufficient | conflicting | max_iterations_reached | ...
    iterations: int = 0
    trajectory: List[dict] = []
    evidence_log: List[dict] = []
    total_tokens: int = 0
    model_used: List[str] = []
    degraded: bool = False
    latency_ms: int = 0
