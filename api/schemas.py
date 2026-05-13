"""
api/schemas.py — Pydantic models for request/response validation.

These enforce the API contract with Team 1.
If a field is missing or wrong type, FastAPI returns a 422 automatically.
"""

from pydantic import BaseModel, Field
from typing import Optional


# ═══════════════════════════════════════════════════════════════════
# Batch Submit (Flow 1) — POST /api/v1/batches/submit
# ═══════════════════════════════════════════════════════════════════

class AuthorSchema(BaseModel):
    """Author metadata from Team 1's data collection."""
    author_id: str
    username: str
    account_created_at: Optional[str] = None
    verified: bool = False
    verification_type: Optional[str] = None
    follower_count: int = 0
    following_count: int = 0
    post_count: int = 0
    profile_picture_present: bool = False
    bio_present: bool = False
    bio_text: Optional[str] = None
    last_active_at: Optional[str] = None
    karma: Optional[int] = None               # Reddit only
    account_protected: bool = False


class EngagementSchema(BaseModel):
    """Engagement metrics for a post/comment."""
    likes: int = 0
    replies: int = 0
    shares: int = 0
    views: int = 0


class BatchItemSchema(BaseModel):
    """A single post/comment in a batch."""
    item_id: str
    platform: str                              # youtube|twitter|reddit|external
    content: str
    posted_at: str                             # ISO8601
    collected_at: str                          # ISO8601
    source_url: Optional[str] = None
    author: Optional[AuthorSchema] = None      # null for web crawl sources
    engagement: EngagementSchema = EngagementSchema()


class BatchSubmitRequest(BaseModel):
    """POST /api/v1/batches/submit request body."""
    batch_id: str
    submitted_at: str                          # ISO8601
    keyword: str
    items: list[BatchItemSchema]


class BatchSubmitResponse(BaseModel):
    """Immediate response after batch is queued."""
    batch_id: str
    job_id: str
    status: str = "queued"
    submitted_at: str


# ═══════════════════════════════════════════════════════════════════
# Job Status — GET /api/v1/batches/{job_id}/status, GET /api/v1/jobs/{job_id}
# ═══════════════════════════════════════════════════════════════════

class JobStatusResponse(BaseModel):
    """Response for job polling endpoints."""
    job_id: str
    status: str                                # queued|processing|completed|failed
    progress: int = 0                          # 0-100
    error: Optional[str] = None
    completed_at: Optional[str] = None


# ═══════════════════════════════════════════════════════════════════
# Records — GET /api/v1/records
# ═══════════════════════════════════════════════════════════════════

class RecordResponse(BaseModel):
    """Single analyzed record returned by GET /records."""
    item_id: str
    sentiment: Optional[str] = None
    confidence_score: Optional[float] = None
    emotion_tags: list[str] = []
    content_type: Optional[str] = None
    bot_flag: Optional[str] = None
    credibility_tier: Optional[str] = None
    crisis_severity: Optional[str] = None
    crisis_theme_group: Optional[str] = None
    impact_score: Optional[float] = None
    language_detected: Optional[str] = None
    low_confidence: Optional[bool] = None
    processed_at: Optional[str] = None


class RecordsListResponse(BaseModel):
    """Paginated response for GET /records."""
    items: list[RecordResponse]
    total: int
    page: int
    page_size: int


# ═══════════════════════════════════════════════════════════════════
# Sentiment Aggregate — GET /api/v1/sentiment/aggregate
# ═══════════════════════════════════════════════════════════════════

class SentimentAggregateResponse(BaseModel):
    """Sentiment distribution for a keyword."""
    positive_pct: float
    neutral_pct: float
    negative_pct: float
    total_items_analyzed: int


# ═══════════════════════════════════════════════════════════════════
# Engagement — POST /api/v1/engagement/generate
# ═══════════════════════════════════════════════════════════════════

class EngagementRequest(BaseModel):
    """POST /api/v1/engagement/generate request body."""
    request_id: str
    keyword: str
    theme_group_id: str
    requested_at: str


class EngagementJobResponse(BaseModel):
    """Immediate response after engagement job is queued."""
    request_id: str
    job_id: str
    status: str = "queued"
    submitted_at: str


# ═══════════════════════════════════════════════════════════════════
# Crisis — POST /api/v1/crisis/generate
# ═══════════════════════════════════════════════════════════════════

class CrisisRequest(BaseModel):
    """POST /api/v1/crisis/generate request body."""
    request_id: str
    crisis_description: str
    keyword: Optional[str] = None
    requested_at: str


class CrisisJobResponse(BaseModel):
    """Immediate response after crisis job is queued."""
    request_id: str
    job_id: str
    status: str = "queued"
    submitted_at: str


# ═══════════════════════════════════════════════════════════════════
# Report — POST /api/v1/reports/generate
# ═══════════════════════════════════════════════════════════════════

class DateRange(BaseModel):
    """Date range for report generation."""
    from_date: str = Field(alias="from")
    to_date: str = Field(alias="to")

    model_config = {"populate_by_name": True}


class ReportRequest(BaseModel):
    """POST /api/v1/reports/generate request body."""
    request_id: str
    keyword: str
    date_range: DateRange
    segments: list[str]
    include_summary: bool = True


class ReportJobResponse(BaseModel):
    """Immediate response after report job is queued."""
    request_id: str
    job_id: str
    status: str = "queued"
    submitted_at: str