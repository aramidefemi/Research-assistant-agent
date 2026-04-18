from typing import TypedDict, Optional

class PaperState(TypedDict):
    filename: str
    pdf_text: str
    summary: str
    key_findings: str
    methodology: str
    relevance_score: float
    relevance_reason: str
    fit: bool
    error: Optional[str]
