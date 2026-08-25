"""Generic project verification engine: ecosystem detection + adapter dispatch."""

from app.services.verification.base import VerificationAdapter, VerificationResult
from app.services.verification.detector import DetectionResult, ProjectDetector
from app.services.verification.engine import VerificationEngine

__all__ = [
    "VerificationAdapter",
    "VerificationResult",
    "DetectionResult",
    "ProjectDetector",
    "VerificationEngine",
]
