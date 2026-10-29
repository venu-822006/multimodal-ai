from __future__ import annotations

from enum import StrEnum


class AnalysisStatus(StrEnum):
    QUEUED = "queued"
    VALIDATING = "validating"
    SEARCHING_DATA = "searching_data"
    DOWNLOADING = "downloading"
    PREPROCESSING = "preprocessing"
    FLOOD_MAPPING = "flood_mapping"
    DAMAGE_ANALYSIS = "damage_analysis"
    CONNECTIVITY_ANALYSIS = "connectivity_analysis"
    GENERATING_REPORT = "generating_report"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


TERMINAL_STATUSES: frozenset[str] = frozenset(
    {AnalysisStatus.COMPLETED, AnalysisStatus.FAILED, AnalysisStatus.CANCELLED}
)

ACTIVE_STATUSES: tuple[str, ...] = (
    AnalysisStatus.VALIDATING,
    AnalysisStatus.SEARCHING_DATA,
    AnalysisStatus.DOWNLOADING,
    AnalysisStatus.PREPROCESSING,
    AnalysisStatus.FLOOD_MAPPING,
    AnalysisStatus.DAMAGE_ANALYSIS,
    AnalysisStatus.CONNECTIVITY_ANALYSIS,
    AnalysisStatus.GENERATING_REPORT,
)

PIPELINE_STAGES: tuple[str, ...] = (
    AnalysisStatus.VALIDATING,
    AnalysisStatus.SEARCHING_DATA,
    AnalysisStatus.DOWNLOADING,
    AnalysisStatus.PREPROCESSING,
    AnalysisStatus.FLOOD_MAPPING,
    AnalysisStatus.DAMAGE_ANALYSIS,
    AnalysisStatus.CONNECTIVITY_ANALYSIS,
    AnalysisStatus.GENERATING_REPORT,
)

ALLOWED_TRANSITIONS: dict[str, frozenset[str]] = {
    AnalysisStatus.QUEUED: frozenset(
        {AnalysisStatus.VALIDATING, AnalysisStatus.FAILED, AnalysisStatus.CANCELLED}
    ),
    AnalysisStatus.VALIDATING: frozenset(
        {AnalysisStatus.SEARCHING_DATA, AnalysisStatus.FAILED, AnalysisStatus.CANCELLED}
    ),
    AnalysisStatus.SEARCHING_DATA: frozenset(
        {AnalysisStatus.DOWNLOADING, AnalysisStatus.FAILED, AnalysisStatus.CANCELLED}
    ),
    AnalysisStatus.DOWNLOADING: frozenset(
        {AnalysisStatus.PREPROCESSING, AnalysisStatus.FAILED, AnalysisStatus.CANCELLED}
    ),
    AnalysisStatus.PREPROCESSING: frozenset(
        {AnalysisStatus.FLOOD_MAPPING, AnalysisStatus.FAILED, AnalysisStatus.CANCELLED}
    ),
    AnalysisStatus.FLOOD_MAPPING: frozenset(
        {
            AnalysisStatus.DAMAGE_ANALYSIS,
            AnalysisStatus.FAILED,
            AnalysisStatus.CANCELLED,
        }
    ),
    AnalysisStatus.DAMAGE_ANALYSIS: frozenset(
        {
            AnalysisStatus.CONNECTIVITY_ANALYSIS,
            AnalysisStatus.FAILED,
            AnalysisStatus.CANCELLED,
        }
    ),
    AnalysisStatus.CONNECTIVITY_ANALYSIS: frozenset(
        {AnalysisStatus.GENERATING_REPORT, AnalysisStatus.FAILED, AnalysisStatus.CANCELLED}
    ),
    AnalysisStatus.GENERATING_REPORT: frozenset(
        {AnalysisStatus.COMPLETED, AnalysisStatus.FAILED, AnalysisStatus.CANCELLED}
    ),
    AnalysisStatus.COMPLETED: frozenset(),
    AnalysisStatus.FAILED: frozenset(),
    AnalysisStatus.CANCELLED: frozenset(),
}


class JobStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ExecutionStatus(StrEnum):
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    SKIPPED = "skipped"


class FailureCode(StrEnum):
    INVALID_REQUEST = "invalid_request"
    NO_IMAGERY = "no_imagery"
    INSUFFICIENT_COVERAGE = "insufficient_coverage"
    IMAGERY_UNAVAILABLE = "imagery_unavailable"
    PROCESSING_ERROR = "processing_error"
    DEPENDENCY_UNAVAILABLE = "dependency_unavailable"
    CANCELLED = "cancelled"
    INTERNAL_ERROR = "internal_error"


class DetectionMethod(StrEnum):
    AUTO = "auto"
    SAR_CHANGE_DETECTION = "sar_change_detection"
    NDWI_OPTICAL = "ndwi_optical"


class ImagerySource(StrEnum):
    SENTINEL1_RTC = "sentinel-1-rtc"
    SENTINEL2_L2A = "sentinel-2-l2a"


def is_valid_transition(current: str, target: str) -> bool:
    return target in ALLOWED_TRANSITIONS.get(current, frozenset())