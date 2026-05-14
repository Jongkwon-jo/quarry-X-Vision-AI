"""split — 번호판 형식별 글자 분리.

3가지 분리 경로:
    1. CharSplitter        — 단일 라인 (현행 8자리 / 구형 7자리)
    2. TwoRowSplitter      — 지역명 2열 번호판
    3. AdaptiveSplitter    — 형식 자동 판별 후 적절한 분리기로 라우팅 (기본 권장)

공통 출력: List[CharCrop] (자리 타입 + 좌→우 인덱스 부여됨)
"""
from .splitter import CharSplitter
from .two_row_splitter import TwoRowSplitter
from .format_classifier import (
    PlateFormatClassifier, PlateFormat, FormatClassification,
    FormatClassifierConfig,
)
from .adaptive_splitter import AdaptiveSplitter

__all__ = [
    'CharSplitter', 'TwoRowSplitter', 'AdaptiveSplitter',
    'PlateFormatClassifier', 'PlateFormat', 'FormatClassification',
    'FormatClassifierConfig',
]
