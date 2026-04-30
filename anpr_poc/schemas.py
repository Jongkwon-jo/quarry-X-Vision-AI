"""
schemas.py — 단계 간 데이터 전달 객체

파이프라인의 각 단계는 dataclass로 정의된 객체를 입출력으로 사용합니다.
이렇게 하면 단계 간 결합이 약해지고 디버깅·테스트가 쉬워집니다.
"""
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Literal

import numpy as np


# 글자 자리 타입
CharType = Literal['digit', 'hangul', 'unknown']


@dataclass
class Detection:
    """① detection 출력: 검출된 번호판 한 개."""
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    confidence: float
    crop: np.ndarray                  # 원본에서 잘라낸 BGR 이미지
    class_name: str = 'license_plate'
    
    @property
    def width(self) -> int:
        return self.bbox[2] - self.bbox[0]
    
    @property
    def height(self) -> int:
        return self.bbox[3] - self.bbox[1]


@dataclass
class PreprocessedPlate:
    """② preprocessing 출력: 정렬·전처리된 번호판."""
    rectified: np.ndarray   # 4점 보정된 컬러 이미지 (글자 분리용)
    enhanced: np.ndarray    # CLAHE + bilateral 적용된 그레이스케일 (OCR용)
    rectification_success: bool  # 4점 코너 검출 성공 여부 (False면 단순 리사이즈)


@dataclass
class CharCrop:
    """③ split 출력: 분리된 단일 글자."""
    image: np.ndarray              # BGR 이미지
    x_range: Tuple[int, int]       # 정렬된 번호판 내 x 좌표 범위
    width: int
    char_type: CharType = 'unknown'
    index: int = -1                # 좌→우 순서 인덱스


@dataclass
class CharRecognition:
    """④ ocr 출력: 단일 글자 인식 결과."""
    char_index: int
    char_type: CharType
    predicted: Optional[str]
    confidence: float
    raw_text: Optional[str] = None  # 후처리 전 OCR 원본
    engine: str = 'unknown'         # 'easyocr', 'paddleocr', 'ensemble'


@dataclass
class PlateResult:
    """전체 파이프라인의 최종 출력."""
    # 입력
    source_image_path: Optional[str] = None
    
    # 단계별 결과
    detection: Optional[Detection] = None
    preprocessed: Optional[PreprocessedPlate] = None
    char_crops: List[CharCrop] = field(default_factory=list)
    char_recognitions: List[CharRecognition] = field(default_factory=list)
    
    # 최종 결과
    raw_text: Optional[str] = None      # 글자별 결과 단순 결합
    corrected_text: Optional[str] = None  # 후처리 보정 후
    is_valid_format: bool = False
    overall_confidence: float = 0.0
    
    # 강화 검증 리포트 (use_enhanced=True인 경우)
    # 순환 임포트 회피를 위해 Any 타입으로 보관
    validation_report: Optional[object] = None
    
    # 디버깅 정보
    error_stage: Optional[str] = None   # 실패 단계 이름
    error_message: Optional[str] = None
    
    @property
    def success(self) -> bool:
        return self.is_valid_format and self.corrected_text is not None
    
    @property
    def trust_level(self) -> str:
        """강화 검증 결과가 있으면 신뢰 등급을 반환, 없으면 형식 검증 기반."""
        if self.validation_report is not None:
            return self.validation_report.trust_level.value
        return 'pass' if self.success else 'fail'
    
    def summary(self) -> str:
        # 강화 검증 결과가 있으면 그쪽 우선 사용
        if self.validation_report is not None:
            return self.validation_report.summary()
        if self.success:
            return f'✓ "{self.corrected_text}" (conf={self.overall_confidence:.3f})'
        elif self.corrected_text:
            return f'? "{self.corrected_text}" (형식 불일치)'
        else:
            return f'✗ 실패: {self.error_stage or "unknown"} — {self.error_message or ""}'
