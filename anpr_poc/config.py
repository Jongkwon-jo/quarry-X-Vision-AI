"""
config.py — 파이프라인 전역 설정

모든 임계값과 경로를 한곳에 모아 관리합니다.
실험 시 이 파일만 수정하면 전체 파이프라인 동작이 바뀝니다.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple


# ===== 한국 번호판 문자 집합 =====
# 자가용 + 사업용에 쓰이는 한글 (순서는 위치 정렬 기준)
PLATE_HANGUL = (
    '가나다라마'
    '거너더러머버서어저'
    '고노도로모보소오조'
    '구누두루무부수우주'
    '바사아자'  # 사업용
    '하허호'    # 외교/특수
)
PLATE_DIGITS = '0123456789'
PLATE_ALLOWLIST = PLATE_HANGUL + PLATE_DIGITS


# ===== OCR 후처리: 혼동 문자 매핑 =====
# 숫자 자리에 영문이 잘못 인식된 경우 자동 보정
DIGIT_FIX_MAP = {
    'O': '0', 'D': '0', 'Q': '0',
    'I': '1', 'L': '1',
    'Z': '2',
    'S': '5',
    'G': '6',
    'B': '8',
}

# 한글 자리에 잘못 인식된 글자를 한글로 매핑 (현장 데이터 보면서 채워나감)
HANGUL_FIX_MAP = {
    # 숫자/영문 → 한글 (자주 헷갈리는 패턴)
    '0': '아',  '6': '바',
    'O': '아',  'D': '아',
    # 비슷한 한글 → 표준 한글
    '카': '가', '갸': '가',
    '냐': '나', '녀': '너',
    '댜': '다', '뎌': '더',
    '랴': '라', '려': '러',
    '먀': '마', '며': '머',
    '뱌': '바', '벼': '버',
    '샤': '사', '셔': '서',
    '야': '아', '여': '어',
    '쟈': '자', '져': '저',
    '햐': '하', '혀': '허',
}


@dataclass
class DetectionConfig:
    """YOLO 번호판 검출 설정."""
    weights_path: str = 'runs/detect/train-29/weights/best.pt'
    conf_threshold: float = 0.5
    iou_threshold: float = 0.45
    device: str = 'cuda'  # 'cuda', 'cpu', '0', '0,1' 등
    plate_class_name: str = 'license_plate'
    vehicle_class_name: str = 'vehicle'  # 멀티클래스 학습 시


@dataclass
class PreprocessingConfig:
    """번호판 전처리 설정."""
    target_size: Tuple[int, int] = (280, 80)  # 정렬 후 크기 (width, height)
    clahe_clip_limit: float = 2.0
    clahe_tile_size: Tuple[int, int] = (8, 8)
    bilateral_d: int = 11
    bilateral_sigma_color: int = 17
    bilateral_sigma_space: int = 17
    canny_low: int = 50
    canny_high: int = 150


@dataclass
class SplitConfig:
    """글자 분리 설정."""
    expected_chars_options: Tuple[int, ...] = (7, 8)
    trim_top_ratio: float = 0.1     # 상단 트리밍 비율
    trim_bottom_ratio: float = 0.1  # 하단 트리밍 비율
    threshold_ratio: float = 0.3    # valley 검출 임계값 (max * ratio)
    min_distance_divisor: int = 12  # min_distance = profile_len / divisor
    smoothing_kernel_divisor: int = 50
    min_char_width: int = 5


@dataclass
class OCRConfig:
    """OCR 인식 설정."""
    engine: str = 'easyocr'       # 'easyocr', 'paddleocr', 'ensemble'
    use_gpu: bool = True
    char_scale: int = 4           # 단일 글자 확대 배율
    text_threshold: float = 0.4
    low_text: float = 0.3
    link_threshold: float = 0.3
    canvas_size: int = 2560
    # 앙상블 시 한글 자리에서 우선할 엔진
    prefer_for_hangul: str = 'paddleocr'
    # 두 엔진 일치 시 신뢰도 가산 비율
    agreement_boost: float = 1.2


@dataclass
class ValidationConfig:
    """형식 검증 설정."""
    auto_fix_digits: bool = True
    auto_fix_hangul: bool = True
    
    # 강화 검증 (4 레이어) 사용 여부
    use_enhanced: bool = True
    
    # 석산 도메인 검증 강제 여부
    # True: 화물차/승용차/승합차가 아니면 SUSPECT로 강등
    # False: 도메인 검증 결과는 정보 제공용으로만 사용
    require_quarry_compatible: bool = False


@dataclass
class PipelineConfig:
    """전체 파이프라인 설정 묶음."""
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    ocr: OCRConfig = field(default_factory=OCRConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    # 디버깅: 중간 결과 저장
    debug: bool = False
    debug_dir: str = 'debug_outputs'
    
    @classmethod
    def default(cls):
        return cls()


# ===== 한국 번호판 정규식 =====
import re
PLATE_PATTERNS = [
    re.compile(r'^\d{2,3}[가-힣]\d{4}$'),               # 12가3456 / 123가4567
    re.compile(r'^[가-힣]{2}\d{2,3}[가-힣]\d{4}$'),     # 서울12가3456 (구형)
]
