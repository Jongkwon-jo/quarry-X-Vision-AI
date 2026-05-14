"""format_classifier.py — 정렬된 번호판의 형식을 자동 판별.

3가지 형식:
    PLATE_8       — 현행(2019.9~) 8자리 한 줄: 123가4567
    PLATE_7       — 구형(2004~2019) 7자리 한 줄: 12가3456
    PLATE_2ROW    — 지역명(1996~2003) 2열: 위 "대구" / 아래 "06라5245"

판별 신호 3가지:
    1. 종횡비 (aspect ratio): 1열 번호판은 width/height ≈ 4~5,
       2열 번호판은 ≈ 1.5~2.5
    2. 세로 프로필 분포: 2열 번호판은 위/아래로 글자 영역이 두 덩어리.
       1열은 가운데 한 덩어리.
    3. 글자 수 추정: 컨투어 분포로 1열 7~8개 vs 2열 2+5~6개 추정

조합 점수로 판별 (가중치 합산).
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional, Tuple

import numpy as np


class PlateFormat(Enum):
    """번호판 형식 enum."""
    PLATE_8 = 'plate_8'         # 현행 8자리 (123가4567)
    PLATE_7 = 'plate_7'         # 구형 7자리 (12가3456)
    PLATE_2ROW = 'plate_2row'   # 지역명 2열 (대구 / 06라5245)
    UNKNOWN = 'unknown'
    
    @property
    def is_single_line(self) -> bool:
        return self in (PlateFormat.PLATE_8, PlateFormat.PLATE_7)
    
    @property
    def is_two_row(self) -> bool:
        return self == PlateFormat.PLATE_2ROW
    
    @property
    def expected_char_count(self) -> int:
        return {
            PlateFormat.PLATE_8: 8,
            PlateFormat.PLATE_7: 7,
            PlateFormat.PLATE_2ROW: 7,  # 2자리 지역명 + 5자리(또는 6자리)
        }.get(self, 7)


@dataclass
class FormatClassifierConfig:
    """형식 판별 설정."""
    
    # 종횡비 임계값
    # 1열: 보통 4.0~5.5, 2열: 보통 1.5~2.5
    aspect_threshold: float = 3.0  # 이 값 미만이면 2열로 추정
    
    # 세로 프로필 신호 가중치
    # 2열은 세로 합 분포에서 위/아래 두 봉우리가 보임
    weight_aspect: float = 0.5
    weight_vertical_profile: float = 0.3
    weight_char_density: float = 0.2
    
    # 2열 판단을 위한 위/아래 영역 분리 허용 폭 (전체 높이 대비)
    valley_min_ratio: float = 0.3   # 중간 영역에 valley가 있어야 2열로 인정
    
    # 8자리 vs 7자리 판단 임계값 (단일 라인 가정)
    # 실제 한국 번호판: 7자리 ≈ 4.0~4.5, 8자리 ≈ 4.5~5.0
    # 4.3을 기준으로 분리 (둘 사이가 모호하면 글자 분리 후 길이로 재판정)
    eight_digit_aspect_min: float = 4.3


@dataclass
class FormatClassification:
    """형식 판별 결과."""
    plate_format: PlateFormat
    confidence: float
    aspect_ratio: float
    is_two_row_score: float = 0.0
    estimated_char_count: int = 0
    signals: Dict[str, float] = field(default_factory=dict)
    
    @property
    def is_confident(self) -> bool:
        return self.plate_format != PlateFormat.UNKNOWN and self.confidence >= 0.5


class PlateFormatClassifier:
    """정렬된 번호판 이미지의 형식을 자동 판별."""
    
    def __init__(self, config: Optional[FormatClassifierConfig] = None):
        self.config = config or FormatClassifierConfig()
    
    def classify(self, plate_image: np.ndarray) -> FormatClassification:
        """이미지를 받아 형식 판별.
        
        Args:
            plate_image: 정렬된 번호판 BGR 또는 그레이스케일 이미지.
        """
        import cv2
        
        if len(plate_image.shape) == 3:
            gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = plate_image
        
        h, w = gray.shape
        aspect_ratio = w / h if h > 0 else 0.0
        
        # 1. 종횡비 신호
        aspect_score = self._aspect_score(aspect_ratio)
        
        # 2. 세로 프로필 신호 (2열 여부)
        two_row_score = self._vertical_profile_score(gray)
        
        # 3. 글자 밀도 추정 (컨투어 수)
        char_count = self._estimate_char_count(gray)
        
        signals = {
            'aspect_ratio': aspect_ratio,
            'aspect_score_2row': aspect_score,
            'two_row_score': two_row_score,
            'estimated_char_count': float(char_count),
        }
        
        # 결합 점수 (높을수록 2열)
        combined_2row = (
            self.config.weight_aspect * aspect_score +
            self.config.weight_vertical_profile * two_row_score
        )
        
        # 판정 로직
        if combined_2row >= 0.5:
            plate_format = PlateFormat.PLATE_2ROW
            confidence = combined_2row
        else:
            # 단일 라인 — 8자리 vs 7자리
            if aspect_ratio >= self.config.eight_digit_aspect_min:
                plate_format = PlateFormat.PLATE_8
            else:
                plate_format = PlateFormat.PLATE_7
            # 단일 라인 신뢰도는 (1 - 2열 점수)
            confidence = 1.0 - combined_2row
        
        return FormatClassification(
            plate_format=plate_format,
            confidence=confidence,
            aspect_ratio=aspect_ratio,
            is_two_row_score=combined_2row,
            estimated_char_count=char_count,
            signals=signals,
        )
    
    # ===== 신호 추출 =====
    
    def _aspect_score(self, aspect_ratio: float) -> float:
        """종횡비 → 2열 가능성 점수 (0~1).
        
        aspect_ratio < 3.0 면 강하게 2열로 의심.
        """
        if aspect_ratio < 2.0:
            return 1.0
        if aspect_ratio >= self.config.aspect_threshold:
            return 0.0
        # 2.0 ~ aspect_threshold 사이를 선형 보간
        return (self.config.aspect_threshold - aspect_ratio) / \
               (self.config.aspect_threshold - 2.0)
    
    def _vertical_profile_score(self, gray: np.ndarray) -> float:
        """세로 프로필 분포로 2열 여부 추정 (0~1).
        
        원리:
            - Otsu 이진화 후 가로 방향으로 픽셀 합계 (1D 프로필).
            - 2열 번호판은 위/아래로 두 개의 봉우리가 보임.
            - 1열 번호판은 가운데 단일 봉우리.
            - 봉우리 사이 valley(저점)가 충분히 깊으면 2열로 판단.
        """
        import cv2
        
        h, w = gray.shape
        if h < 20:
            return 0.0
        
        _, binary = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
        
        # 가로 합 (1D, 길이 h)
        row_sum = np.sum(binary, axis=1).astype(np.float32)
        
        # 스무딩
        kernel = max(3, h // 20)
        row_sum_smooth = np.convolve(
            row_sum, np.ones(kernel) / kernel, mode='same'
        )
        
        # 봉우리 임계값
        threshold = row_sum_smooth.max() * 0.4
        
        # 봉우리 영역 마스크
        peaks = row_sum_smooth >= threshold
        
        # 봉우리 영역이 두 개로 분리되어 있는지 검사
        # 봉우리 → 비봉우리 → 봉우리 패턴이 보이면 2열
        transitions = 0
        in_peak = peaks[0]
        for v in peaks[1:]:
            if v != in_peak:
                transitions += 1
                in_peak = v
        
        # 봉우리 그룹이 2개 이상이면 (transitions >= 3, 시작이 봉우리 기준)
        # 즉 [F-T-F-T-F] 또는 [T-F-T-F-T] 패턴
        peak_groups = self._count_peak_groups(peaks)
        
        if peak_groups < 2:
            return 0.0
        
        # 두 봉우리 사이 valley 깊이 측정
        # valley가 임계값보다 충분히 낮으면 명확한 2열
        first_peak_idx, second_peak_idx, valley_min = \
            self._measure_valley_depth(row_sum_smooth, threshold)
        
        if valley_min is None:
            return 0.5  # 봉우리 2개 있지만 valley 측정 실패
        
        peak_max = row_sum_smooth.max()
        if peak_max == 0:
            return 0.0
        
        # valley가 봉우리 대비 얼마나 낮은지 (0~1)
        valley_ratio = 1.0 - (valley_min / peak_max)
        
        # 0.3 미만 valley_ratio면 단순한 글자 사이 공백일 수 있음
        if valley_ratio < self.config.valley_min_ratio:
            return 0.3
        
        return min(1.0, valley_ratio)
    
    @staticmethod
    def _count_peak_groups(peaks: np.ndarray) -> int:
        """봉우리 영역(True의 연속 구간) 개수 세기."""
        groups = 0
        in_group = False
        for v in peaks:
            if v and not in_group:
                groups += 1
                in_group = True
            elif not v:
                in_group = False
        return groups
    
    @staticmethod
    def _measure_valley_depth(
        profile: np.ndarray, threshold: float
    ) -> Tuple[Optional[int], Optional[int], Optional[float]]:
        """첫 봉우리와 두 번째 봉우리 사이 valley 최소값 반환."""
        peaks = profile >= threshold
        
        # 첫 봉우리 끝 인덱스
        first_end = None
        seen_peak = False
        for i, v in enumerate(peaks):
            if v:
                seen_peak = True
            elif seen_peak:
                first_end = i
                break
        
        if first_end is None:
            return None, None, None
        
        # 두 번째 봉우리 시작 인덱스
        second_start = None
        for i in range(first_end, len(peaks)):
            if peaks[i]:
                second_start = i
                break
        
        if second_start is None:
            return None, None, None
        
        # 사이 구간의 최소값
        valley_segment = profile[first_end:second_start]
        if len(valley_segment) == 0:
            return None, None, None
        
        return first_end, second_start, float(valley_segment.min())
    
    @staticmethod
    def _estimate_char_count(gray: np.ndarray) -> int:
        """대략적인 글자 수 추정 (컨투어 기반)."""
        import cv2
        
        h, w = gray.shape
        _, binary = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
        
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        # 너무 작거나 너무 큰 컨투어 제외 (글자 크기에 해당하는 것만)
        char_count = 0
        for c in contours:
            x, y, cw, ch = cv2.boundingRect(c)
            # 높이가 전체의 20% 이상이고 넓이가 적당한 것
            if ch > h * 0.2 and 5 < cw < w * 0.5:
                char_count += 1
        
        return char_count
