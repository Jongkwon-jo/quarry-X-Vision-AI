"""adaptive_splitter.py — 형식 자동 판별 → 적절한 분리기로 라우팅.

이 클래스를 사용하면 호출부에서는 형식 차이를 신경 쓰지 않고
.split(plate)만 호출하면 된다.
"""
from typing import List, Optional

import numpy as np

from ..config import SplitConfig
from ..schemas import CharCrop
from .splitter import CharSplitter
from .two_row_splitter import TwoRowSplitter
from .format_classifier import (
    PlateFormatClassifier, PlateFormat, FormatClassification,
    FormatClassifierConfig,
)


class AdaptiveSplitter:
    """형식 자동 판별 후 라우팅하는 통합 분리기.
    
    사용 예:
        splitter = AdaptiveSplitter(split_config)
        chars = splitter.split(plate_image)
        
        # 판별 결과를 함께 반환받고 싶을 때
        chars, fmt = splitter.split_with_format(plate_image)
        print(f'형식: {fmt.plate_format.value}')
    """
    
    def __init__(
        self,
        config: SplitConfig,
        format_config: Optional[FormatClassifierConfig] = None,
    ):
        self.config = config
        self.classifier = PlateFormatClassifier(format_config)
        self.single_line_splitter = CharSplitter(config)
        self.two_row_splitter = TwoRowSplitter(config)
    
    def split(self, rectified_plate: np.ndarray) -> List[CharCrop]:
        """형식 판별 후 분리. CharCrop 리스트만 반환."""
        chars, _ = self.split_with_format(rectified_plate)
        return chars
    
    def split_with_format(
        self,
        rectified_plate: np.ndarray,
    ) -> tuple:
        """형식 판별 + 분리. (CharCrop 리스트, FormatClassification) 반환."""
        if rectified_plate is None or rectified_plate.size == 0:
            return [], None
        
        # 1. 형식 판별
        fmt = self.classifier.classify(rectified_plate)
        
        # 2. 라우팅
        if fmt.plate_format.is_two_row:
            chars = self.two_row_splitter.split(rectified_plate)
        else:
            # 단일 라인 — expected_chars는 형식에 따라 결정
            expected = fmt.plate_format.expected_char_count
            if fmt.plate_format == PlateFormat.UNKNOWN:
                expected = None  # 자동 추정
            chars = self.single_line_splitter.split(
                rectified_plate, expected_chars=expected
            )
        
        return chars, fmt
