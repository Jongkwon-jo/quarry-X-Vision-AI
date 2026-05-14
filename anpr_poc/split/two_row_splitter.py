"""two_row_splitter.py — 지역명 2열 번호판 전용 글자 분리기.

지역명 번호판 구조:
    ┌─────────────────┐
    │     대  구       │  ← 윗줄: 지역명 (한글 2자)
    │   06 라 5245     │  ← 아랫줄: 차종(2자) + 한글(1자) + 일련번호(4자)
    └─────────────────┘

처리 흐름:
    1. 가로 프로필로 위/아래 라인 분리점(valley) 찾기
    2. 각 라인을 기존 CharSplitter 로직으로 좌→우 글자 분리
    3. 각 글자에 자리 타입 부여:
       - 윗줄: 모두 hangul (지역명)
       - 아랫줄: 0~1=digit, 2=hangul, 3~6=digit
"""
from typing import List, Optional, Tuple

import numpy as np

from ..config import SplitConfig
from ..schemas import CharCrop
from .splitter import CharSplitter


class TwoRowSplitter:
    """지역명 2열 번호판 전용 분리기.
    
    내부적으로 기존 CharSplitter를 줄별로 두 번 호출.
    """
    
    def __init__(self, config: SplitConfig):
        self.config = config
        # 각 줄에 대한 단일 라인 분리기 (재사용)
        self._line_splitter = CharSplitter(config)
    
    def split(self, rectified_plate: np.ndarray) -> List[CharCrop]:
        """2열 번호판 → 글자 리스트.
        
        Returns:
            CharCrop 리스트. index 순서는 지역명(0~) → 본문(다음~) 순.
            char_type:
                - 지역명 자리: 'hangul'
                - 본문의 한글 자리(인덱스 2 in 본문): 'hangul'
                - 본문의 숫자 자리: 'digit'
        """
        import cv2
        
        if rectified_plate is None or rectified_plate.size == 0:
            return []
        
        # 1. 위·아래 라인 분리
        top_band, bottom_band = self._split_rows(rectified_plate)
        
        if top_band is None or bottom_band is None:
            return []
        
        # 2. 각 라인을 단일 라인 분리기로 처리
        # 윗줄 (지역명): expected_chars = 2 (보통 "서울", "대구" 등 2자)
        top_chars = self._line_splitter.split(top_band, expected_chars=2)
        
        # 아랫줄: 5자(2자리 차종+한글+4자리) 또는 6자(3자리+한글+4자리)
        # 둘 다 시도해 더 자연스러운 결과 채택
        bottom_chars = self._split_bottom_line(bottom_band)
        
        # 3. 자리 타입 부여
        # 윗줄: 모두 hangul
        for i, c in enumerate(top_chars):
            c.index = i
            c.char_type = 'hangul'
        
        # 아랫줄: 인덱스 위치에 따라 digit/hangul
        n_bottom = len(bottom_chars)
        if n_bottom == 7:    # 2자리 차종 + 한글 + 4자리 일련번호
            hangul_local_idx = 2
        elif n_bottom == 8:  # 3자리 차종 + 한글 + 4자리 일련번호
            hangul_local_idx = 3
        else:
            hangul_local_idx = -1  # 알 수 없음 → 자동 판별 어려움
        
        offset = len(top_chars)
        for i, c in enumerate(bottom_chars):
            c.index = offset + i
            if i == hangul_local_idx:
                c.char_type = 'hangul'
            else:
                c.char_type = 'digit' if hangul_local_idx >= 0 else 'unknown'
        
        return top_chars + bottom_chars
    
    # ===== 내부 =====
    
    def _split_rows(
        self, plate: np.ndarray
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """가로 프로필로 위·아래 라인 영역 분리.
        
        Returns:
            (top_band, bottom_band). 분리 실패 시 (None, None).
        """
        import cv2
        
        if len(plate.shape) == 3:
            gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
        else:
            gray = plate
        
        h, w = gray.shape
        if h < 20:
            return None, None
        
        # Otsu 이진화 (글자=255, 배경=0)
        _, binary = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
        
        # 가로 방향 합계 (세로 길이의 1D 프로필)
        row_sum = np.sum(binary, axis=1).astype(np.float32)
        
        # 스무딩
        kernel = max(3, h // 20)
        smoothed = np.convolve(
            row_sum, np.ones(kernel) / kernel, mode='same'
        )
        
        # 가운데 영역에서 가장 깊은 valley 찾기
        # 위·아래 끝 10%는 테두리일 수 있으므로 제외
        margin = int(h * 0.15)
        center_segment = smoothed[margin:h-margin]
        if len(center_segment) == 0:
            return None, None
        
        # 가운데 영역의 최저점 인덱스
        valley_offset = int(np.argmin(center_segment))
        valley_y = margin + valley_offset
        
        # valley가 너무 가장자리에 가까우면 2열이 아닐 수 있음 — 안전 가드
        if valley_y < h * 0.25 or valley_y > h * 0.75:
            return None, None
        
        # 위·아래 영역 분리 (약간의 여유 마진)
        top_band = plate[:valley_y, :]
        bottom_band = plate[valley_y:, :]
        
        # 각 영역이 너무 얇으면 실패
        if top_band.shape[0] < 15 or bottom_band.shape[0] < 15:
            return None, None
        
        return top_band, bottom_band
    
    def _split_bottom_line(self, bottom: np.ndarray) -> List[CharCrop]:
        """아랫줄 분리 — 7자 또는 8자 가정으로 시도."""
        # 먼저 7자 가정
        chars_7 = self._line_splitter.split(bottom, expected_chars=7)
        if len(chars_7) == 7:
            return chars_7
        
        # 7자가 안 되면 8자 시도
        chars_8 = self._line_splitter.split(bottom, expected_chars=8)
        if len(chars_8) == 8:
            return chars_8
        
        # 둘 다 실패하면 7자 결과 반환 (best effort)
        return chars_7
