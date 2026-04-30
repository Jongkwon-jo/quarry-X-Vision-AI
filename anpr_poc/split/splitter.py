"""splitter.py — Projection Profile 기반 글자 분리.

이전 노트북에서 검증한 알고리즘을 모듈화한 버전.
"""
from typing import List, Optional

import cv2
import numpy as np
from scipy.signal import find_peaks

from ..config import SplitConfig
from ..schemas import CharCrop


class CharSplitter:
    """정렬된 번호판 → 7~8개 글자 영역으로 분리."""
    
    def __init__(self, config: SplitConfig):
        self.config = config
    
    def split(
        self,
        rectified_plate: np.ndarray,
        expected_chars: Optional[int] = None,
    ) -> List[CharCrop]:
        """글자 분리 메인 진입점.
        
        Args:
            rectified_plate: 정렬된 번호판 BGR 이미지
            expected_chars: 7 또는 8. None이면 자동 추정.
        
        Returns:
            CharCrop 리스트 (좌→우 순서, 자리 타입 표시 포함)
        """
        # 1. 이진화
        binary = self._binarize(rectified_plate)
        
        # 2. 상하 트리밍 (테두리 노이즈 제거)
        h = binary.shape[0]
        top = int(h * self.config.trim_top_ratio)
        bottom = int(h * (1 - self.config.trim_bottom_ratio))
        trimmed = binary[top:bottom, :]
        
        # 3. 세로 방향 프로필
        profile = np.sum(trimmed, axis=0)
        
        # 4. 스무딩
        kernel_size = max(3, len(profile) // self.config.smoothing_kernel_divisor)
        smoothed = np.convolve(
            profile, np.ones(kernel_size) / kernel_size, mode='same'
        )
        
        # 5. valley 검출
        threshold = smoothed.max() * self.config.threshold_ratio
        min_distance = max(1, len(profile) // self.config.min_distance_divisor)
        valleys, _ = find_peaks(
            -smoothed, distance=min_distance, height=-threshold
        )
        
        # 6. 글자 경계 결정
        boundaries = [0] + list(valleys) + [len(profile)]
        
        # 7. 글자별 크롭 생성
        crops = []
        for i in range(len(boundaries) - 1):
            x1, x2 = int(boundaries[i]), int(boundaries[i + 1])
            width = x2 - x1
            if width < self.config.min_char_width:
                continue
            crops.append(CharCrop(
                image=rectified_plate[:, x1:x2],
                x_range=(x1, x2),
                width=width,
            ))
        
        # 8. 글자 수 보정
        if expected_chars is None:
            # 6~7개면 7로, 7~8개면 가까운 쪽으로 추정
            if len(crops) <= 7:
                expected_chars = 7
            else:
                expected_chars = 8
        
        if len(crops) != expected_chars:
            crops = self._refine(crops, expected_chars, rectified_plate)
        
        # 9. 자리 타입 부여 + 인덱스
        crops = self._assign_types(crops)
        
        return crops
    
    @staticmethod
    def _binarize(plate_image: np.ndarray) -> np.ndarray:
        """Otsu 이진화 (글자=255, 배경=0)."""
        if len(plate_image.shape) == 3:
            gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = plate_image
        _, binary = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
        return binary
    
    def _refine(
        self,
        crops: List[CharCrop],
        target_count: int,
        source_image: np.ndarray,
    ) -> List[CharCrop]:
        """글자 수가 안 맞을 때 폭 기반 병합/분할."""
        crops = sorted(crops, key=lambda c: c.x_range[0])
        
        # 너무 많으면: 좁은 것을 인접 영역에 병합
        while len(crops) > target_count:
            narrowest_idx = min(range(len(crops)),
                                 key=lambda i: crops[i].width)
            
            if narrowest_idx == 0:
                merge_target = 1
            elif narrowest_idx == len(crops) - 1:
                merge_target = len(crops) - 2
            else:
                left_w = crops[narrowest_idx - 1].width
                right_w = crops[narrowest_idx + 1].width
                merge_target = (narrowest_idx - 1
                                if left_w < right_w
                                else narrowest_idx + 1)
            
            a, b = sorted([narrowest_idx, merge_target])
            new_x_range = (crops[a].x_range[0], crops[b].x_range[1])
            crops[a] = CharCrop(
                image=source_image[:, new_x_range[0]:new_x_range[1]],
                x_range=new_x_range,
                width=new_x_range[1] - new_x_range[0],
            )
            del crops[b]
        
        # 너무 적으면: 넓은 것을 절반 분할
        while len(crops) < target_count:
            widest_idx = max(range(len(crops)),
                              key=lambda i: crops[i].width)
            old = crops[widest_idx]
            mid = (old.x_range[0] + old.x_range[1]) // 2
            
            new_left = CharCrop(
                image=source_image[:, old.x_range[0]:mid],
                x_range=(old.x_range[0], mid),
                width=mid - old.x_range[0],
            )
            new_right = CharCrop(
                image=source_image[:, mid:old.x_range[1]],
                x_range=(mid, old.x_range[1]),
                width=old.x_range[1] - mid,
            )
            crops[widest_idx] = new_left
            crops.insert(widest_idx + 1, new_right)
        
        return crops
    
    @staticmethod
    def _assign_types(crops: List[CharCrop]) -> List[CharCrop]:
        """한국 번호판 형식에 따라 자리 타입 부여.
        
        - 7자리(12가3456): 인덱스 2가 한글
        - 8자리(123가4567): 인덱스 3이 한글
        - 그 외: 모두 unknown
        """
        n = len(crops)
        if n == 7:
            hangul_idx = 2
        elif n == 8:
            hangul_idx = 3
        else:
            for i, c in enumerate(crops):
                c.index = i
            return crops
        
        for i, c in enumerate(crops):
            c.index = i
            c.char_type = 'hangul' if i == hangul_idx else 'digit'
        return crops
