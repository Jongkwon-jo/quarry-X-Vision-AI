"""base.py — OCR 엔진의 공통 인터페이스."""
from abc import ABC, abstractmethod
from typing import List, Tuple

import cv2
import numpy as np

from ..config import OCRConfig, PLATE_HANGUL, PLATE_DIGITS
from ..schemas import CharCrop, CharRecognition


class BaseOCR(ABC):
    """모든 OCR 엔진이 구현해야 하는 공통 인터페이스."""
    
    def __init__(self, config: OCRConfig):
        self.config = config
    
    @abstractmethod
    def recognize_char(
        self,
        char_image: np.ndarray,
        char_type: str,
    ) -> Tuple[str, float]:
        """단일 글자 이미지 → (인식된 문자, 신뢰도).
        
        실패 시 (None, 0.0) 반환.
        """
        ...
    
    @property
    @abstractmethod
    def engine_name(self) -> str:
        ...
    
    def recognize_all(self, char_crops: List[CharCrop]) -> List[CharRecognition]:
        """모든 글자에 대해 OCR 수행.
        
        자리 타입에 맞는 화이트리스트를 적용한다.
        """
        results = []
        for crop in char_crops:
            text, conf = self.recognize_char(crop.image, crop.char_type)
            
            # 화이트리스트 후처리
            allowlist = self.get_allowlist(crop.char_type)
            if text:
                filtered = ''.join(c for c in text if c in allowlist)
                final = filtered[0] if filtered else None
            else:
                final = None
            
            results.append(CharRecognition(
                char_index=crop.index,
                char_type=crop.char_type,
                predicted=final,
                confidence=conf if final else 0.0,
                raw_text=text,
                engine=self.engine_name,
            ))
        return results
    
    @staticmethod
    def get_allowlist(char_type: str) -> str:
        """자리 타입에 맞는 화이트리스트 반환."""
        if char_type == 'hangul':
            return PLATE_HANGUL
        elif char_type == 'digit':
            return PLATE_DIGITS
        return PLATE_HANGUL + PLATE_DIGITS
    
    @staticmethod
    def upscale(image: np.ndarray, scale: int) -> np.ndarray:
        """단일 글자 OCR을 위한 확대."""
        if image is None or image.size == 0:
            return image
        h, w = image.shape[:2]
        return cv2.resize(image, (w * scale, h * scale),
                          interpolation=cv2.INTER_CUBIC)
