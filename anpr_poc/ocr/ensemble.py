"""ensemble.py — EasyOCR + PaddleOCR 앙상블."""
from typing import List, Tuple

import numpy as np

from .base import BaseOCR
from .easy_ocr import EasyOCREngine
from .paddle_ocr import PaddleOCREngine
from ..schemas import CharCrop, CharRecognition


class EnsembleOCR(BaseOCR):
    """두 OCR 엔진의 결과를 결합하여 신뢰도를 높이는 엔진.
    
    규칙:
        1. 둘 다 실패 → 실패
        2. 한쪽만 인식 → 그쪽 채택
        3. 두 결과 일치 → 신뢰도 가산 (boost)
        4. 두 결과 불일치 → 한글 자리는 prefer 엔진 우선,
                            그 외에는 신뢰도 높은 쪽
    """
    
    def __init__(self, config):
        super().__init__(config)
        self._easy = EasyOCREngine(config)
        self._paddle = PaddleOCREngine(config)
    
    @property
    def engine_name(self) -> str:
        return 'ensemble'
    
    def recognize_char(
        self,
        char_image: np.ndarray,
        char_type: str,
    ) -> Tuple[str, float]:
        e_text, e_conf = self._easy.recognize_char(char_image, char_type)
        p_text, p_conf = self._paddle.recognize_char(char_image, char_type)
        
        return self._merge(e_text, e_conf, p_text, p_conf, char_type)
    
    def recognize_all(self, char_crops: List[CharCrop]) -> List[CharRecognition]:
        """앙상블에서는 두 엔진 결과를 따로 받아 글자별로 결합."""
        easy_results = self._easy.recognize_all(char_crops)
        paddle_results = self._paddle.recognize_all(char_crops)
        
        merged = []
        for crop, e, p in zip(char_crops, easy_results, paddle_results):
            char, conf = self._merge(
                e.predicted, e.confidence,
                p.predicted, p.confidence,
                crop.char_type,
            )
            merged.append(CharRecognition(
                char_index=crop.index,
                char_type=crop.char_type,
                predicted=char,
                confidence=conf,
                raw_text=f'easy={e.predicted}|paddle={p.predicted}',
                engine='ensemble',
            ))
        return merged
    
    def _merge(self, e_text, e_conf, p_text, p_conf, char_type):
        """두 결과 병합 로직."""
        # Case 1: 둘 다 실패
        if not e_text and not p_text:
            return None, 0.0
        
        # Case 2: 한쪽만
        if not e_text:
            return p_text, p_conf
        if not p_text:
            return e_text, e_conf
        
        # Case 3: 일치 — 신뢰도 가산
        if e_text == p_text:
            boosted = min(1.0, max(e_conf, p_conf) * self.config.agreement_boost)
            return e_text, boosted
        
        # Case 4: 불일치
        if char_type == 'hangul':
            if self.config.prefer_for_hangul == 'paddleocr' and p_conf >= 0.3:
                return p_text, p_conf
            elif self.config.prefer_for_hangul == 'easyocr' and e_conf >= 0.3:
                return e_text, e_conf
        
        # 신뢰도 높은 쪽
        if e_conf >= p_conf:
            return e_text, e_conf
        return p_text, p_conf
