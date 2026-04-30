"""easy_ocr.py — EasyOCR 엔진 래퍼."""
from typing import Tuple

import numpy as np

from .base import BaseOCR


class EasyOCREngine(BaseOCR):
    """EasyOCR을 활용한 단일 글자 인식기."""
    
    def __init__(self, config):
        super().__init__(config)
        self._reader = None
    
    @property
    def engine_name(self) -> str:
        return 'easyocr'
    
    @property
    def reader(self):
        """첫 호출 시 lazy load."""
        if self._reader is None:
            import easyocr
            self._reader = easyocr.Reader(
                ['ko', 'en'],
                gpu=self.config.use_gpu,
                verbose=False,
            )
        return self._reader
    
    def recognize_char(
        self,
        char_image: np.ndarray,
        char_type: str,
    ) -> Tuple[str, float]:
        if char_image is None or char_image.size == 0:
            return None, 0.0
        
        # 단일 글자는 작아서 확대 필요
        enlarged = self.upscale(char_image, self.config.char_scale)
        
        # 자리에 맞는 화이트리스트
        allowlist = self.get_allowlist(char_type)
        
        try:
            results = self.reader.readtext(
                enlarged,
                allowlist=allowlist,
                detail=1,
                paragraph=False,
                text_threshold=self.config.text_threshold,
                low_text=self.config.low_text,
                link_threshold=self.config.link_threshold,
                canvas_size=self.config.canvas_size,
            )
        except Exception:
            return None, 0.0
        
        if not results:
            return None, 0.0
        
        # 신뢰도 최대값
        best = max(results, key=lambda r: r[2])
        text = best[1].strip()
        return text if text else None, float(best[2])
