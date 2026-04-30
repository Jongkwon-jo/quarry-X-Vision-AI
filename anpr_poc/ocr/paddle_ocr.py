"""paddle_ocr.py — PaddleOCR 엔진 래퍼."""
from typing import Tuple

import numpy as np

from .base import BaseOCR


class PaddleOCREngine(BaseOCR):
    """PaddleOCR을 활용한 단일 글자 인식기.
    
    PaddleOCR은 화이트리스트 옵션이 직접 지원되지 않으므로
    인식 후 후처리에서 필터링한다.
    """
    
    def __init__(self, config):
        super().__init__(config)
        self._ocr = None
    
    @property
    def engine_name(self) -> str:
        return 'paddleocr'
    
    @property
    def ocr(self):
        """첫 호출 시 lazy load."""
        if self._ocr is None:
            from paddleocr import PaddleOCR
            self._ocr = PaddleOCR(
                use_angle_cls=False,
                lang='korean',
                show_log=False,
                use_gpu=self.config.use_gpu,
                det=False,  # 검출 단계 끄고 인식만
            )
        return self._ocr
    
    def recognize_char(
        self,
        char_image: np.ndarray,
        char_type: str,
    ) -> Tuple[str, float]:
        if char_image is None or char_image.size == 0:
            return None, 0.0
        
        enlarged = self.upscale(char_image, self.config.char_scale)
        
        try:
            result = self.ocr.ocr(enlarged, det=False, cls=False)
        except Exception:
            return None, 0.0
        
        if not result or not result[0]:
            return None, 0.0
        
        # PaddleOCR 결과 형식 호환 (버전별 차이 처리)
        first = result[0]
        text, conf = self._parse_result(first)
        
        if not text:
            return None, 0.0
        return text, float(conf)
    
    @staticmethod
    def _parse_result(item):
        """PaddleOCR 결과를 (text, conf)로 파싱.
        
        버전에 따라 형식이 다음 중 하나:
            - [(text, conf), ...]
            - [text, conf]
            - (text, conf)
        """
        try:
            if isinstance(item, (list, tuple)) and len(item) > 0:
                first = item[0]
                if isinstance(first, (list, tuple)) and len(first) >= 2:
                    return str(first[0]), float(first[1])
                # [text, conf] 형식
                if len(item) >= 2:
                    return str(item[0]), float(item[1])
            return str(item), 0.0
        except Exception:
            return None, 0.0
