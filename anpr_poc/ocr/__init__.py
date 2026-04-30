"""ocr — 글자별 OCR 인식.

엔진을 교체할 수 있도록 추상 베이스 클래스 + 팩토리 패턴 사용.
"""
from ..config import OCRConfig
from .base import BaseOCR
from .easy_ocr import EasyOCREngine
from .paddle_ocr import PaddleOCREngine
from .ensemble import EnsembleOCR


def create_ocr_engine(config: OCRConfig) -> BaseOCR:
    """설정에 따라 적절한 OCR 엔진 생성."""
    engine_name = config.engine.lower()
    
    if engine_name == 'easyocr':
        return EasyOCREngine(config)
    elif engine_name == 'paddleocr':
        return PaddleOCREngine(config)
    elif engine_name == 'ensemble':
        return EnsembleOCR(config)
    else:
        raise ValueError(f'알 수 없는 OCR 엔진: {config.engine}')


__all__ = ['BaseOCR', 'EasyOCREngine', 'PaddleOCREngine',
           'EnsembleOCR', 'create_ocr_engine']
