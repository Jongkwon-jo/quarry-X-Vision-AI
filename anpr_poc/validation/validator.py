"""validator.py — 번호판 형식 검증과 후처리.

OCR 결과를 한국 번호판 형식에 맞춰 검증·교정한다.
"""
import re
from typing import List, Tuple

from ..config import (
    DIGIT_FIX_MAP, HANGUL_FIX_MAP, PLATE_PATTERNS,
    ValidationConfig, PLATE_HANGUL,
)
from ..schemas import CharRecognition


class PlateValidator:
    """글자별 인식 결과 → 검증된 최종 번호판 텍스트."""
    
    def __init__(self, config: ValidationConfig):
        self.config = config
    
    def validate(
        self,
        recognitions: List[CharRecognition],
    ) -> Tuple[str, str, bool, float]:
        """글자별 인식 결과를 통합·검증.
        
        Returns:
            (raw_text, corrected_text, is_valid_format, overall_confidence)
        """
        # 1. 글자별 결과 결합 (실패한 자리는 ?)
        chars = []
        confs = []
        for rec in sorted(recognitions, key=lambda r: r.char_index):
            chars.append(rec.predicted if rec.predicted else '?')
            confs.append(rec.confidence)
        
        raw_text = ''.join(chars)
        avg_conf = sum(confs) / len(confs) if confs else 0.0
        
        # 2. 자리별 후처리 보정
        if self.config.auto_fix_digits or self.config.auto_fix_hangul:
            corrected = self._auto_fix(recognitions)
        else:
            corrected = raw_text
        
        # 3. 형식 검증
        cleaned = self._clean(corrected)
        is_valid = any(p.match(cleaned) for p in PLATE_PATTERNS)
        
        return raw_text, cleaned, is_valid, avg_conf
    
    def _auto_fix(self, recognitions: List[CharRecognition]) -> str:
        """자리 타입과 문자 매핑을 활용한 자동 보정."""
        chars = []
        for rec in sorted(recognitions, key=lambda r: r.char_index):
            char = rec.predicted
            if not char:
                chars.append('?')
                continue
            
            # 한글 자리에 한글이 아닌 글자 → 한글로 매핑 시도
            if rec.char_type == 'hangul' and char not in PLATE_HANGUL:
                if self.config.auto_fix_hangul and char in HANGUL_FIX_MAP:
                    char = HANGUL_FIX_MAP[char]
            
            # 숫자 자리에 숫자가 아닌 글자 → 숫자로 매핑 시도
            elif rec.char_type == 'digit' and not char.isdigit():
                if self.config.auto_fix_digits and char in DIGIT_FIX_MAP:
                    char = DIGIT_FIX_MAP[char]
            
            chars.append(char)
        return ''.join(chars)
    
    @staticmethod
    def _clean(text: str) -> str:
        """공백·특수문자 제거 후 대문자화."""
        return re.sub(r'[^0-9A-Z가-힣]', '', text.upper())
