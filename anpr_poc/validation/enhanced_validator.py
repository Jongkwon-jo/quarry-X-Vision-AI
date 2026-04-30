"""enhanced_validator.py — 4단계 누적 검증을 수행하는 강화 검증기.

기존 PlateValidator를 상속받아 다음 단계를 추가:
    Layer 1: 형식 검증 (정규식, 글자 수)
    Layer 2: 차종 검증 (앞 숫자 범위)
    Layer 3: 용도 검증 (한글이 등록된 용도 한글인지)
    Layer 4: 도메인 검증 (석산 환경에서 합리적인 차량인지)

각 레이어 통과 여부에 따라 신뢰 등급 (PASS/SUSPECT/FAIL) 부여.
"""
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple

from ..config import PLATE_PATTERNS, ValidationConfig
from ..schemas import CharRecognition
from .plate_codes import (
    VehicleClass, UsageType,
    classify_vehicle_number, classify_usage_hangul,
    is_valid_hangul, is_quarry_compatible,
)
from .validator import PlateValidator


class TrustLevel(Enum):
    """검증 후 신뢰 등급."""
    PASS = 'pass'           # 모든 레이어 통과 — 자동 처리 가능
    SUSPECT = 'suspect'     # 일부 레이어 미통과 — 검수 큐로 보냄
    FAIL = 'fail'           # 형식 자체가 어긋남 — 자동 처리 불가


@dataclass
class ValidationReport:
    """검증 결과 상세 리포트."""
    plate_text: str
    trust_level: TrustLevel
    is_valid_format: bool                  # Layer 1
    vehicle_class: VehicleClass = VehicleClass.UNKNOWN  # Layer 2
    vehicle_number: Optional[int] = None
    usage_type: UsageType = UsageType.UNKNOWN          # Layer 3
    hangul: Optional[str] = None
    is_quarry_compatible: bool = False     # Layer 4
    
    # 디버깅 정보
    layer_passes: List[str] = field(default_factory=list)
    layer_failures: List[str] = field(default_factory=list)
    overall_confidence: float = 0.0
    
    @property
    def passed_count(self) -> int:
        return len(self.layer_passes)
    
    def reasoning(self) -> str:
        """사람이 읽을 수 있는 검증 사유."""
        lines = []
        for ok in self.layer_passes:
            lines.append(f'  ✓ {ok}')
        for fail in self.layer_failures:
            lines.append(f'  ✗ {fail}')
        return '\n'.join(lines)
    
    def summary(self) -> str:
        emoji = {'pass': '✓', 'suspect': '?', 'fail': '✗'}[self.trust_level.value]
        v_str = self.vehicle_class.value
        u_str = self.usage_type.value
        return (f'{emoji} {self.trust_level.value.upper()} | "{self.plate_text}" | '
                f'{v_str}/{u_str} | conf={self.overall_confidence:.3f}')


class EnhancedPlateValidator(PlateValidator):
    """4단계 누적 검증을 수행하는 강화 검증기.
    
    PlateValidator의 검증·보정 로직을 그대로 사용하면서,
    체계적 도메인 지식 검증을 누적 적용한다.
    """
    
    def __init__(
        self,
        config: ValidationConfig,
        require_quarry_compatible: bool = False,
    ):
        """
        Args:
            config: 기본 검증 설정 (auto_fix_* 등)
            require_quarry_compatible: True면 석산 부적합 차량을 SUSPECT로
                강등. False면 도메인 검증은 정보 제공용으로만 사용.
        """
        super().__init__(config)
        self.require_quarry_compatible = require_quarry_compatible
    
    def validate_full(
        self,
        recognitions: List[CharRecognition],
    ) -> ValidationReport:
        """4단계 검증을 모두 거친 상세 리포트 반환.
        
        기존 validate()는 (text, valid, conf) 튜플을 반환하지만,
        이 메서드는 검증의 모든 디테일을 담은 ValidationReport를 반환한다.
        """
        # 부모 클래스의 1단계 검증·보정 활용
        raw_text, corrected, is_valid_format, avg_conf = self.validate(recognitions)
        
        report = ValidationReport(
            plate_text=corrected,
            trust_level=TrustLevel.FAIL,  # 기본값, 아래에서 갱신
            is_valid_format=is_valid_format,
            overall_confidence=avg_conf,
        )
        
        # ===== Layer 1: 형식 검증 =====
        if not is_valid_format:
            report.layer_failures.append(
                f'형식 불일치: "{corrected}" (정규식 통과 실패)'
            )
            report.trust_level = TrustLevel.FAIL
            return report
        report.layer_passes.append(f'형식 일치: 정규식 통과')
        
        # 번호판 구성 분해
        parsed = self._parse_plate(corrected)
        if parsed is None:
            report.layer_failures.append('번호판 구조 파싱 실패')
            report.trust_level = TrustLevel.FAIL
            return report
        
        v_num, hangul, _, num_digits = parsed
        report.vehicle_number = v_num
        report.hangul = hangul
        
        # ===== Layer 2: 차종 숫자 검증 =====
        v_class = classify_vehicle_number(v_num, num_digits=num_digits)
        report.vehicle_class = v_class
        
        if v_class == VehicleClass.UNKNOWN:
            report.layer_failures.append(
                f'차종 숫자 비정상: {v_num} (정의된 범위 밖)'
            )
            report.trust_level = TrustLevel.SUSPECT
            return report
        report.layer_passes.append(
            f'차종 일치: {v_num} → {v_class.value}'
        )
        
        # ===== Layer 3: 용도 한글 검증 =====
        usage = classify_usage_hangul(hangul)
        report.usage_type = usage
        
        if usage == UsageType.UNKNOWN:
            report.layer_failures.append(
                f'용도 한글 비정상: "{hangul}" (등록되지 않은 한글)'
            )
            report.trust_level = TrustLevel.SUSPECT
            return report
        report.layer_passes.append(
            f'용도 일치: "{hangul}" → {usage.value}'
        )
        
        # ===== Layer 4: 도메인 (석산) 검증 =====
        compat = is_quarry_compatible(v_class, usage)
        report.is_quarry_compatible = compat
        
        if not compat:
            msg = f'석산 부적합: {v_class.value}/{usage.value}'
            if self.require_quarry_compatible:
                report.layer_failures.append(msg)
                report.trust_level = TrustLevel.SUSPECT
                return report
            else:
                # 정보 제공용 — 통과로 간주하되 메모
                report.layer_passes.append(
                    f'도메인 검증: 통과 (강제 모드 아님)'
                )
        else:
            report.layer_passes.append(
                f'도메인 일치: 석산 차량 분류'
            )
        
        # 모든 레이어 통과
        report.trust_level = TrustLevel.PASS
        return report
    
    @staticmethod
    def _parse_plate(plate_text: str) -> Optional[Tuple[int, str, str, int]]:
        """번호판 텍스트 → (차종숫자, 한글, 일련번호, 자릿수) 분해.
        
        7자리(12가3456) 또는 8자리(123가4567) 형식 지원.
        """
        # 한글 위치 찾기 (정확히 1개여야 정상)
        match = re.match(r'^(\d{2,3})([가-힣])(\d{4})$', plate_text)
        if not match:
            return None
        
        v_num_str, hangul, serial = match.groups()
        try:
            return int(v_num_str), hangul, serial, len(v_num_str)
        except ValueError:
            return None
