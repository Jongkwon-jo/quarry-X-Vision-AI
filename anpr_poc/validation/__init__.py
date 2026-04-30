"""validation — 형식 검증과 후처리 보정 + 도메인 검증."""
from .validator import PlateValidator
from .enhanced_validator import (
    EnhancedPlateValidator,
    ValidationReport,
    TrustLevel,
)
from .plate_codes import (
    VehicleClass, UsageType,
    classify_vehicle_number, classify_usage_hangul,
    is_valid_hangul, is_quarry_compatible,
    PRIVATE_HANGUL, COMMERCIAL_HANGUL, RENTAL_HANGUL, DELIVERY_HANGUL,
)

__all__ = [
    'PlateValidator',
    'EnhancedPlateValidator', 'ValidationReport', 'TrustLevel',
    'VehicleClass', 'UsageType',
    'classify_vehicle_number', 'classify_usage_hangul',
    'is_valid_hangul', 'is_quarry_compatible',
    'PRIVATE_HANGUL', 'COMMERCIAL_HANGUL', 'RENTAL_HANGUL', 'DELIVERY_HANGUL',
]
