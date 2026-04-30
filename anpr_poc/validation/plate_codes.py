"""plate_codes.py — 한국 자동차 번호판 체계 상수.

자료 출처: 대한민국 자동차관리법 시행규칙 + 관련 블로그/위키 종합.

자동차 번호판은 [차종숫자][한글][일련번호]로 구성되며,
차종 숫자와 용도 한글은 자동차 종류 및 사용 용도에 따라 사전 정의된
범위/집합에서 부여된다. OCR 결과가 이 체계에 부합하지 않으면 인식 오류일
가능성이 높다.
"""
from enum import Enum
from typing import Optional


class VehicleClass(Enum):
    """차종 분류 (앞 숫자 기반)."""
    PASSENGER = 'passenger'    # 승용차
    SUV_BUS = 'suv_bus'        # 승합차 (SUV 포함)
    TRUCK = 'truck'            # 화물차
    SPECIAL = 'special'        # 특수차 (소방·구급 등)
    EMERGENCY = 'emergency'    # 긴급차 (경찰·구조)
    UNKNOWN = 'unknown'


# 차종 숫자 범위 (현행 3자리 기준; 2자리는 100단위 비례 변환)
# 출처: 자동차관리법 시행규칙 제18조의2
VEHICLE_NUMBER_RANGES = {
    VehicleClass.PASSENGER: [(100, 699)],
    VehicleClass.SUV_BUS:   [(700, 799)],
    VehicleClass.TRUCK:     [(800, 979)],
    VehicleClass.SPECIAL:   [(980, 997)],
    VehicleClass.EMERGENCY: [(998, 999)],
}

# 2자리 형식(구형) 차종 범위 — 1996~2003 + 2004~2019 통용
VEHICLE_NUMBER_RANGES_LEGACY = {
    VehicleClass.PASSENGER: [(1, 69)],
    VehicleClass.SUV_BUS:   [(70, 79)],
    VehicleClass.TRUCK:     [(80, 97)],
    VehicleClass.SPECIAL:   [(98, 99)],
}


class UsageType(Enum):
    """용도 분류 (가운데 한글 기반)."""
    PRIVATE = 'private'         # 자가용 / 비사업용
    COMMERCIAL = 'commercial'   # 일반 사업용 (택시·버스·화물영업)
    RENTAL = 'rental'           # 렌터카
    DELIVERY = 'delivery'       # 택배
    UNKNOWN = 'unknown'


# 자가용/비사업용 한글 (32자)
PRIVATE_HANGUL = set(
    '가나다라마'
    '거너더러머버서어저'
    '고노도로모보소오조'
    '구누두루무부수우주'
)

# 사업용 한글: 택시, 버스, 화물영업 (4자)
COMMERCIAL_HANGUL = set('아바사자')

# 렌터카 한글 (3자)
RENTAL_HANGUL = set('하허호')

# 택배 한글 (1자)
DELIVERY_HANGUL = set('배')

# 모든 유효 한글 (44자)
ALL_VALID_HANGUL = (
    PRIVATE_HANGUL | COMMERCIAL_HANGUL | RENTAL_HANGUL | DELIVERY_HANGUL
)


def classify_vehicle_number(number: int, num_digits: int = None) -> VehicleClass:
    """차종 숫자 → 차종 분류.
    
    Args:
        number: 차종 숫자 (정수)
        num_digits: 원본 자릿수 (2 또는 3). None이면 number 크기로 추정.
    
    Returns:
        VehicleClass. 자릿수와 범위가 맞지 않으면 UNKNOWN.
    
    참고:
        - 3자리 형식(2019.9~ 현행): 100~999
        - 2자리 형식(1996~2019): 1~99 (단, 보통 10~99 범위에서 발급)
        - 따라서 3자리인데 100 미만이면(예: '054') 비정상 — UNKNOWN
        - 2자리인데 9 이하면 비정상 — UNKNOWN
    """
    # 자릿수 추정
    if num_digits is None:
        num_digits = 3 if number >= 100 else 2
    
    # 자릿수와 number의 크기 정합성 확인
    if num_digits == 3:
        if number < 100:
            # '054' 같은 비정상 (앞에 0이 있다는 건 OCR 오류 가능성)
            return VehicleClass.UNKNOWN
        ranges = VEHICLE_NUMBER_RANGES
    elif num_digits == 2:
        if number < 10 or number > 99:
            return VehicleClass.UNKNOWN
        ranges = VEHICLE_NUMBER_RANGES_LEGACY
    else:
        return VehicleClass.UNKNOWN
    
    for v_class, range_list in ranges.items():
        for start, end in range_list:
            if start <= number <= end:
                return v_class
    return VehicleClass.UNKNOWN


def classify_usage_hangul(hangul: str) -> UsageType:
    """한글 한 글자 → 용도 분류."""
    if not hangul or len(hangul) != 1:
        return UsageType.UNKNOWN
    if hangul in PRIVATE_HANGUL:
        return UsageType.PRIVATE
    if hangul in COMMERCIAL_HANGUL:
        return UsageType.COMMERCIAL
    if hangul in RENTAL_HANGUL:
        return UsageType.RENTAL
    if hangul in DELIVERY_HANGUL:
        return UsageType.DELIVERY
    return UsageType.UNKNOWN


def is_valid_hangul(hangul: str) -> bool:
    """번호판에 쓰이는 한글인지 확인."""
    return hangul in ALL_VALID_HANGUL


def is_quarry_compatible(v_class: VehicleClass, usage: UsageType) -> bool:
    """석산 운영장에 등장할 만한 차량인지 도메인 검증.
    
    석산은 화물차 위주이지만 외부 방문 차량(승용차)도 가끔 있으므로
    화물차/승합차/승용차까지 허용. 특수·긴급차는 비정상으로 분류.
    """
    allowed_classes = {
        VehicleClass.TRUCK,
        VehicleClass.SUV_BUS,    # 일부 덤프 변형
        VehicleClass.PASSENGER,  # 외부 방문 차량
    }
    if v_class not in allowed_classes:
        return False
    
    # 용도 측면: UNKNOWN(매핑 안 된 한글)은 거의 OCR 오류
    if usage == UsageType.UNKNOWN:
        return False
    
    return True
