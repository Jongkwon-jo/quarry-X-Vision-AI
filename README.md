# ANPR PoC 파이프라인

석산 CCTV 환경에서 화물차 번호판을 인식하는 PoC 시스템.

## 디렉토리 구조

```
anpr_poc/
├── anpr_poc/                  # 패키지 루트
│   ├── __init__.py
│   ├── config.py              # 전역 설정 (임계값, 화이트리스트, 매핑)
│   ├── schemas.py             # 단계 간 데이터 객체
│   ├── pipeline.py            # 메인 파이프라인 (ANPRPipeline)
│   │
│   ├── detection/             # ① 번호판 검출 (YOLO)
│   │   └── plate_detector.py
│   ├── preprocessing/         # ② 정렬 + 화질 개선
│   │   └── preprocessor.py
│   ├── split/                 # ③ 글자별 분리 (Projection Profile)
│   │   └── splitter.py
│   ├── ocr/                   # ④ OCR 인식 (엔진 교체 가능)
│   │   ├── base.py
│   │   ├── easy_ocr.py
│   │   ├── paddle_ocr.py
│   │   └── ensemble.py
│   ├── validation/            # ⑤ 형식 검증 + 후처리
│   │   └── validator.py
│   └── utils/
│       ├── image_io.py        # 한글 경로 호환 입출력
│       └── geometry.py
│
├── scripts/
│   ├── predict_single.py      # 단일 이미지 추론
│   └── predict_batch.py       # 배치 평가
│
├── tests/                     # 단위 테스트
├── configs/                   # 설정 YAML (선택)
├── requirements.txt
└── README.md
```

## 설치

```bash
pip install -r requirements.txt

# PaddleOCR을 쓰려면 추가
pip install paddlepaddle paddleocr
# GPU 사용 시:
# pip install paddlepaddle-gpu paddleocr
```

## 실행

### 단일 이미지

```bash
python scripts/predict_single.py \
    --image data/test/sample.jpg \
    --weights runs/detect/train-29/weights/best.pt \
    --engine easyocr \
    --debug
```

### 배치 평가 (파일명에 정답 포함된 경우)

```bash
python scripts/predict_batch.py \
    --input data/test \
    --output runs/poc_eval \
    --engine ensemble
```

### Python 코드에서 직접 사용

```python
from anpr_poc.pipeline import ANPRPipeline
from anpr_poc.config import PipelineConfig

# 기본 설정
config = PipelineConfig.default()
config.detection.weights_path = 'runs/detect/train-29/weights/best.pt'
config.ocr.engine = 'ensemble'  # 'easyocr', 'paddleocr', 'ensemble'

pipeline = ANPRPipeline(config)
result = pipeline.run('truck.jpg')

print(result.summary())
# → ✓ "12가3456" (conf=0.873)

# 상세 정보
print(f'검출 bbox: {result.detection.bbox}')
print(f'분리된 글자 수: {len(result.char_crops)}')
for r in result.char_recognitions:
    print(f'  #{r.char_index} [{r.char_type}] → {r.predicted}')
```

## 모듈별 역할

| 모듈 | 입력 | 출력 | 핵심 기능 |
|------|------|------|----------|
| detection | 원본 이미지 | Detection (bbox + crop) | YOLO로 번호판 검출 |
| preprocessing | Detection.crop | PreprocessedPlate | 4점 보정 + CLAHE + Bilateral |
| split | rectified plate | List[CharCrop] | Projection Profile로 글자 분리 |
| ocr | List[CharCrop] | List[CharRecognition] | 자리별 화이트리스트 + 단일 글자 OCR |
| validation | List[CharRecognition] | (text, valid, conf) | 정규식 검증 + 후처리 매핑 |

## 설정 변경

`anpr_poc/config.py`의 dataclass들을 수정하거나, 코드에서 직접 인스턴스 속성을 변경:

```python
config = PipelineConfig.default()

# 검출 임계값을 낮춤
config.detection.conf_threshold = 0.3

# OCR 엔진 변경
config.ocr.engine = 'paddleocr'

# 분리 단계 valley 임계값 조정
config.split.threshold_ratio = 0.5

# 디버그 출력 활성화 (각 단계 중간 결과 저장)
config.debug = True
config.debug_dir = 'my_debug'
```

## 한글 OCR 향상 — 후처리 매핑 추가

`config.py`의 `HANGUL_FIX_MAP`에 자주 나오는 오인식 패턴을 추가하면
즉시 정확도가 오릅니다. 예를 들어 OCR이 "가"를 "카"로 자주 잡는다면:

```python
HANGUL_FIX_MAP = {
    ...
    '카': '가',    # 추가
    ...
}
```

## 강화 검증 (4 레이어)

OCR 결과를 단순 정규식만이 아닌 **한국 자동차 번호판 체계**에 따라 검증합니다.

| 레이어 | 검증 내용 | 잡아내는 오류 |
|--------|-----------|---------------|
| 1. 형식 | 정규식 매칭 | `12가345` (글자 수 부족) |
| 2. 차종 | 앞 숫자 범위 확인 | `054가1234` (비정상 숫자) |
| 3. 용도 | 한글이 등록된 용도 한글인지 | `850뽀1234` (한글 노이즈) |
| 4. 도메인 | 석산 환경 합리성 | `985라1234` (소방차는 석산에 없음) |

검증 결과는 3단계 신뢰 등급으로 분류됩니다.

- **PASS**: 모든 레이어 통과 → 자동 처리 가능
- **SUSPECT**: 일부 레이어 미통과 → 검수 큐로 보냄
- **FAIL**: 형식 자체 어긋남 → 자동 처리 불가

```python
from anpr_poc.pipeline import ANPRPipeline
from anpr_poc.config import PipelineConfig

config = PipelineConfig.default()
config.validation.use_enhanced = True
config.validation.require_quarry_compatible = True  # 석산 도메인 강제

pipeline = ANPRPipeline(config)
result = pipeline.run('truck.jpg')

# 검증 리포트 활용
report = result.validation_report
print(report.summary())          # ✓ PASS | "850가1234" | truck/private | conf=0.873
print(report.reasoning())        # 레이어별 통과·실패 사유
print(report.trust_level.value)  # 'pass' / 'suspect' / 'fail'
print(report.vehicle_class)      # VehicleClass.TRUCK
print(report.usage_type)         # UsageType.PRIVATE
```

### 한국 번호판 체계 참고

**차종 분류** (앞 숫자):
- `100~699`: 승용차
- `700~799`: 승합차
- `800~979`: 화물차 ← 석산 주 대상
- `980~997`: 특수차 (소방·구급)
- `998~999`: 긴급차 (경찰)

**용도 분류** (가운데 한글):
- 자가용/비사업용 (32자): 가나다라마, 거너더러머버서어저, 고노도로모보소오조, 구누두루무부수우주
- 사업용 (4자): 아바사자 (택시·버스·화물영업)
- 렌터카 (3자): 하허호
- 택배 (1자): 배

## PoC 성공 기준

- 화각별 ANPR 완전 일치 정확도 80% 이상 (정상 조건)
- 한글 자리 정확도 85% 이상
- 처리 속도 1장당 1초 이하 (GPU 기준)
