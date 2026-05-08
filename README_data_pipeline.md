# 데이터 수집·처리 파이프라인

석산 화물차 인식을 위한 8단계 데이터 워크플로 패키지.
`robotflow.ipynb`의 분산된 코드를 모듈화하여 재사용성과 보안성을 강화한 버전.

## 디렉토리 구조

```
data_pipeline/
├── data_pipeline/                 # 패키지 루트
│   ├── __init__.py
│   ├── config.py                  # 전역 설정 (.env 분리 지원)
│   ├── schemas.py                 # 단계 간 데이터 객체
│   ├── pipeline.py                # 메인 파이프라인 (DataPipeline)
│   │
│   ├── collection/                # ① 수집 폴더 스캔
│   │   └── collector.py
│   ├── filter/                    # ②③ 필터링 + Dataset 복사
│   │   └── filter_copy.py
│   ├── upload/                    # ④ Roboflow 업로드
│   │   └── roboflow_uploader.py
│   ├── download/                  # ⑥ 다운로드 + 원본 교체
│   │   ├── roboflow_downloader.py
│   │   └── image_replacer.py
│   ├── sync/                      # ⑦ FTP 전송
│   │   └── ftp_transfer.py
│   ├── training/                  # ⑧ SSH 학습 트리거
│   │   └── ssh_trigger.py
│   └── utils/
│       └── logging.py
│
├── scripts/
│   └── run_pipeline.py            # CLI 진입점
├── configs/
│   └── .env.template              # 환경변수 템플릿
├── requirements.txt
└── README.md
```

## 설치

```bash
pip install -r requirements.txt
```

## 환경변수 설정 (보안)

기존 노트북은 API 키와 비밀번호가 하드코딩되어 있었습니다.
이 패키지는 환경변수로 분리합니다.

```bash
cp configs/.env.template .env
# .env 파일을 열어 값 입력
# .gitignore에 .env가 포함되어 있는지 확인
```

또는 시스템 환경변수로 직접 등록:

```bash
export ROBOFLOW_API_KEY="..."
export FTP_PASSWORD="..."
export SSH_PASSWORD="..."
```

## 실행

### 전체 파이프라인 (라벨링 대기 포함)

```bash
python scripts/run_pipeline.py --full --source-subdir 20260428
```

⑤ 라벨링 단계에서 일시 정지하여 사용자가 Roboflow 웹 UI에서
라벨링을 완료한 후 Enter를 누르면 다음 단계로 진행합니다.

### 단계별 실행

```bash
# 1차 사이클: 수집 → 업로드만
python scripts/run_pipeline.py --stage upload

# 2차 사이클: 다운로드 + 원본 교체 + FTP 전송 + 학습
python scripts/run_pipeline.py --stage training --rf-version 3
```

### Python 코드에서 직접 사용

```python
from data_pipeline.pipeline import DataPipeline

# 환경변수 로드
pipeline = DataPipeline.from_env()

# 옵션 오버라이드
pipeline.config.collection.source_subdir = '20260428'
pipeline.config.roboflow.download_version = 3

# 단계별 실행
collection = pipeline.run_collection()
filtered   = pipeline.run_filter(collection)
upload     = pipeline.run_upload(filtered, granular=True)

print(f'중복률: {upload.duplicate_rate*100:.1f}%')
```

## 8단계 흐름 요약

| # | 모듈 | 클래스 | 역할 |
|---|------|--------|------|
| ① | collection | ImageCollector | TeamViewer 수집 폴더 스캔 |
| ② | filter | FilterAndCopy | 'front' 키워드 필터링 |
| ③ | filter | FilterAndCopy | dataset 폴더로 복사 (메타데이터 보존) |
| ④ | upload | RoboflowUploader | Roboflow 클라우드 업로드 (UPLOADED/DUPLICATE 분류) |
| ⑤ | (수동) | — | Roboflow 웹 UI 라벨링 |
| ⑥ | download | RoboflowDownloader | YOLO 형식 데이터셋 다운로드 |
|   | download | ImageReplacer | Roboflow 압축 이미지 → 원본 무손실 교체 |
| ⑦ | sync | FTPTransfer | 학습 서버로 FTP 전송 (cp949 한글 호환) |
| ⑧ | training | SSHTrainingTrigger | paramiko + PowerShell로 train.py 실행 |

## 노트북 대비 개선 사항

| 항목 | 노트북 | 본 패키지 |
|------|--------|-----------|
| 비밀값 관리 | 평문 하드코딩 | 환경변수 분리 |
| 단계별 실행 | 셀 순차 실행 | CLI 또는 코드에서 단계별 호출 |
| 재사용성 | 일회성 | 다음 사이클에 그대로 재사용 가능 |
| 결과 추적 | print 출력 | 구조화된 dataclass 결과 |
| 오류 처리 | 즉시 중단 | 단계별 실패 분리 처리 |
| 로깅 | print 산발 | 통일된 로거 (시간·레벨 포맷) |
| 원본 교체 | 셀 4 분리 | ⑥ 다운로드 단계에 통합 |
| 라벨 동기화 | 셀 4 내 | ImageReplacer 메서드로 모듈화 |

## 추가 고려사항

**라벨링 자동화**: 파일명에 차량번호가 이미 포함된 점을 활용해
Roboflow Pre-annotation API 호출 모듈을 추가하면 ⑤ 단계 자동화 가능.

**작업 큐 전환**: 현재 SSH 트리거는 동기식이므로 학습 시간 동안 클라이언트가
연결 유지 필요. 본 시스템에선 REST API + 작업 큐 방식으로 전환 검토.

**중복 다양성 관리**: 노트북 분석에서 발견된 ~35% 중복률을 줄이려면
업로드 전 로컬 해시 비교로 중복을 사전 필터링하는 단계 추가 가능.
