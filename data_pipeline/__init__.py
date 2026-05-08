"""
data_pipeline — 석산 화물차 데이터 수집·처리 파이프라인

robotflow.ipynb의 8단계 워크플로를 모듈화한 패키지.
ANPR PoC 패키지(anpr_poc)와 동일한 설계 패턴을 따른다.

8 stages:
    ① collection   — TeamViewer 수집된 이미지 리스트업
    ② filter       — 'front' 키워드 필터링
    ③ upload       — Roboflow 업로드 (UPLOADED/DUPLICATE 분류)
    ④ (수동 라벨링)
    ⑤ download     — YOLO 데이터셋 다운로드
    ⑥ replace      — 원본 무손실 이미지로 교체 + 라벨 동기화
    ⑦ sync         — FTP로 학습 서버 전송
    ⑧ training     — SSH 원격 학습 트리거
"""

__version__ = '0.1.0'
