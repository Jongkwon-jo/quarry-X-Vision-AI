"""
run_pipeline.py — 데이터 파이프라인 실행 스크립트.

사용 예 (전체 자동):
    python scripts/run_pipeline.py --full

사용 예 (단계별):
    python scripts/run_pipeline.py --stage upload      # ①②③④까지
    python scripts/run_pipeline.py --stage download    # ⑥⑦까지 (다운로드+원본교체)
    python scripts/run_pipeline.py --stage sync        # ⑦ FTP 전송만
    python scripts/run_pipeline.py --stage training    # ⑧ 학습만
    
환경변수:
    ROBOFLOW_API_KEY   (필수, upload/download 단계)
    FTP_PASSWORD       (필수, sync 단계)
    SSH_PASSWORD       (필수, training 단계)
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from data_pipeline.pipeline import DataPipeline


def main():
    parser = argparse.ArgumentParser(description='석산 데이터 수집·학습 파이프라인')
    
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--full', action='store_true',
                      help='전체 파이프라인 실행 (라벨링 대기 포함)')
    mode.add_argument('--stage', choices=['upload', 'download', 'sync', 'training'],
                      help='특정 단계까지만 실행')
    
    parser.add_argument('--source-subdir', help='수집 회차 폴더명 (예: 20260428)')
    parser.add_argument('--rf-version', type=int, help='Roboflow 데이터셋 버전')
    parser.add_argument('--granular-upload', action='store_true',
                        help='파일별 개별 업로드 (정확한 통계)')
    parser.add_argument('--skip-labeling-wait', action='store_true',
                        help='라벨링 대기 건너뛰기 (이미 라벨링됨)')
    args = parser.parse_args()
    
    # 환경변수에서 비밀값 로드
    pipeline = DataPipeline.from_env()
    
    # CLI 옵션으로 설정 오버라이드
    if args.source_subdir:
        pipeline.config.collection.source_subdir = args.source_subdir
    if args.rf_version:
        pipeline.config.roboflow.download_version = args.rf_version
    
    # 실행
    if args.full:
        pipeline.run_full(
            skip_labeling_wait=args.skip_labeling_wait,
            granular_upload=args.granular_upload,
        )
    else:
        # 단계별 실행
        collection = pipeline.run_collection()
        
        if args.stage == 'upload':
            filtered = pipeline.run_filter(collection)
            pipeline.run_upload(filtered, granular=args.granular_upload)
        
        elif args.stage == 'download':
            download = pipeline.run_download()
            pipeline.run_replace(download, collection.source_path)
        
        elif args.stage == 'sync':
            download = pipeline.run_download()
            pipeline.run_replace(download, collection.source_path)
            pipeline.run_sync(download)
        
        elif args.stage == 'training':
            pipeline.run_training()

def main2():
    import os
    from dotenv import load_dotenv

    # .env 파일 로드
    env_path = Path("/config/.env")
    load_dotenv(dotenv_path=env_path)

    # 환경변수에서 비밀값 로드
    pipeline = DataPipeline.from_env()
    
    source_dir = r'C:\Users\User\Desktop\Project\VISION_AI\hdi'
    for dir in os.listdir(source_dir):
        pipeline.config.collection.source_subdir = os.path.join(source_dir, dir)
        collection = pipeline.run_collection()
        filtered = pipeline.run_filter(collection)    


if __name__ == '__main__':
    main2()