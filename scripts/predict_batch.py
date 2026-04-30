"""
predict_batch.py — 폴더의 모든 이미지에 대해 일괄 추론 + 평가.

파일명 패턴: front_{ts}_{차량번호}.{ext} 형식이면 정답을 자동 추출하여
정확도 평가도 함께 수행.

사용 예:
    python scripts/predict_batch.py --input data/test --output runs/poc_eval
    python scripts/predict_batch.py --input data/test --engine ensemble
"""
import argparse
import re
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from anpr_poc.config import PipelineConfig
from anpr_poc.pipeline import ANPRPipeline


def extract_gt_from_filename(filename: str):
    """파일명에서 정답 번호판 추출.
    
    예: 'front_1777329632545_대구06라5245.BMP' → '대구06라5245'
    """
    stem = Path(filename).stem
    parts = stem.rsplit('_', 1)
    if len(parts) == 2:
        gt = parts[1]
        # 구형 지역명 제거 시도 (PoC 단계에선 그대로 유지)
        return gt
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='입력 이미지 폴더')
    parser.add_argument('--output', default='runs/batch_eval',
                        help='출력 디렉토리')
    parser.add_argument('--weights', default='runs/detect/train-29/weights/best.pt')
    parser.add_argument('--engine', default='easyocr',
                        choices=['easyocr', 'paddleocr', 'ensemble'])
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--conf', type=float, default=0.5)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--limit', type=int, default=None,
                        help='최대 처리 이미지 수 (디버깅용)')
    args = parser.parse_args()
    
    # 입력 이미지 수집
    input_dir = Path(args.input)
    images = []
    for ext in ('*.jpg', '*.jpeg', '*.bmp', '*.png'):
        images.extend(input_dir.rglob(ext))
        images.extend(input_dir.rglob(ext.upper()))
    images = sorted(set(images))
    if args.limit:
        images = images[:args.limit]
    
    print(f'발견된 이미지: {len(images)}장')
    if not images:
        return
    
    # 출력 디렉토리
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 파이프라인 구성
    config = PipelineConfig.default()
    config.detection.weights_path = args.weights
    config.detection.device = args.device
    config.detection.conf_threshold = args.conf
    config.ocr.engine = args.engine
    config.ocr.use_gpu = (args.device != 'cpu')
    config.debug = args.debug
    config.debug_dir = str(output_dir / 'debug')
    
    pipeline = ANPRPipeline(config)
    
    # 일괄 처리
    rows = []
    for img_path in tqdm(images, desc='Processing'):
        gt = extract_gt_from_filename(img_path.name)
        result = pipeline.run(img_path)
        
        rows.append({
            'filename': img_path.name,
            'gt': gt,
            'predicted': result.corrected_text,
            'raw': result.raw_text,
            'is_valid_format': result.is_valid_format,
            'overall_confidence': result.overall_confidence,
            'detection_conf': result.detection.confidence if result.detection else None,
            'n_chars_split': len(result.char_crops),
            'error_stage': result.error_stage,
            'error_message': result.error_message,
            'exact_match': (gt is not None and result.corrected_text == gt),
            'success': result.success,
        })
    
    df = pd.DataFrame(rows)
    
    # 결과 저장
    df.to_csv(output_dir / 'detail.csv', index=False, encoding='utf-8-sig')
    
    # 요약 통계
    print('\n' + '=' * 60)
    print('배치 평가 결과')
    print('=' * 60)
    print(f'전체 이미지        : {len(df)}')
    print(f'검출 성공          : {df["detection_conf"].notna().sum()} '
          f'({df["detection_conf"].notna().mean()*100:.1f}%)')
    print(f'형식 유효율        : {df["is_valid_format"].sum()} '
          f'({df["is_valid_format"].mean()*100:.1f}%)')
    
    has_gt = df['gt'].notna()
    if has_gt.any():
        gt_df = df[has_gt]
        print(f'\n정답 비교 가능     : {len(gt_df)}장')
        print(f'완전 일치 정확도   : {gt_df["exact_match"].sum()} '
              f'({gt_df["exact_match"].mean()*100:.1f}%)')
    
    # 단계별 실패 통계
    failed = df[df['error_stage'].notna()]
    if len(failed) > 0:
        print(f'\n단계별 실패:')
        print(failed['error_stage'].value_counts().to_string())
    
    print(f'\n결과 저장: {output_dir / "detail.csv"}')


if __name__ == '__main__':
    main()
