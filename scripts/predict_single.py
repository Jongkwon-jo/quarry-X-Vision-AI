"""
predict_single.py — 단일 이미지에 대해 ANPR 파이프라인 실행.

사용 예:
    python scripts/predict_single.py --image path/to/plate.jpg
    python scripts/predict_single.py --image path/to/plate.jpg --debug
    python scripts/predict_single.py --image path/to/plate.jpg --engine paddleocr
"""
import argparse
import sys
from pathlib import Path

# 패키지 루트 경로 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

from anpr_poc.config import PipelineConfig
from anpr_poc.pipeline import ANPRPipeline


def main():
    parser = argparse.ArgumentParser(description='ANPR 단일 이미지 추론')
    parser.add_argument('--image', required=True, help='입력 이미지 경로')
    parser.add_argument('--weights', default='runs/detect/train-29/weights/best.pt',
                        help='YOLO 가중치 경로')
    parser.add_argument('--engine', default='easyocr',
                        choices=['easyocr', 'paddleocr', 'ensemble'],
                        help='OCR 엔진 선택')
    parser.add_argument('--device', default='cuda', help='cuda 또는 cpu')
    parser.add_argument('--conf', type=float, default=0.5, help='검출 임계값')
    parser.add_argument('--debug', action='store_true', help='중간 결과 저장')
    args = parser.parse_args()
    
    # 설정 구성
    config = PipelineConfig.default()
    config.detection.weights_path = args.weights
    config.detection.device = args.device
    config.detection.conf_threshold = args.conf
    config.ocr.engine = args.engine
    config.ocr.use_gpu = (args.device != 'cpu')
    config.debug = args.debug
    
    # 파이프라인 실행
    pipeline = ANPRPipeline(config)
    result = pipeline.run(args.image)
    
    # 결과 출력
    print('\n' + '=' * 60)
    print(f'입력: {args.image}')
    print(f'결과: {result.summary()}')
    print('=' * 60)
    
    if result.detection:
        print(f'\n[검출] bbox={result.detection.bbox}, '
              f'conf={result.detection.confidence:.3f}')
    
    if result.preprocessed:
        print(f'[전처리] 4점 보정 {"성공" if result.preprocessed.rectification_success else "실패(리사이즈)"}')
    
    if result.char_crops:
        print(f'[분리] {len(result.char_crops)}개 글자')
        for c in result.char_crops:
            print(f'  #{c.index} [{c.char_type}] x={c.x_range} w={c.width}')
    
    if result.char_recognitions:
        print(f'[OCR] 엔진={result.char_recognitions[0].engine}')
        for r in result.char_recognitions:
            print(f'  #{r.char_index} [{r.char_type}] '
                  f'→ "{r.predicted or "?"}" (conf={r.confidence:.3f})')
    
    if result.corrected_text:
        print(f'\n[최종] raw="{result.raw_text}" → corrected="{result.corrected_text}" '
              f'(valid={result.is_valid_format})')


if __name__ == '__main__':
    main()
