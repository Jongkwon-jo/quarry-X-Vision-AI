"""pipeline.py — 전체 ANPR 파이프라인.

5단계 모듈을 순서대로 실행하는 메인 진입점.
이 클래스 하나가 사용자 코드의 인터페이스가 된다.
"""
from pathlib import Path
from typing import List, Optional, Union

import numpy as np

from .config import PipelineConfig
from .schemas import PlateResult
from .detection import PlateDetector
from .preprocessing import PlatePreprocessor
from .split import CharSplitter
from .ocr import create_ocr_engine
from .validation import PlateValidator, EnhancedPlateValidator
from .utils import load_image, save_image


class ANPRPipeline:
    """ANPR 전체 파이프라인.
    
    사용 예:
        pipeline = ANPRPipeline.from_default()
        result = pipeline.run('truck.jpg')
        print(result.summary())
    """
    
    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig.default()
        
        # 각 단계 모듈 초기화 (lazy load는 각 모듈에서 처리)
        self.detector = PlateDetector(self.config.detection)
        self.preprocessor = PlatePreprocessor(self.config.preprocessing)
        self.splitter = CharSplitter(self.config.split)
        self.ocr_engine = create_ocr_engine(self.config.ocr)
        
        # 강화 검증 사용 여부에 따라 다른 검증기 인스턴스화
        if self.config.validation.use_enhanced:
            self.validator = EnhancedPlateValidator(
                self.config.validation,
                require_quarry_compatible=self.config.validation.require_quarry_compatible,
            )
        else:
            self.validator = PlateValidator(self.config.validation)
    
    @classmethod
    def from_default(cls):
        """기본 설정으로 파이프라인 생성."""
        return cls(PipelineConfig.default())
    
    def run(
        self,
        image_or_path: Union[str, Path, np.ndarray],
    ) -> PlateResult:
        """단일 이미지 처리.
        
        Args:
            image_or_path: 이미지 경로 또는 BGR np.ndarray
        
        Returns:
            PlateResult (성공/실패 모두 포함, .summary()로 확인)
        """
        result = PlateResult()
        
        # 입력 로드
        if isinstance(image_or_path, np.ndarray):
            image = image_or_path
            result.source_image_path = None
        else:
            result.source_image_path = str(image_or_path)
            image = load_image(image_or_path)
            if image is None:
                result.error_stage = 'load'
                result.error_message = f'이미지 로드 실패: {image_or_path}'
                return result
        
        # ① 검출
        try:
            detection = self.detector.detect_best(image)
            if detection is None:
                result.error_stage = 'detection'
                result.error_message = '번호판 미검출'
                return result
            result.detection = detection
        except Exception as e:
            result.error_stage = 'detection'
            result.error_message = str(e)
            return result
        
        # ② 전처리
        try:
            preprocessed = self.preprocessor.process(detection.crop)
            result.preprocessed = preprocessed
        except Exception as e:
            result.error_stage = 'preprocessing'
            result.error_message = str(e)
            return result
        
        # ③ 글자 분리
        try:
            char_crops = self.splitter.split(preprocessed.rectified)
            if not char_crops:
                result.error_stage = 'split'
                result.error_message = '글자 분리 실패'
                return result
            result.char_crops = char_crops
        except Exception as e:
            result.error_stage = 'split'
            result.error_message = str(e)
            return result
        
        # ④ OCR
        try:
            recognitions = self.ocr_engine.recognize_all(char_crops)
            result.char_recognitions = recognitions
        except Exception as e:
            result.error_stage = 'ocr'
            result.error_message = str(e)
            return result
        
        # ⑤ 검증
        try:
            raw, corrected, is_valid, conf = self.validator.validate(recognitions)
            result.raw_text = raw
            result.corrected_text = corrected
            result.is_valid_format = is_valid
            result.overall_confidence = conf
            
            # 강화 검증기 사용 시 추가 리포트 생성
            if isinstance(self.validator, EnhancedPlateValidator):
                report = self.validator.validate_full(recognitions)
                result.validation_report = report
                # 강화 검증 결과로 is_valid_format 업데이트
                # (여기선 PASS만 success로 인정)
                from .validation import TrustLevel
                result.is_valid_format = (report.trust_level == TrustLevel.PASS)
        except Exception as e:
            result.error_stage = 'validation'
            result.error_message = str(e)
            return result
        
        # 디버그 출력
        if self.config.debug:
            self._save_debug_outputs(result)
        
        return result
    
    def run_batch(
        self,
        image_paths: List[Union[str, Path]],
        verbose: bool = True,
    ) -> List[PlateResult]:
        """여러 이미지 일괄 처리."""
        results = []
        for i, path in enumerate(image_paths):
            result = self.run(path)
            results.append(result)
            if verbose:
                print(f'[{i+1}/{len(image_paths)}] {Path(str(path)).name}: '
                      f'{result.summary()}')
        return results
    
    def _save_debug_outputs(self, result: PlateResult) -> None:
        """디버깅: 단계별 중간 결과 저장."""
        if not result.source_image_path:
            return
        debug_dir = Path(self.config.debug_dir) / Path(result.source_image_path).stem
        debug_dir.mkdir(parents=True, exist_ok=True)
        
        if result.detection is not None:
            save_image(result.detection.crop, debug_dir / '01_detection.jpg')
        if result.preprocessed is not None:
            save_image(result.preprocessed.rectified, debug_dir / '02_rectified.jpg')
            save_image(result.preprocessed.enhanced, debug_dir / '02_enhanced.jpg')
        for crop in result.char_crops:
            save_image(crop.image,
                       debug_dir / f'03_char_{crop.index:02d}_{crop.char_type}.jpg')
