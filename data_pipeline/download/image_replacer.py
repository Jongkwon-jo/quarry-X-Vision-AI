"""image_replacer.py — Roboflow 다운로드 이미지를 원본 무손실 이미지로 교체.

원본 노트북 셀 4를 클래스화. 학습 품질 보존을 위한 핵심 단계.

배경:
    Roboflow는 다운로드 시 이미지를 자동으로 JPG/PNG로 변환·압축하여
    파일명도 'xxx.rf.<hash>.jpg' 형식으로 변경한다. BMP 무손실 데이터를
    학습에 그대로 쓰려면 이 변환된 이미지를 다시 원본으로 교체해야 한다.

처리 흐름 (subset마다):
    1. images/ 폴더 내 각 파일에 대해 Roboflow 파일명을 원본 파일명으로 복원
    2. 원본 폴더에서 해당 파일을 찾아 복사
    3. labels/ 폴더의 .txt 파일명도 동일하게 변경 (이미지와 1:1 매칭 유지)
"""
import os
import re
import shutil
from pathlib import Path
from typing import List, Tuple

from ..config import ReplaceConfig
from ..schemas import ReplaceResult
from ..utils import get_logger


class ImageReplacer:
    """Roboflow 변환 이미지 → 원본 무손실 이미지 교체."""
    
    def __init__(self, config: ReplaceConfig, verbose: bool = True):
        self.config = config
        self.logger = get_logger('image_replacer', verbose=verbose)
        self._hash_re = re.compile(self.config.rf_hash_pattern, re.IGNORECASE)
    
    def restore_original_info(self, rf_filename: str) -> Tuple[str, str]:
        """Roboflow 파일명에서 원본 파일명·확장자 복원.
        
        Roboflow가 다운로드 시 파일명을 변경하는 패턴:
            front_xxx_대구06라5245.BMP
              → front_xxx_대구06라5245_BMP.rf.<hash>.jpg
        
        복원 단계:
            1. .rf.<hash> 제거
            2. Roboflow가 추가한 .jpg/.png 확장자 제거
            3. _BMP / _jpg 마커를 실제 확장자로 복원
        
        Returns:
            (원본_전체_파일명, 확장자_제외_순수_파일명)
        """
        # 1. .rf.<hash>... 부분 제거
        clean = self._hash_re.sub('', rf_filename)
        
        # 2. Roboflow가 자동으로 추가한 확장자 (.jpg/.jpeg/.png) 제거
        # 마커(_BMP, _jpg) 처리 전에 떼어내야 함
        for rf_ext in ('.jpg', '.jpeg', '.png', '.JPG', '.PNG'):
            if clean.endswith(rf_ext):
                clean = clean[:-len(rf_ext)]
                break
        
        # 3. _BMP / _jpg 등 마커 → 실제 확장자로 복원
        actual_name = clean
        for marker, real_ext in self.config.extension_map.items():
            if clean.endswith(marker):
                actual_name = clean[:-len(marker)] + real_ext
                break
        
        # 4. 확장자 제외한 순수 이름 (라벨 파일 매칭용)
        pure_name = os.path.splitext(actual_name)[0]
        return actual_name, pure_name
    
    def replace_subset(
        self,
        dataset_root: Path,
        subset: str,
        original_source_dir: Path,
    ) -> ReplaceResult:
        """단일 subset(train/valid/test)에 대해 교체 수행."""
        result = ReplaceResult(subset=subset)
        
        subset_path = dataset_root / subset
        if not subset_path.exists():
            self.logger.debug(f'subset 없음: {subset_path}')
            return result
        
        img_dir = subset_path / 'images'
        lbl_dir = subset_path / 'labels'
        if not img_dir.exists():
            return result
        
        self.logger.info(f'[{subset.upper()}] 원본 교체 + 라벨 정리 시작')
        
        for rf_filename in os.listdir(img_dir):
            old_img_path = img_dir / rf_filename
            if not old_img_path.is_file():
                continue
            
            # 1. 원본 파일명 복원
            original_name, pure_name = self.restore_original_info(rf_filename)
            source_path = original_source_dir / original_name
            
            # 2. 원본이 존재하는지 확인
            if not source_path.exists():
                result.missing_source.append(original_name)
                continue
            
            # 3. Roboflow 변환 이미지 삭제 + 원본 복사
            try:
                old_img_path.unlink()
                shutil.copy2(source_path, img_dir / original_name)
                result.replaced += 1
            except Exception as e:
                self.logger.warning(f'  교체 실패: {rf_filename} ({e})')
                continue
            
            # 4. 라벨 파일명 동기화
            if lbl_dir.exists():
                # 기존 라벨: <rf_filename(확장자 제외)>.txt
                rf_stem = os.path.splitext(rf_filename)[0]
                old_lbl_path = lbl_dir / f'{rf_stem}.txt'
                new_lbl_path = lbl_dir / f'{pure_name}.txt'
                
                if old_lbl_path.exists() and old_lbl_path != new_lbl_path:
                    if new_lbl_path.exists():
                        new_lbl_path.unlink()
                    old_lbl_path.rename(new_lbl_path)
                    result.label_renamed += 1
        
        self.logger.info(
            f'  ✓ {subset}: 교체 {result.replaced}개, '
            f'라벨 변경 {result.label_renamed}개, '
            f'원본 누락 {len(result.missing_source)}개'
        )
        return result
    
    def replace_all(
        self,
        dataset_root: Path,
        original_source_dir: Path,
    ) -> List[ReplaceResult]:
        """train/valid/test 모든 subset에 대해 교체."""
        if not dataset_root.exists():
            raise FileNotFoundError(f'데이터셋 루트 없음: {dataset_root}')
        if not original_source_dir.exists():
            raise FileNotFoundError(f'원본 소스 없음: {original_source_dir}')
        
        self.logger.info(f'원본 소스: {original_source_dir}')
        self.logger.info(f'데이터셋:   {dataset_root}')
        
        results = []
        for subset in self.config.sub_sets:
            results.append(self.replace_subset(
                dataset_root, subset, original_source_dir
            ))
        
        total_replaced = sum(r.replaced for r in results)
        total_missing = sum(len(r.missing_source) for r in results)
        self.logger.info(
            f'전체 완료 — 교체 {total_replaced}개, 원본 누락 {total_missing}개'
        )
        return results
