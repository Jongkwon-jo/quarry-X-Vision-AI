"""filter_copy.py — 키워드 필터링 + 작업 폴더 복사.

원본 노트북 셀 1의 동작을 클래스화:
    1. CollectionResult에서 'front' 키워드 포함 파일 필터링
    2. dataset 폴더로 복사 (shutil.copy2로 메타데이터 보존)
"""
import shutil
from pathlib import Path
from typing import List

from ..config import FilterConfig
from ..schemas import CollectionResult, FilterResult
from ..utils import get_logger


class FilterAndCopy:
    """수집 결과를 필터링하여 Roboflow 업로드용 폴더에 복사."""
    
    def __init__(self, config: FilterConfig, verbose: bool = True):
        self.config = config
        self.logger = get_logger('filter_copy', verbose=verbose)
    
    def run(
        self,
        collection: CollectionResult,
        dest_root: Path,
    ) -> FilterResult:
        """필터링 + 복사 수행.
        
        Args:
            collection: ImageCollector.collect()의 결과
            dest_root: dataset 폴더가 만들어질 루트 경로
                       (보통 collection.source_path.parent)
        
        Returns:
            FilterResult (성공/스킵/실패 분류)
        """
        dest_path = dest_root / self.config.dest_subdir
        dest_path.mkdir(parents=True, exist_ok=True)
        
        result = FilterResult(dest_path=dest_path)
        keyword = self.config.keyword.lower()
        
        for img in collection.images:
            # 필터 1: 키워드 포함 여부
            if keyword not in img.filename.lower():
                result.skipped.append(img.path)
                continue
            
            # 필터 2: 허용된 확장자
            if img.extension not in self.config.allowed_extensions:
                result.skipped.append(img.path)
                continue
            
            target = dest_path / img.filename
            try:
                shutil.copy2(img.path, target)
                result.copied.append(target)
            except Exception as e:
                result.failed.append((img.path, str(e)))
                self.logger.warning(f'복사 실패: {img.filename} ({e})')
        
        self.logger.info(
            f'복사 {result.success_count}개 / 스킵 {len(result.skipped)}개 '
            f'/ 실패 {len(result.failed)}개  → {dest_path}'
        )
        return result
