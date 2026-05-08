"""collector.py — 수집된 이미지 폴더 스캔.

원본 노트북 셀 0의 동작을 클래스화:
    - 지정 폴더 내 파일들을 스캔
    - CollectedImage 객체 리스트로 반환
"""
from pathlib import Path
from typing import List

from ..config import CollectionConfig
from ..schemas import CollectedImage, CollectionResult
from ..utils import get_logger


class ImageCollector:
    """현장 PC에서 복사받은 이미지 폴더를 스캔."""
    
    def __init__(self, config: CollectionConfig, verbose: bool = True):
        self.config = config
        self.logger = get_logger('collector', verbose=verbose)
    
    def collect(self) -> CollectionResult:
        """source_path 내 모든 파일을 CollectedImage로 변환.
        
        주의: 이 단계에서는 필터링하지 않는다 (다음 단계 책임).
        파일이 아닌 경로(폴더)는 자동 제외.
        """
        source = self.config.source_path
        
        if not source.exists():
            raise FileNotFoundError(
                f'수집 폴더가 존재하지 않습니다: {source}\n'
                f'TeamViewer 동기화가 완료되었는지 확인하세요.'
            )
        
        self.logger.info(f'스캔 시작: {source}')
        
        images: List[CollectedImage] = []
        for path in source.iterdir():
            if not path.is_file():
                continue
            images.append(CollectedImage(
                path=path,
                filename=path.name,
                extension=path.suffix,
            ))
        
        self.logger.info(f'발견된 파일: {len(images)}개')
        
        return CollectionResult(
            source_path=source,
            images=images,
        )
