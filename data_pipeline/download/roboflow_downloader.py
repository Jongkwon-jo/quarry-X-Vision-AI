"""roboflow_downloader.py — 라벨링 완료된 데이터셋 다운로드.

원본 노트북 셀 3을 클래스화. project.version(N).download(format)을 호출.
"""
from pathlib import Path

from ..config import RoboflowConfig
from ..schemas import DownloadResult
from ..utils import get_logger


class RoboflowDownloader:
    """Roboflow에서 학습용 데이터셋 다운로드."""
    
    def __init__(self, config: RoboflowConfig, verbose: bool = True):
        self.config = config
        self.logger = get_logger('roboflow_download', verbose=verbose)
        self._rf = None
    
    @property
    def rf(self):
        if self._rf is None:
            self.config.validate()
            import roboflow
            self._rf = roboflow.Roboflow(api_key=self.config.api_key)
        return self._rf
    
    def download(self) -> DownloadResult:
        """설정된 버전·형식으로 데이터셋 다운로드.
        
        Returns:
            DownloadResult — location은 Roboflow가 만든 폴더 (보통
            현재 작업 디렉토리 하위에 '{project}-{version}' 형식)
        """
        version = self.config.download_version
        fmt = self.config.download_format
        location = self.config.download_location
        
        self.logger.info(
            f'다운로드 시작: {self.config.project_name} v{version} ({fmt})'
        )
        
        project = self.rf.workspace(self.config.workspace_name) \
                         .project(self.config.project_name)
        dataset = project.version(version).download(model_format=fmt, location=location, overwrite=True)
        
        location = Path(dataset.location)
        self.logger.info(f'다운로드 완료: {location}')
        
        return DownloadResult(
            location=location,
            version=version,
            format=fmt,
        )
