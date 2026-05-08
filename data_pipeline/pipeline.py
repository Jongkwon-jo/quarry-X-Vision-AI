"""pipeline.py — 8단계 데이터 수집·학습 파이프라인 메인 진입점.

ANPRPipeline과 동일한 패턴으로 구성. 각 단계를 독립적으로 실행 가능하며
run_full()로 전체 흐름을 한 번에 트리거할 수 있다.

사용 예 (전체 실행):
    pipeline = DataPipeline.from_env()
    pipeline.run_full()

사용 예 (단계별 실행):
    pipeline = DataPipeline.from_env()
    
    # 1차 사이클: 수집 → 업로드만
    collection = pipeline.run_collection()
    filtered   = pipeline.run_filter(collection)
    pipeline.run_upload(filtered)
    
    # ... Roboflow에서 사람이 라벨링 ...
    
    # 2차 사이클: 다운로드 → 학습 트리거
    download = pipeline.run_download()
    pipeline.run_replace(download, collection.source_path)
    pipeline.run_sync(download)
    pipeline.run_training()
"""
from dataclasses import dataclass, field
from typing import List, Optional

from .config import PipelineConfig
from .schemas import (
    CollectionResult, FilterResult, UploadResult,
    DownloadResult, ReplaceResult,
    FTPTransferResult, TrainingResult,
)
from .collection import ImageCollector
from .filter import FilterAndCopy
from .upload import RoboflowUploader
from .download import RoboflowDownloader, ImageReplacer
from .sync import FTPTransfer
from .training import SSHTrainingTrigger
from .utils import get_logger


@dataclass
class PipelineRunReport:
    """전체 실행의 단계별 결과 묶음."""
    collection: Optional[CollectionResult] = None
    filter: Optional[FilterResult] = None
    upload: Optional[UploadResult] = None
    download: Optional[DownloadResult] = None
    replace: List[ReplaceResult] = field(default_factory=list)
    sync: Optional[FTPTransferResult] = None
    training: Optional[TrainingResult] = None


class DataPipeline:
    """8단계 데이터 수집·학습 파이프라인."""
    
    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig.default()
        self.logger = get_logger('pipeline', verbose=self.config.verbose)
        
        # 각 단계 모듈 초기화
        self.collector  = ImageCollector(self.config.collection, self.config.verbose)
        self.filterer   = FilterAndCopy(self.config.filter, self.config.verbose)
        self.uploader   = RoboflowUploader(self.config.roboflow, self.config.verbose)
        self.downloader = RoboflowDownloader(self.config.roboflow, self.config.verbose)
        self.replacer   = ImageReplacer(self.config.replace, self.config.verbose)
        self.ftp_sync   = FTPTransfer(self.config.ftp, self.config.verbose)
        self.trainer    = SSHTrainingTrigger(self.config.ssh, self.config.verbose)
    
    # ===== 팩토리 =====
    
    @classmethod
    def default(cls) -> 'DataPipeline':
        return cls(PipelineConfig.default())
    
    @classmethod
    def from_env(cls) -> 'DataPipeline':
        return cls(PipelineConfig.from_env())
    
    # ===== 단계별 실행 =====
    
    def run_collection(self) -> CollectionResult:
        """① 이미지 수집 폴더 스캔."""
        self.logger.info('=' * 60)
        self.logger.info('① 이미지 수집')
        return self.collector.collect()
    
    def run_filter(self, collection: CollectionResult) -> FilterResult:
        """② 키워드 필터링 + ③ Dataset 폴더 복사."""
        self.logger.info('=' * 60)
        self.logger.info('② 필터링 + ③ Dataset 복사')
        # 원본 source_path의 부모 디렉토리를 dest_root로 사용
        dest_root = collection.source_path.parent / collection.source_path.name
        # 노트북 동작과 동일: VISION_AI/dataset/ 형태가 되도록
        # config.collection.source_root 하위에 dataset 폴더를 만든다
        dest_root = collection.source_path.parent
        return self.filterer.run(collection, dest_root)
    
    def run_upload(
        self,
        filter_result: FilterResult,
        granular: bool = False,
    ) -> UploadResult:
        """④ Roboflow 업로드.
        
        Args:
            granular: True면 파일별 개별 업로드 (느림, 통계 정확)
                     False면 일괄 업로드 (빠름, 통계 부정확)
        """
        self.logger.info('=' * 60)
        self.logger.info('④ Roboflow 업로드')
        if granular:
            return self.uploader.granular_upload(filter_result.dest_path)
        return self.uploader.simple_upload(filter_result.dest_path)
    
    def run_download(self) -> DownloadResult:
        """⑥ YOLO 데이터셋 다운로드."""
        self.logger.info('=' * 60)
        self.logger.info('⑥ YOLO 데이터셋 다운로드')
        return self.downloader.download()
    
    def run_replace(
        self,
        download: DownloadResult,
        original_source_dir,
    ) -> List[ReplaceResult]:
        """⑦ Roboflow 변환 이미지 → 원본 무손실 이미지 교체."""
        self.logger.info('=' * 60)
        self.logger.info('⑦ 원본 이미지 교체 + 라벨 정리')
        return self.replacer.replace_all(
            dataset_root=download.location,
            original_source_dir=original_source_dir,
        )
    
    def run_sync(
        self,
        download: DownloadResult,
        remote_subdir: Optional[str] = None,
    ) -> FTPTransferResult:
        """⑦ FTP 학습 서버 전송."""
        self.logger.info('=' * 60)
        self.logger.info('⑦ FTP 학습 서버 전송')
        # if remote_subdir is None:
        #     # 데이터셋 폴더명을 그대로 사용
        #     remote_subdir = download.location.name
        return self.ftp_sync.upload_directory(
            local_root=download.location,
            remote_subdir=remote_subdir,
        )
    
    def run_training(self, stream_output: bool = True) -> TrainingResult:
        """⑧ SSH 원격 학습 트리거."""
        self.logger.info('=' * 60)
        self.logger.info('⑧ SSH 원격 학습 트리거')
        return self.trainer.trigger(stream_output=stream_output)
    
    # ===== 전체 실행 =====
    
    def run_full(
        self,
        skip_labeling_wait: bool = False,
        granular_upload: bool = False,
    ) -> PipelineRunReport:
        """전체 파이프라인 실행.
        
        주의: ⑤ 라벨링 단계는 사람의 수동 작업이라
        skip_labeling_wait=False (기본)인 경우 업로드 후 일시 정지하고
        사용자에게 라벨링 완료 후 Enter를 누르도록 요청한다.
        
        Args:
            skip_labeling_wait: True면 라벨링 대기 없이 다운로드까지 바로 진행
                                (이미 라벨링된 데이터셋이 Roboflow에 있을 때).
            granular_upload: 업로드를 파일별 개별 모드로 수행할지.
        """
        # 보안 검증: 모든 비밀값이 설정되어 있는지 확인
        self.config.validate_secrets()
        
        report = PipelineRunReport()
        
        # 1차 사이클
        report.collection = self.run_collection()
        report.filter = self.run_filter(report.collection)
        report.upload = self.run_upload(report.filter, granular=granular_upload)
        
        # ⑤ 라벨링 대기
        if not skip_labeling_wait:
            self.logger.info('=' * 60)
            self.logger.info('⑤ 라벨링 단계 (수동) — Roboflow 웹 UI에서 라벨링 완료 후 Enter 입력')
            try:
                input('  >> 라벨링 완료 후 Enter를 누르세요: ')
            except (EOFError, KeyboardInterrupt):
                self.logger.warning('사용자 중단 — 다운로드 단계 건너뜀')
                return report
        
        # 2차 사이클
        report.download = self.run_download()
        report.replace = self.run_replace(
            report.download, report.collection.source_path
        )
        report.sync = self.run_sync(report.download)
        report.training = self.run_training()
        
        self.logger.info('=' * 60)
        self.logger.info('파이프라인 완료')
        return report
