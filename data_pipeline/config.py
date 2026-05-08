"""
config.py — 데이터 파이프라인 전역 설정

모든 경로·인증정보·임계값을 한곳에서 관리한다.
환경변수에서 비밀값을 읽어와 노트북에 평문 저장하는 보안 이슈를 해결.

사용 예:
    from data_pipeline.config import PipelineConfig
    
    config = PipelineConfig.from_env()  # .env 파일 로드
    config = PipelineConfig.default()   # 기본값 (개발용)
"""
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


# ===== 단계별 설정 =====

@dataclass
class CollectionConfig:
    """① 이미지 수집 단계 설정."""
    source_root: str = r'C:\Users\User\Desktop\Project\VISION_AI'
    source_subdir: str = '20260428'  # 수집 회차별 폴더
    
    @property
    def source_path(self) -> Path:
        return Path(self.source_root) / self.source_subdir


@dataclass
class FilterConfig:
    """② 파일 필터링 + ③ Dataset 폴더 복사 설정."""
    # 정면 화각만 추출 (필요시 'side', 'rear' 추가 가능)
    keyword: str = ''#'front'
    # Roboflow 업로드 작업 디렉토리
    dest_subdir: str = 'dataset'
    # 지원 확장자
    allowed_extensions: tuple = ('.bmp', '.BMP')#'.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG'


@dataclass
class RoboflowConfig:
    """④ Roboflow 업로드 + ⑥ 다운로드 설정."""
    api_key: str = ''  # 환경변수에서 로드
    workspace_name: str = 'jongkwons-workspace'
    project_name: str = '20260428'
    project_type: str = 'object-detection'
    upload_workers: int = 10
    
    # 다운로드 시 사용할 데이터셋 버전과 형식
    download_version: int = 3
    download_format: str = 'yolo26'
    
    def validate(self) -> None:
        if not self.api_key:
            raise ValueError(
                'Roboflow API 키가 설정되지 않았습니다. '
                'ROBOFLOW_API_KEY 환경변수를 설정하거나 config.api_key를 직접 지정하세요.'
            )


@dataclass
class ReplaceConfig:
    """⑦ Roboflow 압축 이미지 → 원본 교체 설정.
    
    Roboflow가 다운로드 시 자동으로 이미지를 변환·압축하므로,
    학습 품질 보존을 위해 원본 무손실 이미지로 다시 교체한다.
    """
    sub_sets: tuple = ('train', 'valid', 'test')
    # Roboflow 파일명 정규화 패턴 (.rf.<hash>... 부분 제거)
    rf_hash_pattern: str = r'\.rf\.[a-z0-9]+'
    # 확장자 변환 매핑 (Roboflow 형식 → 원본)
    extension_map: dict = field(default_factory=lambda: {
        '_BMP': '.BMP',
        '_jpg': '.jpg',
        '_JPG': '.JPG',
        '_jpeg': '.jpeg',
        '_png': '.png',
    })


@dataclass
class FTPConfig:
    """⑦ FTP 전송 설정."""
    host: str = '192.168.1.179'
    port: int = 21
    user: str = 'administrator'
    password: str = ''  # 환경변수에서 로드
    encoding: str = 'cp949'  # 한글 파일명 호환
    # 학습 서버 내 데이터셋 저장 위치
    remote_base_dir: str = '/ai_vision/yolo'
    # 청크 크기
    block_size: int = 8192
    
    def validate(self) -> None:
        if not self.password:
            raise ValueError(
                'FTP 비밀번호가 설정되지 않았습니다. '
                'FTP_PASSWORD 환경변수를 설정하세요.'
            )


@dataclass
class SSHConfig:
    """⑧ SSH 학습 트리거 설정."""
    host: str = '192.168.1.179'
    port: int = 22
    user: str = 'administrator'
    password: str = ''  # 환경변수에서 로드
    
    # 학습 서버의 가상환경 + 학습 스크립트 경로
    venv_python: str = r'D:\ai_vision\yolo\venv\Scripts\python.exe'
    work_dir: str = r'D:\ai_vision\yolo'
    train_script: str = 'train.py'
    
    # 출력 디코딩
    stdout_encoding: str = 'cp949'
    
    # 명령 실행 타임아웃 (초). None이면 무제한 (학습이 길어서 None 권장)
    command_timeout: Optional[int] = None
    
    def validate(self) -> None:
        if not self.password:
            raise ValueError(
                'SSH 비밀번호가 설정되지 않았습니다. '
                'SSH_PASSWORD 환경변수를 설정하세요.'
            )


# ===== 전체 파이프라인 설정 묶음 =====

@dataclass
class PipelineConfig:
    """전체 파이프라인 설정."""
    collection: CollectionConfig = field(default_factory=CollectionConfig)
    filter: FilterConfig = field(default_factory=FilterConfig)
    roboflow: RoboflowConfig = field(default_factory=RoboflowConfig)
    replace: ReplaceConfig = field(default_factory=ReplaceConfig)
    ftp: FTPConfig = field(default_factory=FTPConfig)
    ssh: SSHConfig = field(default_factory=SSHConfig)
    
    # 디버그 로그 출력 여부
    verbose: bool = True
    
    @classmethod
    def default(cls) -> 'PipelineConfig':
        """기본값 (개발·테스트용). 비밀값은 비어있음."""
        return cls()
    
    @classmethod
    def from_env(cls) -> 'PipelineConfig':
        """환경변수에서 비밀값을 로드한 설정.
        
        지원 환경변수:
            ROBOFLOW_API_KEY
            FTP_PASSWORD
            SSH_PASSWORD
            COLLECTION_SOURCE_ROOT (선택)
            COLLECTION_SOURCE_SUBDIR (선택)
            ROBOFLOW_PROJECT (선택)
        """
        config = cls.default()
        
        # 비밀값
        config.roboflow.api_key = os.environ.get('ROBOFLOW_API_KEY', '')
        config.ftp.password = os.environ.get('FTP_PASSWORD', '')
        config.ssh.password = os.environ.get('SSH_PASSWORD', '')
        
        # 선택적 오버라이드
        if 'COLLECTION_SOURCE_ROOT' in os.environ:
            config.collection.source_root = os.environ['COLLECTION_SOURCE_ROOT']
        if 'COLLECTION_SOURCE_SUBDIR' in os.environ:
            config.collection.source_subdir = os.environ['COLLECTION_SOURCE_SUBDIR']
        if 'ROBOFLOW_PROJECT' in os.environ:
            config.roboflow.project_name = os.environ['ROBOFLOW_PROJECT']
        
        return config
    
    def validate_secrets(self) -> None:
        """비밀값이 모두 설정되어 있는지 확인."""
        self.roboflow.validate()
        self.ftp.validate()
        self.ssh.validate()
