"""
schemas.py — 단계 간 데이터 전달 객체.

각 단계가 dataclass를 입출력으로 사용해 결합도를 낮추고
디버깅·테스트 용이성을 확보한다.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class CollectedImage:
    """① collection 단계 출력."""
    path: Path
    filename: str
    extension: str
    
    @property
    def has_front_keyword(self) -> bool:
        return 'front' in self.filename.lower()


@dataclass
class CollectionResult:
    """① collection 단계 결과 묶음."""
    source_path: Path
    images: List[CollectedImage] = field(default_factory=list)
    
    @property
    def count(self) -> int:
        return len(self.images)


@dataclass
class FilterResult:
    """② filter + ③ copy 단계 결과."""
    dest_path: Path
    copied: List[Path] = field(default_factory=list)
    skipped: List[Path] = field(default_factory=list)
    failed: List[tuple] = field(default_factory=list)  # (path, error_msg)
    
    @property
    def success_count(self) -> int:
        return len(self.copied)


@dataclass
class UploadResult:
    """④ Roboflow 업로드 결과."""
    workspace: str
    project: str
    uploaded: List[str] = field(default_factory=list)   # 신규 업로드된 파일명
    duplicated: List[str] = field(default_factory=list) # 중복 처리된 파일명
    failed: List[tuple] = field(default_factory=list)
    
    @property
    def total(self) -> int:
        return len(self.uploaded) + len(self.duplicated) + len(self.failed)
    
    @property
    def duplicate_rate(self) -> float:
        return len(self.duplicated) / self.total if self.total else 0.0


@dataclass
class DownloadResult:
    """⑥ 데이터셋 다운로드 결과."""
    location: Path
    version: int
    format: str


@dataclass
class ReplaceResult:
    """⑦ 원본 이미지 교체 결과 (subset별)."""
    subset: str  # 'train' / 'valid' / 'test'
    replaced: int = 0
    label_renamed: int = 0
    missing_source: List[str] = field(default_factory=list)


@dataclass
class FTPTransferResult:
    """⑦ FTP 전송 결과."""
    host: str
    remote_base: str
    transferred: List[str] = field(default_factory=list)
    failed: List[tuple] = field(default_factory=list)
    total_bytes: int = 0
    
    @property
    def success_count(self) -> int:
        return len(self.transferred)


@dataclass
class TrainingResult:
    """⑧ 원격 학습 트리거 결과."""
    host: str
    command: str
    exit_status: Optional[int] = None
    stdout: str = ''
    stderr: str = ''
    
    @property
    def success(self) -> bool:
        return self.exit_status == 0
