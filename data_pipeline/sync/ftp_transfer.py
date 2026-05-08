"""ftp_transfer.py — 학습 서버로 데이터셋 FTP 전송.

원본 노트북 셀 5를 클래스화. 핵심 기능:
    - cp949 인코딩으로 한글 파일명 호환
    - 중첩된 원격 폴더 자동 생성 (ftp_makedirs)
    - tqdm 진행률 표시
"""
from ftplib import FTP, error_perm
from pathlib import Path
from typing import Iterable

from ..config import FTPConfig
from ..schemas import FTPTransferResult
from ..utils import get_logger


class FTPTransfer:
    """FTP를 통해 학습 서버로 디렉토리/파일 전송."""

    SUB_SETS = ('train', 'valid', 'test')
    
    def __init__(self, config: FTPConfig, verbose: bool = True):
        self.config = config
        self.logger = get_logger('ftp_transfer', verbose=verbose)
    
    def _connect(self) -> FTP:
        """FTP 연결 생성. 호출자가 close() 책임."""
        self.config.validate()
        ftp = FTP()
        ftp.connect(self.config.host, self.config.port)
        ftp.login(user=self.config.user, passwd=self.config.password)
        ftp.encoding = self.config.encoding
        self.logger.debug(
            f'FTP 연결: {self.config.user}@{self.config.host}:{self.config.port}'
        )
        return ftp
    
    @staticmethod
    def _ftp_makedirs(ftp: FTP, remote_path: str) -> None:
        """중첩 원격 폴더 자동 생성.
        
        원본 노트북의 ftp_makedirs 함수와 동일한 동작:
        경로를 슬래시로 분해해 단계별로 cwd 시도, 실패 시 mkd.
        """
        path_parts = [p for p in remote_path.split('/') if p]
        current = ''
        for part in path_parts:
            current += f'/{part}'
            try:
                ftp.cwd(current)
            except error_perm:
                ftp.mkd(current)
                ftp.cwd(current)

    def _ftp_rmtree(self, ftp: FTP, path: str) -> None:
        """업로드 전에 기존 폴더 삭제"""
        try:
            ftp.cwd(path)
        except Exception:
            return  # 폴더 없음

        file_list = ftp.nlst()

        for name in file_list:
            full_path = f"{path}/{name}"
            try:
                ftp.cwd(full_path)
                ftp.cwd("..")
                self._ftp_rmtree(ftp, full_path)
            except Exception:
                ftp.delete(full_path)

        ftp.rmd(path)

    def _clear_remote_subsets(self, ftp: FTP, remote_base: str) -> None:
        """remote_base 하위의 subset 폴더(train/valid/test)만 삭제."""
        for subset in self.SUB_SETS:
            remote_subset_path = f'{remote_base}/{subset}'
            self._ftp_rmtree(ftp, remote_subset_path)
    
    def upload_directory(
        self,
        local_root: Path,
        remote_subdir: str = '',
        clear_remote_before_upload: bool = True,
    ) -> FTPTransferResult:
        """로컬 디렉토리 전체를 학습 서버로 업로드.
        
        Args:
            local_root: 업로드할 로컬 디렉토리
            remote_subdir: remote_base_dir 하위에 추가할 서브 경로
                          (보통 데이터셋 버전 디렉토리명)
            clear_remote_before_upload: 업로드 전에 기존 원격 폴더 삭제 여부
        
        Returns:
            FTPTransferResult
        """
        if not local_root.exists():
            raise FileNotFoundError(f'로컬 디렉토리 없음: {local_root}')
        
        remote_base = self.config.remote_base_dir.rstrip('/')
        if remote_subdir:
            remote_base = f'{remote_base}/{remote_subdir.strip("/")}'
        
        result = FTPTransferResult(
            host=self.config.host,
            remote_base=remote_base,
        )
        
        # 전송 대상 파일 목록 수집
        files = [p for p in local_root.rglob('*') if p.is_file()]
        if not files:
            self.logger.warning(f'전송할 파일이 없습니다: {local_root}')
            return result
        
        self.logger.info(
            f'FTP 업로드 시작: {len(files)}개 파일 → {remote_base}'
        )
        
        # tqdm은 선택 의존성으로 처리
        try:
            from tqdm import tqdm
            iterator = tqdm(files, desc='Upload', unit='file')
        except ImportError:
            iterator = files
        
        ftp = self._connect()
        try:
            if clear_remote_before_upload:
                try:
                    self._clear_remote_subsets(ftp, remote_base)
                except Exception as e:
                    self.logger.warning(f'기존 원격 폴더 삭제 실패: {remote_base} ({e})')

            for local_path in iterator:
                relative = local_path.relative_to(local_root).as_posix()
                remote_path = f'{remote_base}/{relative}'
                
                # 디렉토리 부분 자동 생성
                remote_dir = '/'.join(remote_path.split('/')[:-1])
                
                try:
                    self._ftp_makedirs(ftp, remote_dir)
                    with open(local_path, 'rb') as f:
                        ftp.storbinary(
                            f'STOR {remote_path}',
                            f,
                            blocksize=self.config.block_size,
                        )
                    result.transferred.append(remote_path)
                    result.total_bytes += local_path.stat().st_size
                except Exception as e:
                    result.failed.append((str(local_path), str(e)))
                    self.logger.warning(f'  실패: {local_path.name} ({e})')
        finally:
            try:
                ftp.quit()
            except Exception:
                ftp.close()
        
        mb = result.total_bytes / (1024 * 1024)
        self.logger.info(
            f'FTP 완료 — 성공 {result.success_count}, '
            f'실패 {len(result.failed)} ({mb:.1f} MB 전송)'
        )
        return result
