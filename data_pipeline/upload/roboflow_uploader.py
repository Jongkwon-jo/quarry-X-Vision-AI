"""roboflow_uploader.py — Roboflow 워크스페이스로 일괄 업로드.

원본 노트북 셀 2를 클래스화. workspace.upload_dataset()을 사용하며
UPLOADED/DUPLICATE 분류는 Roboflow가 자동으로 stdout에 기록한다.

기존 노트북 코드의 한계:
    - upload_dataset()은 응답 객체를 반환하지 않아 결과를 프로그래밍적으로
      수집할 수 없음.
    - 정확한 통계가 필요하면 개별 파일 업로드 모드를 사용해야 함.

이 클래스는 두 모드를 모두 지원:
    - simple_upload(): 노트북과 동일하게 일괄 업로드 (빠름, 통계 부정확)
    - granular_upload(): 파일별 개별 업로드 (느림, 정확한 통계)
"""
from pathlib import Path
from typing import Optional

from ..config import RoboflowConfig
from ..schemas import UploadResult
from ..utils import get_logger


class RoboflowUploader:
    """Roboflow 클라우드 업로드 래퍼."""
    
    def __init__(self, config: RoboflowConfig, verbose: bool = True):
        self.config = config
        self.logger = get_logger('roboflow_upload', verbose=verbose)
        self._rf = None
        self._workspace = None
    
    @property
    def rf(self):
        """첫 호출 시에만 Roboflow 클라이언트 초기화."""
        if self._rf is None:
            self.config.validate()
            import roboflow
            self._rf = roboflow.Roboflow(api_key=self.config.api_key)
            self._workspace = self._rf.workspace()
            self.logger.info(f'Roboflow 워크스페이스 연결: {self.config.workspace_name}')
        return self._rf
    
    @property
    def workspace(self):
        if self._workspace is None:
            _ = self.rf  # lazy init 트리거
        return self._workspace
    
    def simple_upload(self, dataset_dir: Path) -> UploadResult:
        """노트북과 동일한 일괄 업로드 (빠름).
        
        주의: upload_dataset이 결과 객체를 반환하지 않으므로
        UPLOADED/DUPLICATE 카운트가 0으로 표시될 수 있음.
        정확한 통계가 필요하면 granular_upload()를 사용.
        """
        if not dataset_dir.exists():
            raise FileNotFoundError(f'dataset 폴더 없음: {dataset_dir}')
        
        self.logger.info(f'Roboflow 일괄 업로드 시작: {dataset_dir}')
        
        # 노트북 셀 2와 동일한 호출
        self.workspace.upload_dataset(
            str(dataset_dir),
            self.config.project_name,
            num_workers=self.config.upload_workers,
            project_type=self.config.project_type,
        )
        
        result = UploadResult(
            workspace=self.config.workspace_name,
            project=self.config.project_name,
        )
        self.logger.info('업로드 완료 (상세 카운트는 Roboflow 대시보드에서 확인)')
        return result
    
    def granular_upload(self, dataset_dir: Path) -> UploadResult:
        """파일별 개별 업로드 (느리지만 정확한 통계 수집).
        
        각 파일에 대해 project.upload(image_path)를 호출하고
        반환값으로 UPLOADED/DUPLICATE를 분류한다.
        """
        if not dataset_dir.exists():
            raise FileNotFoundError(f'dataset 폴더 없음: {dataset_dir}')
        
        project = self.workspace.project(self.config.project_name)
        result = UploadResult(
            workspace=self.config.workspace_name,
            project=self.config.project_name,
        )
        
        # 이미지 파일만 추려냄
        images = [
            p for p in dataset_dir.iterdir()
            if p.is_file() and p.suffix.lower() in
            ('.jpg', '.jpeg', '.png', '.bmp')
        ]
        self.logger.info(f'파일별 업로드 시작 ({len(images)}개)')
        
        for i, img_path in enumerate(images, 1):
            try:
                response = project.upload(str(img_path))
                # Roboflow API 응답에서 duplicate 여부 추출
                # 응답 형식이 버전마다 다르므로 안전하게 처리
                is_duplicate = self._is_duplicate_response(response)
                
                if is_duplicate:
                    result.duplicated.append(img_path.name)
                else:
                    result.uploaded.append(img_path.name)
                
                if i % 10 == 0 or i == len(images):
                    self.logger.info(
                        f'  [{i}/{len(images)}] '
                        f'UPLOADED={len(result.uploaded)} '
                        f'DUPLICATE={len(result.duplicated)}'
                    )
            except Exception as e:
                result.failed.append((img_path.name, str(e)))
                self.logger.warning(f'  실패: {img_path.name} ({e})')
        
        self.logger.info(
            f'업로드 완료 — 신규 {len(result.uploaded)}, '
            f'중복 {len(result.duplicated)}, 실패 {len(result.failed)} '
            f'(중복률 {result.duplicate_rate*100:.1f}%)'
        )
        return result
    
    @staticmethod
    def _is_duplicate_response(response) -> bool:
        """Roboflow API 응답에서 중복 여부 판정.
        
        응답 형식이 dict/str 등 버전별로 다르므로 모두 처리.
        """
        if isinstance(response, dict):
            if response.get('duplicate'):
                return True
            if response.get('status') == 'duplicate':
                return True
            if 'duplicate' in str(response).lower():
                return True
        elif isinstance(response, str):
            return 'duplicate' in response.lower()
        return False
