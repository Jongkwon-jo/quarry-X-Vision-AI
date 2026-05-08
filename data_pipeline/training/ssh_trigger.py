"""ssh_trigger.py — 원격 학습 서버에 SSH 접속하여 학습 시작.

원본 노트북 셀 6을 클래스화. paramiko로 Windows 학습 서버에 접속해
PowerShell 명령으로 가상환경 Python을 호출하여 train.py를 실행한다.

핵심 처리:
    - cp949 디코딩으로 한글 출력 깨짐 방지
    - exit_status 반환으로 성공/실패 판정
    - 스트리밍 출력 옵션 (긴 학습 로그를 실시간으로 받기)
"""
from typing import Callable, Optional

from ..config import SSHConfig
from ..schemas import TrainingResult
from ..utils import get_logger


class SSHTrainingTrigger:
    """SSH로 원격 학습 서버에 접속하여 학습 명령을 실행."""
    
    def __init__(self, config: SSHConfig, verbose: bool = True):
        self.config = config
        self.logger = get_logger('ssh_training', verbose=verbose)
    
    def _build_powershell_command(self) -> str:
        """학습을 시작하는 PowerShell one-liner 구성.
        
        형태:
            powershell.exe -Command "Set-Location '<work_dir>'; & '<venv_python>' '<train_script>'"
        """
        return (
            f'powershell.exe -Command "'
            f'Set-Location \'{self.config.work_dir}\'; '
            f'& \'{self.config.venv_python}\' \'{self.config.train_script}\''
            f'"'
        )
    
    def trigger(
        self,
        stream_output: bool = True,
        on_line: Optional[Callable[[str], None]] = None,
    ) -> TrainingResult:
        """SSH 접속 후 학습 명령 실행.
        
        Args:
            stream_output: True면 출력을 라인 단위로 스트리밍 (긴 학습용).
                           False면 모든 출력이 끝난 뒤 한 번에 받음.
            on_line: 스트리밍 모드에서 각 줄마다 호출될 콜백.
                     None이면 logger.info로 출력.
        
        Returns:
            TrainingResult — exit_status로 성공 여부 판정
        """
        self.config.validate()
        
        # paramiko는 무거운 의존성이라 함수 진입 시점에 import
        import paramiko
        
        command = self._build_powershell_command()
        self.logger.info(f'SSH 접속: {self.config.user}@{self.config.host}')
        self.logger.debug(f'실행 명령: {command}')
        
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        
        result = TrainingResult(
            host=self.config.host,
            command=command,
        )
        
        try:
            ssh.connect(
                hostname=self.config.host,
                port=self.config.port,
                username=self.config.user,
                password=self.config.password,
                timeout=30,
            )
            
            stdin, stdout, stderr = ssh.exec_command(
                command,
                timeout=self.config.command_timeout,
            )
            
            if stream_output:
                # 라인 단위 스트리밍
                stdout_lines = []
                for raw_line in iter(stdout.readline, ''):
                    if not raw_line:
                        break
                    line = self._decode_line(raw_line)
                    stdout_lines.append(line)
                    
                    if on_line:
                        on_line(line)
                    else:
                        self.logger.info(f'  | {line.rstrip()}')
                
                result.stdout = ''.join(stdout_lines)
                result.stderr = self._decode_bytes(stderr.read())
            else:
                # 일괄 읽기
                result.stdout = self._decode_bytes(stdout.read())
                result.stderr = self._decode_bytes(stderr.read())
            
            result.exit_status = stdout.channel.recv_exit_status()
            
            if result.success:
                self.logger.info('학습 명령 정상 종료 (exit_status=0)')
            else:
                self.logger.warning(
                    f'학습 명령 비정상 종료 (exit_status={result.exit_status})'
                )
                if result.stderr:
                    self.logger.warning(f'stderr:\n{result.stderr}')
        finally:
            ssh.close()
        
        return result
    
    def _decode_bytes(self, data: bytes) -> str:
        """SSH 출력 바이트를 cp949로 디코딩."""
        try:
            return data.decode(self.config.stdout_encoding, errors='ignore')
        except Exception:
            return data.decode('utf-8', errors='ignore')
    
    def _decode_line(self, raw) -> str:
        """paramiko readline 결과 (bytes 또는 str) 처리."""
        if isinstance(raw, bytes):
            return self._decode_bytes(raw)
        return raw
