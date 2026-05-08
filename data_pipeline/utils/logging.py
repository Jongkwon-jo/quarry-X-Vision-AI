"""logging.py — 간단한 로거 헬퍼.

노트북의 print 호출을 대체하여 단계별로 일관된 출력 형식을 제공.
"""
import logging
import sys


def get_logger(name: str, verbose: bool = True) -> logging.Logger:
    """이름별로 로거를 생성. 중복 핸들러는 추가하지 않음."""
    logger = logging.getLogger(name)
    
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            '[%(asctime)s] %(name)-20s %(levelname)-7s %(message)s',
            datefmt='%H:%M:%S',
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    logger.propagate = False
    return logger
