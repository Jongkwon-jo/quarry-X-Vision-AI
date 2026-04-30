"""geometry.py — 기하 처리 유틸."""
import numpy as np


def order_corners(pts: np.ndarray) -> np.ndarray:
    """4개 점을 좌상-우상-우하-좌하 순서로 정렬.
    
    Args:
        pts: shape (4, 2) 의 점 좌표
    
    Returns:
        정렬된 점 좌표 (4, 2) float32
    """
    rect = np.zeros((4, 2), dtype='float32')
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]   # 좌상 (x+y 최소)
    rect[2] = pts[np.argmax(s)]   # 우하 (x+y 최대)
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # 우상 (y-x 최소)
    rect[3] = pts[np.argmax(diff)]  # 좌하 (y-x 최대)
    return rect
