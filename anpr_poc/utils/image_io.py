"""image_io.py — 한글 경로 호환 이미지 입출력."""
from pathlib import Path
from typing import Optional, Union

import cv2
import numpy as np


def load_image(path: Union[str, Path]) -> Optional[np.ndarray]:
    """이미지를 BGR 형태로 로드. 한글 경로도 안전하게 처리.
    
    Windows의 cv2.imread는 한글 경로에서 None을 반환하는 경우가 있어
    np.fromfile + cv2.imdecode로 폴백한다.
    """
    path = str(path)
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is not None:
        return img
    
    try:
        buf = np.fromfile(path, dtype=np.uint8)
        if buf.size == 0:
            return None
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        return img
    except Exception:
        return None


def save_image(image: np.ndarray, path: Union[str, Path]) -> bool:
    """이미지 저장. 한글 경로도 안전하게 처리."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # 확장자 추출
    ext = path.suffix.lower() or '.jpg'
    
    try:
        success, buf = cv2.imencode(ext, image)
        if not success:
            return False
        buf.tofile(str(path))
        return True
    except Exception:
        # fallback
        return cv2.imwrite(str(path), image)
