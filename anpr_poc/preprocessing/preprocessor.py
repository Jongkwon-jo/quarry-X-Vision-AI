"""preprocessor.py — 번호판 정렬과 화질 개선."""
from typing import Optional

import cv2
import numpy as np

from ..config import PreprocessingConfig
from ..schemas import PreprocessedPlate
from ..utils.geometry import order_corners


class PlatePreprocessor:
    """검출된 번호판 영역을 OCR에 적합하게 가공."""
    
    def __init__(self, config: PreprocessingConfig):
        self.config = config
    
    def process(self, plate_crop: np.ndarray) -> PreprocessedPlate:
        """번호판 크롭 → 정렬 → 화질 개선."""
        rectified, success = self.rectify(plate_crop)
        enhanced = self.enhance(rectified)
        
        return PreprocessedPlate(
            rectified=rectified,
            enhanced=enhanced,
            rectification_success=success,
        )
    
    def rectify(self, plate_crop: np.ndarray) -> tuple:
        """4점 코너 기반 원근 보정.
        
        실패 시 단순 리사이즈로 fallback.
        
        Returns:
            (rectified_image, success_flag)
        """
        target_w, target_h = self.config.target_size
        gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, self.config.canny_low, self.config.canny_high)
        
        contours, _ = cv2.findContours(
            edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        if contours:
            largest = max(contours, key=cv2.contourArea)
            epsilon = 0.02 * cv2.arcLength(largest, True)
            approx = cv2.approxPolyDP(largest, epsilon, True)
            
            if len(approx) == 4:
                pts = order_corners(approx.reshape(4, 2).astype('float32'))
                dst = np.array(
                    [[0, 0], [target_w, 0],
                     [target_w, target_h], [0, target_h]],
                    dtype='float32'
                )
                M = cv2.getPerspectiveTransform(pts, dst)
                warped = cv2.warpPerspective(plate_crop, M, (target_w, target_h))
                return warped, True
        
        # Fallback: 단순 리사이즈
        resized = cv2.resize(plate_crop, (target_w, target_h))
        return resized, False
    
    def enhance(self, plate_image: np.ndarray) -> np.ndarray:
        """석산 환경 (먼지·역광)에 강한 전처리.
        
        CLAHE로 명암 평준화 + Bilateral Filter로 노이즈 제거.
        그레이스케일을 반환 (OCR에 적합).
        """
        gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
        
        clahe = cv2.createCLAHE(
            clipLimit=self.config.clahe_clip_limit,
            tileGridSize=self.config.clahe_tile_size,
        )
        enhanced = clahe.apply(gray)
        
        denoised = cv2.bilateralFilter(
            enhanced,
            self.config.bilateral_d,
            self.config.bilateral_sigma_color,
            self.config.bilateral_sigma_space,
        )
        return denoised
