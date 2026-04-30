"""plate_detector.py — Ultralytics YOLO 기반 번호판 검출."""
from typing import List, Optional

import numpy as np

from ..config import DetectionConfig
from ..schemas import Detection


class PlateDetector:
    """파인튜닝된 YOLO 모델로 번호판 영역을 검출."""
    
    def __init__(self, config: DetectionConfig):
        self.config = config
        self._model = None  # lazy load
    
    @property
    def model(self):
        """첫 호출 시 모델 로드 (시작 속도 최적화)."""
        if self._model is None:
            from ultralytics import YOLO
            self._model = YOLO(self.config.weights_path)
        return self._model
    
    def detect(self, image: np.ndarray) -> List[Detection]:
        """이미지에서 번호판 모두 검출.
        
        Args:
            image: BGR 이미지 (np.ndarray)
        
        Returns:
            Detection 리스트 (신뢰도 내림차순)
        """
        results = self.model(
            image,
            conf=self.config.conf_threshold,
            iou=self.config.iou_threshold,
            device=self.config.device,
            verbose=False,
        )[0]
        
        detections = []
        names = self.model.names
        h, w = image.shape[:2]
        
        for box in results.boxes:
            cls_id = int(box.cls[0])
            cls_name = names.get(cls_id, str(cls_id)) if isinstance(names, dict) else names[cls_id]
            
            # 번호판 클래스만 채택 (vehicle은 제외)
            if cls_name != self.config.plate_class_name:
                continue
            
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            
            # 이미지 경계로 클램프
            x1, x2 = max(0, x1), min(w, x2)
            y1, y2 = max(0, y1), min(h, y2)
            
            if x2 <= x1 or y2 <= y1:
                continue
            
            detections.append(Detection(
                bbox=(int(x1), int(y1), int(x2), int(y2)),
                confidence=float(box.conf[0]),
                crop=image[y1:y2, x1:x2].copy(),
                class_name=cls_name,
            ))
        
        # 신뢰도 내림차순
        detections.sort(key=lambda d: d.confidence, reverse=True)
        return detections
    
    def detect_best(self, image: np.ndarray) -> Optional[Detection]:
        """가장 신뢰도 높은 번호판 하나만 반환."""
        detections = self.detect(image)
        return detections[0] if detections else None
