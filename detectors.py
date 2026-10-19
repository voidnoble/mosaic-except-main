"""Phase 3: 검출기 추상화 (detectors.py)

- SCRFDDetector: InsightFace buffalo_l (정확도 우선, landmark_2d_106 제공)
- YuNetDetector:  OpenCV DNN YuNet (모바일/저전력 분기, ~17배 빠름, 5점 랜드마크)
  YuNet 얼굴의 임베딩은 ArcFace rec 모델로 별도 추출 (kps 기반 정렬).

공통 인터페이스:
  det = make_detector("scrfd" | "yunet", ...)
  faces = det.detect(img)          # [Face] (bbox, det_score, kps, embedding or None)
  det.embed(img, faces)            # embedding 비어 있으면 채움
"""
import cv2
import numpy as np

import prototype_scrfd_arcface as proto

YUNET_DEFAULT = "models/face_detection_yunet_2023mar.onnx"


class Face:
    """insightface Face 호환 경량 객체."""
    def __init__(self, bbox, det_score, kps=None, embedding=None, lm106=None):
        self.bbox = np.asarray(bbox, dtype=np.float32)
        self.det_score = float(det_score)
        self.kps = kps  # (5,2) or None
        self.embedding = embedding
        self.landmark_2d_106 = lm106


class SCRFDDetector:
    name = "scrfd"

    def __init__(self, det_size=640, providers=None):
        self.app = proto.get_app(det_size=(det_size, det_size),
                                 providers=providers)

    def detect(self, img):
        out = []
        for f in self.app.get(img):
            out.append(Face(f.bbox, f.det_score, kps=getattr(f, "kps", None),
                            embedding=getattr(f, "embedding", None),
                            lm106=getattr(f, "landmark_2d_106", None)))
        return out

    def embed(self, img, faces):
        pass  # detect 시점에 이미 임베딩 포함


class YuNetDetector:
    name = "yunet"

    def __init__(self, model_path=YUNET_DEFAULT, score_thresh=0.5,
                 providers=None):
        # rec 모델은 임베딩 추출용으로만 로드 (검출은 YuNet)
        self.app = proto.get_app(det_size=(320, 320), providers=providers)
        from insightface.app.common import Face as IFFace
        self._IFFace = IFFace
        self.model_path = model_path
        self.score_thresh = score_thresh
        self._det = None
        self._size = None

    def _ensure(self, img):
        h, w = img.shape[:2]
        if self._det is None or self._size != (w, h):
            self._det = cv2.FaceDetectorYN.create(
                self.model_path, "", (w, h))
            self._det.setScoreThreshold(self.score_thresh)
            self._det.setNMSThreshold(0.3)
            self._size = (w, h)

    def detect(self, img):
        self._ensure(img)
        _, raw = self._det.detect(img)
        out = []
        if raw is not None:
            for r in raw:
                x, y, w, h = r[:4]
                kps = r[4:14].reshape(5, 2).astype(np.float32)
                out.append(Face((x, y, x + w, y + h), r[14], kps=kps))
        return out

    def embed(self, img, faces):
        rec = self.app.models["recognition"]  # 동적 조회 (백본 교체 대응)
        for f in faces:
            if f.embedding is not None or f.kps is None:
                continue
            iff = self._IFFace(bbox=f.bbox, kps=f.kps,
                               det_score=f.det_score)
            try:
                rec.get(img, iff)
                f.embedding = iff.embedding
            except Exception:
                f.embedding = None


def make_detector(kind, **kw):
    if kind == "yunet":
        kw.pop("det_size", None)  # YuNet은 입력 해상도 자동 (원본 크기)
        return YuNetDetector(**kw)
    kw.pop("model_path", None)
    return SCRFDDetector(**kw)
