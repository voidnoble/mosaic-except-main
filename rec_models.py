"""인식 백본 교체 모듈 (rec_models.py)

- ArcFace: InsightFace buffalo_l 기본 (w600k_r50, ResNet50@WebFace600K)
- AdaFace: IR-101@MS1MV2 — 저품질(저해상도·블러) 얼굴에 강함
  가중치: https://github.com/yakhyo/adaface-onnx (MIT)

AdaFace 전처리: RGB 112x112, (x/255 - 0.5) / 0.5 정규화, 512-d 임베딩.
DB는 백본별로 따로 구축해야 함 (임베딩 공간이 다름).

사용법:
  app = proto.get_app(...)
  swap_recognition_model(app, "models/adaface_ir_101.onnx")  # 또는 None=ArcFace
"""
import cv2
import numpy as np
import onnxruntime as ort


class AdaFaceONNX:
    """insightface rec 모델과 동일한 .get(img, face) 인터페이스."""

    def __init__(self, model_path, providers=None):
        self.session = ort.InferenceSession(
            model_path, providers=providers or ["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name

    def get(self, img, face):
        from insightface.utils import face_align
        kps = getattr(face, "kps", None)
        if kps is None:
            # 5점 랜드마크가 없으면 bbox 중심 기준 근사 정렬
            x1, y1, x2, y2 = face.bbox
            w, h = x2 - x1, y2 - y1
            kps = np.array([
                [x1 + w * 0.3, y1 + h * 0.35],
                [x1 + w * 0.7, y1 + h * 0.35],
                [x1 + w * 0.5, y1 + h * 0.55],
                [x1 + w * 0.35, y1 + h * 0.75],
                [x1 + w * 0.65, y1 + h * 0.75]], dtype=np.float32)
        aimg, _ = face_align.norm_crop2(img, kps, image_size=112)
        rgb = cv2.cvtColor(aimg, cv2.COLOR_BGR2RGB).astype(np.float32)
        blob = ((rgb / 255.0) - 0.5) / 0.5
        blob = blob.transpose(2, 0, 1)[None]
        emb = self.session.run(None, {self.input_name: blob})[0][0]
        n = np.linalg.norm(emb)
        face.embedding = (emb / max(n, 1e-9)).astype(np.float32)
        return face.embedding


def swap_recognition_model(face_app, adaface_path=None, providers=None):
    """face_app.models['recognition']을 AdaFace로 교체. adaface_path=None이면
    원래 ArcFace로 복원."""
    if adaface_path is None:
        # 원래 모델로 복원: FaceAnalysis를 다시 prepare할 수 없으므로 보관 필요
        orig = getattr(face_app, "_orig_recognition", None)
        if orig is not None:
            face_app.models["recognition"] = orig
        return "arcface"
    if not hasattr(face_app, "_orig_recognition"):
        face_app._orig_recognition = face_app.models["recognition"]
    face_app.models["recognition"] = AdaFaceONNX(adaface_path,
                                                providers=providers)
    return "adaface"
