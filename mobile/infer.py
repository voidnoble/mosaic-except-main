"""온디바이스 추론: YuNet 검출 + ArcFace 임베딩 (onnxruntime, insightface 불필요).

의존성: onnxruntime, opencv-python-headless, numpy
사용법:
  python infer.py --models ./models --db face_db.npz --input in.jpg --output out.jpg
  python infer.py --models ./models --db face_db.npz --input in.mp4 --output out.mp4
"""
import argparse
import os

import cv2
import numpy as np
import onnxruntime as ort

# ArcFace 표준 5점 템플릿 (112x112)
ARCFACE_DST = np.array([
    [38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
    [41.5493, 92.3655], [70.7299, 92.2041]], dtype=np.float64)


def umeyama(src, dst):
    """Umeyama 유사변환 추정 (skimage 없이 numpy로). src→dst, 반환 2x3 행렬."""
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    n = src.shape[0]
    mu_s, mu_d = src.mean(0), dst.mean(0)
    s_c, d_c = src - mu_s, dst - mu_d
    var = (s_c ** 2).sum() / n
    cov = d_c.T @ s_c / n
    u, d, vt = np.linalg.svd(cov)
    s_mat = np.eye(2)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        s_mat[1, 1] = -1
    r = u @ s_mat @ vt
    scale = (d * np.diag(s_mat)).sum() / var
    t = mu_d - scale * r @ mu_s
    return np.hstack([scale * r, t[:, None]]).astype(np.float32)


def align_face(img, kps):
    M = umeyama(kps, ARCFACE_DST)
    return cv2.warpAffine(img, M, (112, 112), borderValue=0.0)


class MobilePipeline:
    def __init__(self, models_dir, db_path, threshold=0.4):
        self.det = cv2.FaceDetectorYN.create(
            os.path.join(models_dir, "face_detection_yunet_2023mar.onnx"),
            "", (320, 320))
        self.det.setScoreThreshold(0.5)
        self.det.setNMSThreshold(0.3)
        self.rec = ort.InferenceSession(
            os.path.join(models_dir, "w600k_r50.onnx"),
            providers=["CPUExecutionProvider"])
        self.rec_in = self.rec.get_inputs()[0].name
        d = np.load(db_path, allow_pickle=True)
        names = [str(n) for n in d["names"]]
        key = "embeddings" if "embeddings" in d else "embs"
        embs = d[key].astype(np.float32)
        embs /= np.linalg.norm(embs, axis=1, keepdims=True) + 1e-9
        self.names, self.embs, self.threshold = names, embs, threshold
        self._size = None

    def _embed(self, img, kps):
        face = align_face(img, kps)                      # BGR 112x112
        blob = (face[:, :, ::-1].astype(np.float32) - 127.5) / 127.5
        blob = blob.transpose(2, 0, 1)[None]
        emb = self.rec.run(None, {self.rec_in: blob})[0][0]
        return emb / (np.linalg.norm(emb) + 1e-9)

    def process(self, img):
        h, w = img.shape[:2]
        if self._size != (w, h):
            self.det.setInputSize((w, h))
            self._size = (w, h)
        _, raw = self.det.detect(img)
        out = img
        if raw is None:
            return out
        for r in raw:
            x, y, bw, bh = r[:4].astype(int)
            kps = r[4:14].reshape(5, 2).astype(np.float64)
            emb = self._embed(img, kps)
            sims = self.embs @ emb
            i = int(np.argmax(sims))
            name = self.names[i] if sims[i] >= self.threshold else "unknown"
            x1, y1 = max(x, 0), max(y, 0)
            x2, y2 = min(x + bw, w), min(y + bh, h)
            color = (0, 255, 0) if name != "unknown" else (0, 0, 255)
            if name == "unknown" and x2 - x1 > 12 and y2 - y1 > 12:
                roi = out[y1:y2, x1:x2]
                small = cv2.resize(roi, ((x2 - x1) // 12, (y2 - y1) // 12),
                                   interpolation=cv2.INTER_AREA)
                out[y1:y2, x1:x2] = cv2.resize(
                    small, (x2 - x1, y2 - y1),
                    interpolation=cv2.INTER_NEAREST)
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            cv2.putText(out, f"{name} {sims[i]:.2f}", (x1, max(y1 - 6, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        return out


def main():
    p = argparse.ArgumentParser(description="온디바이스 얼굴 모자이크 추론")
    p.add_argument("--models", default="models")
    p.add_argument("--db", required=True)
    p.add_argument("--input", required=True)
    p.add_argument("--output", default="mobile_out.jpg")
    p.add_argument("--threshold", type=float, default=0.4)
    a = p.parse_args()

    pipe = MobilePipeline(a.models, a.db, a.threshold)
    if a.input.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
        cap = cv2.VideoCapture(a.input)
        fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(a.output, cv2.VideoWriter_fourcc(*"mp4v"),
                              fps, (w, h))
        n = 0
        while True:
            ret, img = cap.read()
            if not ret:
                break
            out.write(pipe.process(img))
            n += 1
        cap.release()
        out.release()
        print(f"{n} 프레임 처리 → {a.output}")
    else:
        img = cv2.imread(a.input)
        cv2.imwrite(a.output, pipe.process(img))
        print(f"저장: {a.output}")


if __name__ == "__main__":
    main()
