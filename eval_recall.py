"""인식 백본 평가 (eval_recall.py)

AI-Hub GT 데이터셋은 회원 가입·이용 승인이 필요해 이 환경에서 접근 불가.
LFW 원본 서버(UMass)도 이 환경에서 차단됨.
대신 다음 프로토콜로 동등한 지표(accuracy, TAR@FAR, 열화 강건성)를 측정:

1. Genuine/impostor 분포: 동일 인물 증강(augmentation) 쌍 vs 타인 쌍
   → verification accuracy, TAR@FAR=1%
2. 저화질 강건성: 갤러리=원본, 쿼리=열화(축소/블러/JPEG) → rank-1 정확도
   (AdaFace가 저품질에 강하다는 논문 주장 검증)

사용법:
  python eval_recall.py --gallery data/test_images --backbones arcface,adaface
"""
import argparse
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import rec_models
from detectors import make_detector


def embed_face(det, img):
    faces = det.detect(img)
    if not faces:
        return None
    f = max(faces, key=lambda x: x.det_score)
    det.embed(img, [f])
    e = f.embedding
    if e is None:
        return None
    return e / max(np.linalg.norm(e), 1e-9)


def augment(img, seed):
    """동일 인물의 변형 이미지 (genuine 쌍용)."""
    rng = np.random.default_rng(seed)
    h, w = img.shape[:2]
    out = img
    if rng.random() < 0.5:
        out = cv2.flip(out, 1)
    ang = rng.uniform(-8, 8)
    M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, rng.uniform(0.92, 1.05))
    out = cv2.warpAffine(out, M, (w, h), borderValue=0)
    if rng.random() < 0.5:  # 밝기 지터
        out = np.clip(out.astype(np.float32) * rng.uniform(0.85, 1.15),
                      0, 255).astype(np.uint8)
    return out


def verification_test(det, gallery_dir, n_gen=60, n_imp=60):
    """증강 genuine 쌍 + 타인 impostor 쌍 → accuracy, TAR@FAR."""
    persons = sorted(d for d in os.listdir(gallery_dir)
                     if os.path.isdir(os.path.join(gallery_dir, d)))
    base = {}
    for p in persons:
        for fn in sorted(os.listdir(os.path.join(gallery_dir, p)))[:1]:
            img = cv2.imread(os.path.join(gallery_dir, p, fn))
            if img is not None:
                base[p] = img
    sims, labels = [], []
    rng = np.random.default_rng(7)
    # genuine
    for i in range(n_gen):
        p = persons[i % len(persons)]
        e1 = embed_face(det, base[p])
        e2 = embed_face(det, augment(base[p], 1000 + i))
        if e1 is None or e2 is None:
            continue
        sims.append(float(e1 @ e2))
        labels.append(True)
    # impostor
    for i in range(n_imp):
        p1, p2 = rng.choice(persons, 2, replace=False)
        e1 = embed_face(det, base[p1])
        e2 = embed_face(det, augment(base[p2], 2000 + i))
        if e1 is None or e2 is None:
            continue
        sims.append(float(e1 @ e2))
        labels.append(False)
    sims = np.array(sims)
    labels = np.array(labels)
    thrs = np.linspace(0.1, 0.9, 81)
    accs = [((sims >= t) == labels).mean() for t in thrs]
    bi = int(np.argmax(accs))
    gs, im = sims[labels], sims[~labels]
    far1_thr = np.quantile(im, 0.99) if len(im) else 1.0
    tar = (gs >= far1_thr).mean() if len(gs) else 0.0
    return {"n": len(labels), "acc": accs[bi], "thr": thrs[bi],
            "tar@far1e-2": tar,
            "gen_mean": float(gs.mean()) if len(gs) else 0.0,
            "imp_mean": float(im.mean()) if len(im) else 0.0}


def degrade(img, kind):
    h, w = img.shape[:2]
    if kind == "x56":
        small = cv2.resize(img, (56, 56), interpolation=cv2.INTER_AREA)
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    if kind == "x28":
        small = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    if kind == "blur":
        return cv2.GaussianBlur(img, (15, 15), 0)
    if kind == "jpeg":
        _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 10])
        return cv2.imdecode(buf, cv2.IMREAD_COLOR)
    return img


def degradation_test(det, gallery_dir):
    persons = sorted(d for d in os.listdir(gallery_dir)
                     if os.path.isdir(os.path.join(gallery_dir, d)))
    g_embs, g_names = [], []
    for p in persons:
        for fn in sorted(os.listdir(os.path.join(gallery_dir, p)))[:1]:
            img = cv2.imread(os.path.join(gallery_dir, p, fn))
            e = embed_face(det, img) if img is not None else None
            if e is not None:
                g_embs.append(e)
                g_names.append(p)
    g_embs = np.stack(g_embs)
    res = {}
    for kind in ["orig", "x56", "x28", "blur", "jpeg"]:
        ok, n = 0, 0
        for p in persons:
            for fn in sorted(os.listdir(os.path.join(gallery_dir, p)))[:1]:
                img = cv2.imread(os.path.join(gallery_dir, p, fn))
                if img is None:
                    continue
                q = embed_face(det, degrade(img, kind))
                if q is None:
                    continue
                pred = g_names[int(np.argmax(g_embs @ q))]
                ok += (pred == p)
                n += 1
        res[kind] = ok / max(n, 1)
    return res


def build_detector(backbone, providers):
    det = make_detector("scrfd", providers=providers)
    if backbone == "adaface":
        adaface = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "models", "adaface_ir_101.onnx")
        if not os.path.exists(adaface):
            raise FileNotFoundError(f"AdaFace 가중치 없음: {adaface}")
        rec_models.swap_recognition_model(det.app, adaface,
                                          providers=providers)
    return det


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gallery", default="data/test_images")
    p.add_argument("--backbones", default="arcface,adaface")
    a = p.parse_args()
    providers = ["CPUExecutionProvider"]
    for bb in a.backbones.split(","):
        bb = bb.strip()
        print(f"\n[{bb}] 평가 중...", flush=True)
        t0 = time.time()
        det = build_detector(bb, providers)
        r = verification_test(det, a.gallery)
        print(f"  verification 정확도: {r['acc']*100:.1f}% "
              f"(thr={r['thr']:.2f}, n={r['n']})")
        print(f"  TAR@FAR=1%: {r['tar@far1e-2']*100:.1f}%")
        print(f"  genuine {r['gen_mean']:.3f} / impostor {r['imp_mean']:.3f}")
        d = degradation_test(det, a.gallery)
        print("  열화 강건성(rank-1): " +
              ", ".join(f"{k}={v*100:.0f}%" for k, v in d.items()))
        print(f"  소요: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
