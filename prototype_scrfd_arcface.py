"""SCRFD + ArcFace 인식 파이프라인 프로토타입 (Phase 2 핵심).

InsightFace `buffalo_l` 팩 사용: SCRFD 검출 + ArcFace(ResNet50@WebFace600K) 인식.
기존 파이프라인(InceptionResnetV1 + L2거리 + 임계값 0.8)을 대체하는 검증용 스크립트.

사용법:
  python prototype_scrfd_arcface.py build-db --known_dir <인물별 폴더> --db face_db.npz
  python prototype_scrfd_arcface.py calibrate --known_dir <인물별 폴더>
  python prototype_scrfd_arcface.py image --db face_db.npz --input in.jpg --output out.jpg
  python prototype_scrfd_arcface.py video --db face_db.npz --input in.mp4 --output out.mp4 [--track]

설치: pip install insightface onnxruntime opencv-python-headless numpy
주의: InsightFace 사전학습 가중치는 비상업 연구용 라이선스.
"""
import argparse
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def get_app(det_size=(640, 640), providers=None):
    from insightface.app import FaceAnalysis
    providers = providers or ["CPUExecutionProvider"]
    app = FaceAnalysis(name="buffalo_l", providers=providers)
    # ctx_id=-1 → CPU, det_size 클수록 작은 얼굴 검출↑·속도↓
    app.prepare(ctx_id=-1, det_size=det_size)
    return app


def l2norm(v):
    return v / max(np.linalg.norm(v), 1e-9)


# ---------------------------------------------------------------- DB 구축
def build_db(app, known_dir, db_path, min_det_score=0.5):
    """인물별 폴더 → 품질 가중 평균 임베딩. 저장: npz(names, embeddings)."""
    names, embs = [], []
    for person in sorted(os.listdir(known_dir)):
        pdir = os.path.join(known_dir, person)
        if not os.path.isdir(pdir):
            continue
        vecs, weights = [], []
        for fn in sorted(os.listdir(pdir)):
            if not fn.lower().endswith(IMG_EXTS):
                continue
            img = cv2.imread(os.path.join(pdir, fn))
            if img is None:
                continue
            for f in app.get(img):
                if f.det_score < min_det_score:
                    continue
                # 품질 가중: 검출 신뢰도 × 얼굴 크기(정규화)
                x1, y1, x2, y2 = f.bbox
                area = max(x2 - x1, 0) * max(y2 - y1, 0)
                w = float(f.det_score) * min(area / (112 * 112), 1.0)
                vecs.append(f.embedding.astype(np.float64))
                weights.append(w)
        if not vecs:
            print(f"[경고] {person}: 유효 얼굴 없음 — DB에서 제외")
            continue
        vecs = np.stack(vecs)
        weights = np.array(weights)
        rep = l2norm((vecs * weights[:, None]).sum(axis=0))
        names.append(person)
        embs.append(rep.astype(np.float32))
        print(f"{person}: {len(vecs)}장 → 대표 임베딩 생성")
    np.savez(db_path, names=np.array(names),
             embeddings=np.stack(embs) if embs else np.zeros((0, 512)))
    print(f"DB 저장: {db_path} ({len(names)}명)")
    return db_path


def load_db(db_path):
    z = np.load(db_path, allow_pickle=True)
    return [str(n) for n in z["names"]], z["embeddings"].astype(np.float32)


# ---------------------------------------------------------------- 임계값 보정
def calibrate(app, known_dir, min_det_score=0.5):
    """known_dir 이미지들로 genuine/impostor 쌍을 만들어 FAR/FRR 테이블 출력."""
    per_person = {}
    for person in sorted(os.listdir(known_dir)):
        pdir = os.path.join(known_dir, person)
        if not os.path.isdir(pdir):
            continue
        vecs = []
        for fn in sorted(os.listdir(pdir)):
            if not fn.lower().endswith(IMG_EXTS):
                continue
            img = cv2.imread(os.path.join(pdir, fn))
            if img is None:
                continue
            for f in app.get(img):
                if f.det_score >= min_det_score:
                    vecs.append(f.embedding)
        if vecs:
            # 대표 임베딩과 동일하게 쿼리도 정규화 후 페어 구성
            per_person[person] = [l2norm(v.astype(np.float64)) for v in vecs]

    genuine, impostor = [], []
    persons = sorted(per_person)
    for p in persons:
        v = per_person[p]
        for i in range(len(v)):
            for j in range(i + 1, len(v)):
                genuine.append(float(np.dot(v[i], v[j])))
    for i in range(len(persons)):
        for j in range(i + 1, len(persons)):
            for a in per_person[persons[i]][:3]:
                for b in per_person[persons[j]][:3]:
                    impostor.append(float(np.dot(a, b)))
    genuine = np.array(genuine)
    impostor = np.array(impostor)
    print(f"genuine 쌍 {len(genuine)}개, impostor 쌍 {len(impostor)}개")
    if len(genuine) < 2 or len(impostor) < 2:
        print("쌍이 부족해 보정 불가 — 인물당 2장 이상 권장")
        return

    print(f"{'thr':>6} {'FAR':>8} {'FRR':>8}")
    best = None
    for thr in np.arange(0.20, 0.81, 0.05):
        far = float((impostor >= thr).mean())
        frr = float((genuine < thr).mean())
        print(f"{thr:6.2f} {far:8.3f} {frr:8.3f}")
        if best is None or abs(far - frr) < abs(best[1] - best[2]):
            best = (thr, far, frr)
    print(f"제안 임계값(EER 근처): {best[0]:.2f} (FAR={best[1]:.3f}, FRR={best[2]:.3f})")
    print("운영 팁: 누락(모자이크 안 됨)이 치명적이면 임계값을 제안값보다 낮게(보수적으로) 설정")


# ---------------------------------------------------------------- 인식
def recognize_faces(faces, names, db_embs, threshold,
                    min_det_score=0.5, min_face_size=20):
    """각 얼굴 → (이름|'unknown', 유사도). 품질 미달은 판정 보류 → 'unknown'."""
    out = []
    for f in faces:
        x1, y1, x2, y2 = f.bbox
        if f.det_score < min_det_score or min(x2 - x1, y2 - y1) < min_face_size:
            out.append(("unknown", 0.0))  # 판정 보류 → fail-closed
            continue
        q = l2norm(f.embedding.astype(np.float64)).astype(np.float32)
        sims = db_embs @ q  # 코사인 유사도 (양쪽 정규화)
        i = int(np.argmax(sims))
        if sims[i] >= threshold:
            out.append((names[i], float(sims[i])))
        else:
            out.append(("unknown", float(sims[i])))
    return out


def mosaic(img, boxes, n=12):
    out = img
    for x1, y1, x2, y2 in boxes:
        x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
        x1, y1 = max(x1, 0), max(y1, 0)
        x2, y2 = min(x2, img.shape[1]), min(y2, img.shape[0])
        if x2 - x1 < n or y2 - y1 < n:
            continue
        roi = out[y1:y2, x1:x2]
        small = cv2.resize(roi, ((x2 - x1) // n, (y2 - y1) // n),
                           interpolation=cv2.INTER_AREA)
        out[y1:y2, x1:x2] = cv2.resize(small, (x2 - x1, y2 - y1),
                                       interpolation=cv2.INTER_NEAREST)
    return out


def annotate(img, faces, results):
    for f, (name, sim) in zip(faces, results):
        x1, y1, x2, y2 = map(int, f.bbox)
        color = (0, 255, 0) if name != "unknown" else (0, 0, 255)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, f"{name} {sim:.2f}", (x1, max(y1 - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return img


def process_image(app, args):
    names, db_embs = load_db(args.db)
    img = cv2.imread(args.input)
    faces = app.get(img)
    t0 = time.time()
    results = recognize_faces(faces, names, db_embs, args.threshold)
    print(f"검출 {len(faces)}명, 인식 {(time.time()-t0)*1000:.1f} ms")
    boxes = [f.bbox for f, (n, _) in zip(faces, results) if n == "unknown"]
    out = mosaic(img, boxes)
    out = annotate(out, faces, results)
    cv2.imwrite(args.output, out)
    for (name, sim), f in zip(results, faces):
        print(f"  {name} (sim={sim:.3f}, det={f.det_score:.2f})")
    print(f"저장: {args.output}")


def process_video(app, args):
    names, db_embs = load_db(args.db)
    cap = cv2.VideoCapture(args.input)
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"),
                          fps, (w, h))
    tracker = None
    if args.track:
        from tracking import BYTETracker
        tracker = BYTETracker(track_thresh=0.5, max_time_lost=30)

    n_frames, t_all, recog_calls = 0, time.time(), 0
    while True:
        ret, img = cap.read()
        if not ret:
            break
        faces = app.get(img)
        if tracker is None:
            results = recognize_faces(faces, names, db_embs, args.threshold)
            recog_calls += len(faces)
            boxes = [f.bbox for f, (n, _) in zip(faces, results)
                     if n == "unknown"]
        else:
            from tracking.track import TrackState
            dets = []
            for f in faces:
                x1, y1, x2, y2 = f.bbox
                dets.append((np.array([x1, y1, x2 - x1, y2 - y1],
                                      dtype=np.float32), float(f.det_score)))
            tracks = tracker.update(dets)
            # 트랙↔얼굴 매칭: IoU 최대 (간단 매칭)
            boxes, annotated = [], []
            for t in tracks:
                if t.state != TrackState.Tracked:
                    continue
                tx1, ty1, tx2, ty2 = t.render_tlbr
                best, biou = None, 0.0
                for f in faces:
                    fx1, fy1, fx2, fy2 = f.bbox
                    ix1, iy1 = max(tx1, fx1), max(ty1, fy1)
                    ix2, iy2 = min(tx2, fx2), min(ty2, fy2)
                    iw, ih = max(ix2 - ix1, 0), max(iy2 - iy1, 0)
                    union = ((tx2 - tx1) * (ty2 - ty1) + (fx2 - fx1) * (fy2 - fy1)
                             - iw * ih)
                    iou = iw * ih / max(union, 1e-6)
                    if iou > biou:
                        best, biou = f, iou
                if not t.identity_locked and best is not None and biou > 0.3:
                    r = recognize_faces([best], names, db_embs,
                                        args.threshold)[0]
                    t.vote(r[0])
                    recog_calls += 1
                ident = t.identity or "unknown"
                annotated.append(((tx1, ty1, tx2, ty2), ident))
                if ident == "unknown":
                    boxes.append((tx1, ty1, tx2, ty2))
            faces, results = [], []
            for (bb, ident) in annotated:
                faces.append(type("F", (), {"bbox": np.array(bb)})())
                results.append((ident, 0.0))
        out.write(annotate(mosaic(img, boxes), faces, results))
        n_frames += 1
    cap.release()
    out.release()
    dt = time.time() - t_all
    print(f"{n_frames} frames, {dt:.1f}s ({n_frames/max(dt,1e-6):.1f} FPS), "
          f"인식 호출 {recog_calls}회, 저장: {args.output}")


def main():
    p = argparse.ArgumentParser(description="SCRFD+ArcFace 프로토타입")
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build-db")
    b.add_argument("--known_dir", required=True)
    b.add_argument("--db", default="face_db.npz")
    b.add_argument("--rec-model", default="arcface",
                   choices=["arcface", "adaface"],
                   help="AdaFace 가중치 필요: models/adaface_ir_101.onnx")
    c = sub.add_parser("calibrate")
    c.add_argument("--known_dir", required=True)
    i = sub.add_parser("image")
    i.add_argument("--db", required=True)
    i.add_argument("--input", required=True)
    i.add_argument("--output", default="out.jpg")
    i.add_argument("--threshold", type=float, default=0.4)
    v = sub.add_parser("video")
    v.add_argument("--db", required=True)
    v.add_argument("--input", required=True)
    v.add_argument("--output", default="out.mp4")
    v.add_argument("--threshold", type=float, default=0.4)
    v.add_argument("--track", action="store_true", help="Phase 1 ByteTrack 연동")
    ns = p.parse_args()

    app = get_app()
    if getattr(ns, "rec_model", "arcface") == "adaface":
        import rec_models
        ckpt = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "models", "adaface_ir_101.onnx")
        rec_models.swap_recognition_model(app, ckpt)
        print("인식 백본: AdaFace IR-101")
    if ns.cmd == "build-db":
        build_db(app, ns.known_dir, ns.db)
    elif ns.cmd == "calibrate":
        calibrate(app, ns.known_dir)
    elif ns.cmd == "image":
        process_image(app, ns)
    elif ns.cmd == "video":
        process_video(app, ns)


if __name__ == "__main__":
    main()
