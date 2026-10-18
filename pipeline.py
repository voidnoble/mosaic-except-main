"""Phase 1 비디오 파이프라인 (기존 main.py의 Video 분기 대체용).

흐름: 검출 → ByteTrack → 신규 트랙만 인식(신원 투표) → 트랙 기반 모자이크
- 기존 baselinewithfacenet/ 모듈(detection, ml_part, util)을 그대로 재사용
- 실행:  python pipeline.py --video in.mp4 --out out.mp4 [--detector retinaface]

전제: 이 파일을 baselinewithfacenet/ 디렉토리 안에 복사하거나,
      PYTHONPATH에 baselinewithfacenet/ 를 추가하고 tracking/ 을 같은 위치에 둘 것.
"""
import argparse
import os
import sys
import time

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tracking import BYTETracker
from tracking.track import TrackState

# 기존 모듈 재사용
import ml_part as ML
from detection import load_face_db, mtcnn_get_embeddings, mtcnn_recognition
from util import Mosaic, DrawRectImg


def detect_with_scores(img_det, args, model_args):
    """ML.Detection 래퍼 → [(tlwh, score)].

    retinaface: (bboxes, scores) 반환. mtcnn: 점수가 없어 0.99로 둠
    (전부 high-conf로 취급 — mtcnn 분기는 2단계 매칭 이득이 작음).
    """
    bboxes, scores = ML.Detection(img_det, args, model_args)
    if bboxes is None or len(bboxes) == 0:
        return []
    if scores is None:
        scores = np.full(len(bboxes), 0.99, dtype=np.float32)
    dets = []
    for (x1, y1, x2, y2), s in zip(bboxes, scores):
        w, h = x2 - x1, y2 - y1
        if w <= 0 or h <= 0:
            continue
        dets.append((np.array([x1, y1, w, h], dtype=np.float32), float(s)))
    return dets


def recognize_one(img_det, tlbr, args, model_args):
    """단일 박스에 대해 인식 수행 → 이름 또는 'unknown'."""
    x1, y1, x2, y2 = tlbr
    bbox = np.array([[x1, y1, x2, y2]], dtype=np.float32)
    try:
        _, emb = mtcnn_get_embeddings(
            model_args["Mtcnn"], model_args["Recognition"],
            img_det, bbox, model_args["Device"],
        )
        face_ids, _ = mtcnn_recognition(
            model_args["Face_db"], emb, args["RECOG_THRESHOLD"]
        )
        return face_ids[0] if face_ids else "unknown"
    except Exception:
        return "unknown"  # fail-closed: 인식 실패 시 모자이크


def process_video(args, model_args, tracker):
    cap = cv2.VideoCapture(args["VIDEO_PATH"])
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {args['VIDEO_PATH']}")

    # 원본 규격 유지 (기존 1280x720·XVID 고정 문제 해결)
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(args["SAVE_PATH"], fourcc, fps, (w, h))

    frame_idx = 0
    t_total = time.time()
    stats = {"detect_ms": 0.0, "track_ms": 0.0, "recog_ms": 0.0,
             "recog_calls": 0, "frames": 0}

    while True:
        ret, bgr = cap.read()
        if not ret:
            break

        # --- 검출 입력 준비 (기존 색상 규칙 유지) ---
        if args["DETECTOR"] == "mtcnn":
            img_det = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        else:  # retinaface: BGR 그대로
            img_det = bgr

        t0 = time.time()
        dets = detect_with_scores(img_det, args, model_args)
        stats["detect_ms"] += (time.time() - t0) * 1000

        t0 = time.time()
        tracks = tracker.update(dets)
        stats["track_ms"] += (time.time() - t0) * 1000

        # --- 신규 트랙만 인식 (신원 투표) ---
        for trk in tracks:
            if trk.state == TrackState.Lost or trk.identity_locked:
                continue
            t0 = time.time()
            x1, y1, x2, y2 = trk.tlbr
            name = recognize_one(img_det, (x1, y1, x2, y2), args, model_args)
            trk.vote(name)
            stats["recog_ms"] += (time.time() - t0) * 1000
            stats["recog_calls"] += 1

        # --- 트랙 기반 모자이크 (Lost 트랙도 유지 → 누락 방지) ---
        render_boxes, face_ids = [], []
        for trk in tracks:
            x1, y1, x2, y2 = trk.render_tlbr  # EMA 스무딩 박스
            render_boxes.append([x1, y1, x2, y2])
            # 미확정(None)도 'unknown' 취급 → fail-closed
            face_ids.append(trk.identity if trk.identity else "unknown")

        if render_boxes:
            bgr = Mosaic(bgr, np.array(render_boxes), face_ids, n=10)
            bgr = DrawRectImg(bgr, np.array(render_boxes), face_ids)

        out.write(bgr)
        frame_idx += 1
        stats["frames"] += 1

    cap.release()
    out.release()
    dt = time.time() - t_total
    print(f"done. {stats['frames']} frames in {dt:.1f}s "
          f"({stats['frames']/max(dt,1e-6):.1f} FPS)")
    print(f"detect {stats['detect_ms']/stats['frames']:.1f} ms/frame, "
          f"track {stats['track_ms']/stats['frames']:.1f} ms/frame, "
          f"recognition {stats['recog_calls']} calls total "
          f"({stats['recog_ms']/max(stats['recog_calls'],1):.1f} ms/call)")


def build_args():
    p = argparse.ArgumentParser(description="MOFY Phase 1 pipeline")
    p.add_argument("--video", required=True, help="입력 영상")
    p.add_argument("--out", default="output.mp4", help="출력 영상")
    p.add_argument("--detector", default="retinaface",
                   choices=["mtcnn", "retinaface"])
    p.add_argument("--recog_thrs", type=float, default=0.8)
    p.add_argument("--known_images", default="../data/test_images",
                   help="등록 인물 이미지 폴더")
    p.add_argument("--face_db", default="./database/face_db")
    p.add_argument("--img_db", default="./database/img_db")
    return p.parse_args()


def main():
    ns = build_args()
    # 기존 init() 로직 재사용을 위해 main.py의 init을 import
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "baselinewithfacenet"))
    from main import init as baseline_init

    args = {
        "DETECTOR": ns.detector,
        "RECOG_THRESHOLD": ns.recog_thrs,
        "DEBUG_MODE": False,
        "PROCESS_TARGET": "Video",
        "VIDEO_PATH": ns.video,
        "SAVE_PATH": ns.out,
    }
    model_args = baseline_init({
        "DETECTOR": ns.detector,
        "DEBUG_MODE": False,
    })
    # face_db 경로를 CLI 인자로 override (init은 "../data/test_images" 하드코딩)
    face_db_path = ns.face_db + ("_BGR" if ns.detector == "retinaface" else "")
    model_args["Face_db"] = load_face_db(
        ns.known_images, face_db_path, ns.img_db,
        model_args["Device"],
        {"DETECTOR": ns.detector, "DEBUG_MODE": False},
        model_args,
    )
    tracker = BYTETracker(track_thresh=0.5, max_time_lost=30)
    process_video(args, model_args, tracker)


if __name__ == "__main__":
    main()
