"""Phase 2 통합 파이프라인 (모델 현대화).

- 검출: SCRFD / 인식: ArcFace (InsightFace `buffalo_l`)
- Phase 1 ByteTrack: 트랙 단위 신원 투표, EMA 스무딩 박스
- 품질 게이팅: det_score/얼굴 크기 미달 → 판정 보류 → 모자이크 (fail-closed)
- 프레임 스킵: K프레임마다 검출, 사이는 Kalman 예측으로 보간
- 임계값: prototype_scrfd_arcface.py calibrate로 정한 값 사용 (코사인 유사도)

사용법:
  python pipeline_v2.py --db face_db.npz --input in.mp4 --output out.mp4
  python pipeline_v2.py --db face_db.npz --input in.mp4 --output out.mp4 \
      --threshold 0.35 --skip 3 --det-size 640 --providers cuda
"""
import argparse
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tracking import BYTETracker
from tracking.track import TrackState
from tracking.matching import iou_batch

import anonymize
import detectors
import prototype_scrfd_arcface as proto
from stream_out import make_writer


def to_dets(faces):
    dets = []
    for f in faces:
        x1, y1, x2, y2 = f.bbox
        w, h = x2 - x1, y2 - y1
        if w <= 0 or h <= 0:
            continue
        dets.append((np.array([x1, y1, w, h], dtype=np.float32),
                     float(f.det_score)))
    return dets


def match_tracks_faces(tracks, faces, iou_thresh=0.3):
    """트랙 렌더 박스와 검출 얼굴을 IoU 탐욕 매칭 → {track_id: face}."""
    tracked = [t for t in tracks if t.state == TrackState.Tracked]
    if not tracked or not faces:
        return {}
    tboxes = np.asarray([t.render_tlbr for t in tracked], dtype=np.float32)
    fboxes = np.asarray([f.bbox for f in faces], dtype=np.float32)
    ious = iou_batch(tboxes, fboxes)
    matched, used_f = {}, set()
    for ti in range(len(tracked)):
        fi = int(np.argmax(ious[ti]))
        if ious[ti, fi] >= iou_thresh and fi not in used_f:
            matched[tracked[ti].track_id] = faces[fi]
            used_f.add(fi)
    return matched


def process_video(args):
    providers = (["CUDAExecutionProvider", "CPUExecutionProvider"]
                 if args.providers == "cuda" else ["CPUExecutionProvider"])
    detector = detectors.make_detector(
        args.detector, det_size=args.det_size, providers=providers,
        model_path=detectors.YUNET_DEFAULT)
    names, db_embs = proto.load_db(args.db)

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {args.input}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # Phase 3 stretch: rtmp:// 출력이면 RTMP 실시간 송출
    out = make_writer(args.output, fps, w, h)

    tracker = BYTETracker(track_thresh=0.5, max_time_lost=30,
                          ema_alpha=0.6, vote_to_lock=3)
    prev_gray = None
    track_lm = {}  # Phase 3: {track_id: 최신 랜드마크(106,2)} — 마스크 익명화용
    track_masks = {}  # Phase 3 stretch: {track_id: SAM2 마스크(H,W) bool}
    sam2seg = None
    if args.mask_region == "sam2":
        import sam2_mask
        ckpt = sam2_mask.download_checkpoint("tiny", "models/")
        cfg = sam2_mask.CONFIGS["tiny"][0]
        sam2seg = sam2_mask.SAM2Segmenter(ckpt, cfg)
        print("SAM 2 마스크 모드 (tiny, CPU)")
    n_frames, t_all = 0, time.time()
    recog_calls, detect_calls = 0, 0
    ms = {"detect": 0.0, "track": 0.0, "recog": 0.0}
    n_scene_cuts = 0

    while True:
        ret, img = cap.read()
        if not ret:
            break
        if args.max_frames and n_frames >= args.max_frames:
            break

        # ---- 장면 전환 감지: 이전 장면의 Lost 트랙이 새 장면을 오염시키지 않도록 리셋
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if prev_gray is not None:
            diff = cv2.absdiff(gray, prev_gray).mean()
            if diff > args.scene_thresh:
                tracker = BYTETracker(track_thresh=0.5, max_time_lost=30,
                                      ema_alpha=0.6, vote_to_lock=3)
                track_lm.clear()  # 이전 장면의 랜드마크도 폐기
                track_masks.clear()  # SAM2 마스크도 폐기
                n_scene_cuts += 1
        prev_gray = gray

        if n_frames % args.skip == 0:
            # ---- 검출 프레임 ----
            t0 = time.time()
            faces = detector.detect(img)
            detector.embed(img, faces)  # YuNet: ArcFace rec 모델로 임베딩 추출
            ms["detect"] += (time.time() - t0) * 1000
            detect_calls += 1

            t0 = time.time()
            tracks = tracker.update(to_dets(faces))
            ms["track"] += (time.time() - t0) * 1000

            # 신규 트랙만 품질 게이팅 인식 → 투표
            id2face = match_tracks_faces(tracks, faces)
            t0 = time.time()
            for t in tracks:
                if t.state != TrackState.Tracked or t.identity_locked:
                    continue
                f = id2face.get(t.track_id)
                if f is None:
                    continue
                # Phase 3: 트랙별 최신 랜드마크 보관 (마스크 기반 익명화용)
                lm = getattr(f, "landmark_2d_106", None)
                if lm is not None:
                    track_lm[t.track_id] = np.asarray(lm, dtype=np.float32)
                name, _ = proto.recognize_faces(
                    [f], names, db_embs, args.threshold,
                    min_det_score=args.min_det_score,
                    min_face_size=args.min_face_size)[0]
                t.vote(name)
                recog_calls += 1
            ms["recog"] += (time.time() - t0) * 1000

            # Phase 3 stretch: SAM2 정밀 마스크 (검출 프레임의 unknown 얼굴)
            if sam2seg is not None:
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                for t in tracks:
                    if t.state != TrackState.Tracked:
                        continue
                    if (t.identity or "unknown") != "unknown":
                        continue
                    f = id2face.get(t.track_id)
                    if f is None:
                        continue
                    try:
                        track_masks[t.track_id] = sam2seg.mask_for_box(
                            rgb, tuple(f.bbox))
                    except Exception:
                        pass
        else:
            # ---- 스킵 프레임: Kalman 예측 보간 ----
            t0 = time.time()
            tracks = tracker.predict_only()
            ms["track"] += (time.time() - t0) * 1000

        # ---- Phase 3: 트랙 기반 마스크 익명화 (Tracked + Lost, EMA 스무딩 박스) ----
        regions, draw_faces, results = [], [], []
        for t in tracks:
            if t.state not in (TrackState.Tracked, TrackState.Lost):
                continue
            ident = t.identity or "unknown"  # 미확정 → fail-closed
            bb = tuple(t.render_tlbr)
            if ident == "unknown":
                regions.append({"bbox": bb,
                                "lm": track_lm.get(t.track_id),
                                "mask": track_masks.get(t.track_id)})
            draw_faces.append(type("F", (), {"bbox": np.array(bb)})())
            results.append((ident, 0.0))

        out_img = anonymize.anonymize(img, regions, region=args.mask_region,
                                      mode=args.mask_mode)
        out.write(proto.annotate(out_img, draw_faces, results))
        n_frames += 1

    cap.release()
    out.release()
    dt = time.time() - t_all
    print(f"{n_frames} frames, {dt:.1f}s ({n_frames/max(dt, 1e-6):.1f} FPS)")
    print(f"검출 {detect_calls}회 "
          f"({ms['detect']/max(detect_calls,1):.0f} ms/회), "
          f"트랙 {ms['track']/n_frames:.1f} ms/frame, "
          f"인식 {recog_calls}회 "
          f"({ms['recog']/max(recog_calls,1):.1f} ms/회)")
    print(f"저장: {args.output}")
    if n_scene_cuts:
        print(f"장면 전환 {n_scene_cuts}회 감지 → 트래커 리셋")


def main():
    p = argparse.ArgumentParser(description="MOFY Phase 2 통합 파이프라인")
    p.add_argument("--db", required=True)
    p.add_argument("--input", required=True)
    p.add_argument("--output", default="out_v2.mp4",
                   help="출력 파일 또는 rtmp:// URL (RTMP 실시간 송출)")
    p.add_argument("--threshold", type=float, default=0.4,
                   help="코사인 유사도 임계값 (calibrate 권장값)")
    p.add_argument("--skip", type=int, default=3,
                   help="검출 주기 (프레임). 사이는 Kalman 보간")
    p.add_argument("--det-size", type=int, default=640,
                   help="SCRFD 검출 입력 크기 (클수록 작은 얼굴↑·느림)")
    p.add_argument("--providers", default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--detector", default="scrfd", choices=["scrfd", "yunet"],
                   help="검출기 (Phase 3: yunet=모바일 경량 분기, CPU에서 ~19 FPS)")
    p.add_argument("--min-det-score", type=float, default=0.5)
    p.add_argument("--min-face-size", type=int, default=20)
    p.add_argument("--scene-thresh", type=float, default=40.0,
                   help="프레임 평균 밝기 차이 임계값 (초과 시 장면 전환으로 트래커 리셋)")
    p.add_argument("--max-frames", type=int, default=0,
                   help="처리할 최대 프레임 수 (0=전체, 테스트용)")
    p.add_argument("--mask-region", default="head",
                   choices=["bbox", "face", "head", "sam2"],
                   help="익명화 영역 (Phase 3: 랜드마크 마스크, "
                        "sam2=SAM 2 정밀 마스크)")
    p.add_argument("--mask-mode", default="pixelate",
                   choices=["pixelate", "blur", "solid"],
                   help="익명화 방식 (solid=검은색 채움이 가장 안전)")
    process_video(p.parse_args())


if __name__ == "__main__":
    main()
