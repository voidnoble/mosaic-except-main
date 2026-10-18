"""Phase 1 기준선 측정 스크립트 (모델 없이 트래커+타이밍만 측정 가능).

1) python eval_baseline.py --synthetic   : 합성 궤적으로 트래커 자가 검증
2) python eval_baseline.py --video in.mp4 : pipeline.process_video를 래핑해
   프레임별 타이밍/카운트를 CSV로 기록 (모델 로딩은 pipeline.main과 동일)

CSV 컬럼: frame, n_det, n_tracked, n_lost, n_unknown,
          detect_ms, track_ms, recog_ms
요약: 평균 FPS, 단계별 ms/frame, 총 인식 호출 수, Lost-유지 프레임 수
      (Lost-유지 = 검출 실패에도 모자이크가 유지된 프레임 → 누락 방지 지표)
"""
import argparse
import csv
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

CSV_HEADER = ["frame", "n_det", "n_tracked", "n_lost", "n_unknown",
              "detect_ms", "track_ms", "recog_ms"]


def run_synthetic():
    from tracking import BYTETracker
    from tracking.track import TrackState

    tr = BYTETracker(track_thresh=0.5, max_time_lost=5)
    id_switches, frames = 0, 10
    prev = {}
    for f in range(frames):
        dets = [((100 + f * 10, 100, 60, 60), 0.9)]
        if f not in (3, 4):  # 2프레임 가려짐
            dets.append(((300 - f * 5, 200, 60, 60), 0.9))
        tracks = tr.update(dets)
        cur = {t.track_id: tuple(np.round(t.predicted_tlwh[:2]).astype(int))
               for t in tracks if t.state == TrackState.Tracked}
        # 같은 위치 근처에 다른 ID가 나타나면 switch로 계수 (허용 오차 40px)
        for tid, pos in cur.items():
            for ptid, ppos in prev.items():
                if ptid != tid and abs(pos[0] - ppos[0]) < 40 \
                        and abs(pos[1] - ppos[1]) < 40:
                    id_switches += 1
        prev = cur
    tracked_ids = sorted({t for t in []})
    print(f"synthetic: {frames} frames, id_switches={id_switches} "
          f"(목표 0)")


def run_video(video_path, csv_path, detector="retinaface"):
    import cv2  # 비디오 모드에서만 필요
    # pipeline의 무거운 import는 여기서 (torch 필요)
    import pipeline as P
    from main import init as baseline_init
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "baselinewithfacenet"))
    from tracking import BYTETracker
    from tracking.track import TrackState

    args = {"DETECTOR": detector, "RECOG_THRESHOLD": 0.8,
            "DEBUG_MODE": False, "PROCESS_TARGET": "Video",
            "VIDEO_PATH": video_path, "SAVE_PATH": "/dev/null"}
    model_args = baseline_init({"DETECTOR": detector, "DEBUG_MODE": False})
    tracker = BYTETracker(track_thresh=0.5, max_time_lost=30)

    cap = cv2.VideoCapture(video_path)
    rows, t_all = [], time.time()
    lost_kept_frames = 0
    frame_idx = 0
    from PIL import Image
    while True:
        ret, bgr = cap.read()
        if not ret:
            break
        img_det = (Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
                   if detector == "mtcnn" else bgr)
        t0 = time.time()
        dets = P.detect_with_scores(img_det, args, model_args)
        d_ms = (time.time() - t0) * 1000
        t0 = time.time()
        tracks = tracker.update(dets)
        t_ms = (time.time() - t0) * 1000
        t0 = time.time()
        n_unknown, recog_ms = 0, 0.0
        for trk in tracks:
            if trk.state == TrackState.Lost or trk.identity_locked:
                continue
            name = P.recognize_one(img_det, trk.tlbr, args, model_args)
            trk.vote(name)
            if (trk.identity or "unknown") == "unknown":
                n_unknown += 1
        recog_ms = (time.time() - t0) * 1000
        n_tr = sum(1 for t in tracks if t.state == TrackState.Tracked)
        n_lo = sum(1 for t in tracks if t.state == TrackState.Lost)
        if n_lo > 0:
            lost_kept_frames += 1
        rows.append([frame_idx, len(dets), n_tr, n_lo, n_unknown,
                     round(d_ms, 2), round(t_ms, 2), round(recog_ms, 2)])
        frame_idx += 1
    cap.release()

    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(CSV_HEADER)
        w.writerows(rows)
    dt = time.time() - t_all
    arr = np.array([r[5:] for r in rows])
    print(f"frames={len(rows)} total={dt:.1f}s "
          f"({len(rows)/max(dt,1e-6):.1f} FPS)")
    print(f"detect {arr[:,0].mean():.1f} ms/frame | "
          f"track {arr[:,1].mean():.1f} ms/frame | "
          f"recog {arr[:,2].mean():.1f} ms/frame")
    print(f"Lost-유지 프레임: {lost_kept_frames} "
          f"(검출 실패에도 모자이크 유지 → 누락 방지)")
    print(f"CSV 저장: {csv_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--synthetic", action="store_true")
    p.add_argument("--video", default=None)
    p.add_argument("--csv", default="baseline_metrics.csv")
    p.add_argument("--detector", default="retinaface",
                   choices=["mtcnn", "retinaface"])
    ns = p.parse_args()
    if ns.synthetic or not ns.video:
        run_synthetic()
    else:
        run_video(ns.video, ns.csv, ns.detector)


if __name__ == "__main__":
    main()
