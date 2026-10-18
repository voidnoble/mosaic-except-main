"""SCRFD 작은 얼굴 강건성 테스트 (Phase 2: 검출기 교체 근거).

테스트 이미지를 단계적으로 축소하며 검출 수/신뢰도를 측정한다.
원본 대비 축소본에서 얼굴이 계속 검출되는지 = 작은 얼굴 성능의 대리 지표.

사용법: python eval_small_faces.py --img_dir <이미지 폴더> [--det-size 640]
"""
import argparse
import glob
import os
import sys
import time

import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import prototype_scrfd_arcface as proto


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--img_dir", required=True)
    p.add_argument("--det-size", type=int, default=640)
    p.add_argument("--widths", default="0,640,320,160,80",
                   help="0=원본, 이후 목표 너비(px)")
    ns = p.parse_args()

    app = proto.get_app(det_size=(ns.det_size, ns.det_size))
    widths = [int(x) for x in ns.widths.split(",")]
    paths = sorted(glob.glob(os.path.join(ns.img_dir, "*")))
    paths = [x for x in paths
             if os.path.splitext(x)[1].lower() in proto.IMG_EXTS]
    # 하위 폴더(인물별)도 수집
    for d in sorted(glob.glob(os.path.join(ns.img_dir, "*"))):
        if os.path.isdir(d):
            paths += sorted(
                x for x in glob.glob(os.path.join(d, "*"))
                if os.path.splitext(x)[1].lower() in proto.IMG_EXTS)
    if not paths:
        print("이미지 없음:", ns.img_dir)
        return

    print(f"{'image':<28} {'orig_w':>7} " +
          " ".join(f"w{w:>4}" for w in widths if w) +
          "   (검출 수)")
    for path in paths:
        img0 = cv2.imread(path)
        if img0 is None:
            continue
        h0, w0 = img0.shape[:2]
        row = [f"{os.path.basename(path):<28} {w0:>7}"]
        for tw in widths:
            if tw == 0:
                img, tag = img0, "orig"
            else:
                s = tw / w0
                img = cv2.resize(img0, (tw, max(int(h0 * s), 1)))
                tag = tw
            t0 = time.time()
            faces = app.get(img)
            dt = (time.time() - t0) * 1000
            n = len(faces)
            min_s = min((f.det_score for f in faces), default=0.0)
            row.append(f"{n:>2}({min_s:.2f},{dt:4.0f}ms)")
        print(" ".join(row))
    print("\n형식: 검출수(최소신뢰도,추론ms). "
          "축소해도 검출 수가 유지되면 작은 얼굴에 강건.")


if __name__ == "__main__":
    main()
