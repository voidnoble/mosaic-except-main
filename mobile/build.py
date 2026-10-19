"""온디바이스 배포 패키지 빌드 (mobile/build.py).

~/.insightface/models/buffalo_l/w600k_r50.onnx (ArcFace 인식) +
models/face_detection_yunet_2023mar.onnx (YuNet 검출) +
mobile/infer.py + requirements + DB를 하나의 zip으로 묶는다.

사용법:
  python mobile/build.py --db face_db.npz --out mofy-mobile.zip
"""
import argparse
import os
import shutil
import sys
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NEED = {
    "face_detection_yunet_2023mar.onnx": os.path.join(
        ROOT, "models", "face_detection_yunet_2023mar.onnx"),
    "w600k_r50.onnx": os.path.expanduser(
        "~/.insightface/models/buffalo_l/w600k_r50.onnx"),
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--db", required=True, help="얼굴 DB npz")
    p.add_argument("--out", default="mofy-mobile.zip")
    a = p.parse_args()

    missing = [k for k, v in NEED.items() if not os.path.exists(v)]
    if missing:
        sys.exit(f"모델 파일 없음: {missing}")
    if not os.path.exists(a.db):
        sys.exit(f"DB 없음: {a.db}")

    with zipfile.ZipFile(a.out, "w", zipfile.ZIP_DEFLATED) as z:
        for name, path in NEED.items():
            z.write(path, f"mofy-mobile/models/{name}")
            print(f"  models/{name} "
                  f"({os.path.getsize(path)/1e6:.1f} MB)")
        z.write(os.path.join(ROOT, "mobile", "infer.py"),
                "mofy-mobile/infer.py")
        z.write(os.path.join(ROOT, "mobile", "requirements-mobile.txt"),
                "mofy-mobile/requirements-mobile.txt")
        z.write(os.path.join(ROOT, "mobile", "README.md"),
                "mofy-mobile/README.md")
        z.write(a.db, "mofy-mobile/face_db.npz")
    print(f"패키지 생성: {a.out} "
          f"({os.path.getsize(a.out)/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
