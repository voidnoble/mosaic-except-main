# MOFY Mobile — 온디바이스 패키지

insightface 없이 onnxruntime만으로 동작하는 경량 얼굴 모자이크 추론 패키지.

## 구성
- `models/face_detection_yunet_2023mar.onnx` — YuNet 얼굴 검출 (0.3 MB)
- `models/w600k_r50.onnx` — ArcFace 인식 (WebFace600K, 170 MB)
- `infer.py` — 검출→정렬(Umeyama)→임베딩→코사인 매칭→모자이크
- `face_db.npz` — 등록 얼굴 DB

## 설치·실행
```bash
pip install -r requirements-mobile.txt
unzip mofy-mobile.zip && cd mofy-mobile
python infer.py --models ./models --db face_db.npz --input in.jpg --output out.jpg
python infer.py --models ./models --db face_db.npz --input in.mp4 --output out.mp4
```

## 주의
- w600k_r50 가중치는 비상업 연구용 라이선스 (InsightFace). 상용 이용 시 라이선스 검토 필요.
- 임계값 기본 0.4 (코사인 유사도). `prototype_scrfd_arcface.py calibrate`로 조정 권장.
