# Phase 2 구현 기록 (모델 현대화)

## 변경 사항

| 항목 | 내용 |
|---|---|
| 통합 파이프라인 | `pipeline_v2.py` 신규 — SCRFD + ArcFace (InsightFace `buffalo_l`) + Phase 1 ByteTrack |
| 프레임 스킵 | `--skip K` — K프레임마다 검출, 사이는 Kalman 예측 보간 (`BYTETracker.predict_only()` 신규) |
| 품질 게이팅 | `det_score`/`얼굴 크기` 미달 → 판정 보류 → 모자이크 (fail-closed) |
| 임계값 | 코사인 유사도, `prototype_scrfd_arcface.py calibrate`로 재산정 |
| 장면 전환 처리 | 프레임 밝기 차이로 감지 → 트래커 리셋 (`--scene-thresh`) |
| 제공자 선택 | `--providers cpu/cuda` (GPU 시 `CUDAExecutionProvider`) |
| 작은 얼굴 테스트 | `eval_small_faces.py` — 축소 스윕으로 검출 강건성 측정 |

## 검증 중 발견한 버그와 수정

**증상**: 장면 전환 후 이전 장면의 Lost 트랙이 최대 30프레임 동안 모자이크를 유지하면서,
새 장면의 등록 인물 얼굴 위에 겹쳐 모자이크가 잘못 적용됨.

**원인**: `max_time_lost=30` 유지 로직이 장면 전환을 구분하지 못함.

**수정**: `pipeline_v2.py`에 장면 전환 감지(프레임 간 평균 밝기 차이 > 40) 추가,
감지 시 ByteTrack 인스턴스를 리셋. 수정 후 등록 인물 프레임이 정상 출력됨
(초록 박스, 모자이크 없음, 잔상 박스 없음).

## 측정 결과 (CPU, 30프레임 테스트 영상, --skip 3)

| 지표 | 값 |
|---|---|
| 처리 시간 | 20.0s (1.4 FPS) — Phase 1 프로토타입 0.5 FPS 대비 **약 3배 향상** |
| 검출 호출 | 10회 (2047 ms/회) |
| 트랙 | 0.8 ms/frame |
| 인식 호출 | 21회 (0.4 ms/회) |
| 장면 전환 | 1회 감지 → 트래커 리셋 정상 |

병목은 여전히 SCRFD CPU 추론 (~2s/회). 실시간 처리는 GPU 필수.

## 작은 얼굴 강건성 (SCRFD, det_size=640)

5/6 테스트 이미지에서 너비 80px까지 검출 수·신뢰도 유지 (0.72~0.92).
`ob5.jpeg`(200px 원본)는 전 스케일에서 미검출 — 난이도가 높은 이미지로 별도 확인 필요.

## 남은 작업 (Phase 2 stretch)

- AdaFace 백본 교체 옵션 (저화질 영상 대응) — 별도 가중치 필요
- ONNX FP16 / TensorRT 적용 (GPU 환경에서)
- AI-Hub 방송 데이터셋 기반 재현율·오식별률 측정 (GT 라벨 필요)
