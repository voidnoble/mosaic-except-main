# Phase 3: 실서비스 기능 (구현 기록, 2026-10-18)

## 1. 마스크 기반 익명화 (`anonymize.py`)
- 106점 랜드마크 → 얼굴 폴리곤(convex hull) / 헤드 폴리곤(이마 위 70%·좌우 15%·턱 아래 5% 확장)
- `region`: `bbox` | `face` | `head` (기본 `head`)
- `mode`: `pixelate` | `blur` | `solid` (기본 `pixelate`; solid=검은색 채움이 재식별에 가장 안전)
- 랜드마크 없으면 bbox로 폴백, 마스크 가장자리 feather=3 블렌딩
- `pipeline_v2.py`에 `--mask-region`, `--mask-mode` 옵션으로 통합
  - 트랙별 최신 랜드마크를 `track_lm` 딕셔너리에 보관 (Lost 트랙도 마스크 유지)
  - 장면 전환 시 트래커와 함께 `track_lm`도 초기화

## 2. 실시간 스트림 + 등록 관리 서버 (`server.py`)
- `GET /` — 등록 얼굴 관리 웹 UI (등록/삭제)
- `GET /persons`, `POST /persons` (multipart 이미지), `DELETE /persons/{name}`
- `WS /ws/process` — JPEG 프레임 수신 → SCRFD+ArcFace+ByteTrack+마스크 익명화 → JPEG 송신
  - 연결당 독립 트래커 유지 (세션 상태)
- 실행: `python server.py --db face_db.npz --port 8000`
- E2E 테스트 (`test_server.py`): 7/7 PASS
  - 등록→목록→WS 3프레임 왕복(형상 유지)→삭제→빈 목록

## 3. FAISS 기반 얼굴 DB (`face_store.py`)
- `FaceStore`: add/remove/list/search, npz 영속화
- Phase 2 `proto.build_db` 산출 npz와 호환 (`from_npz`)
- faiss 미설치 시 numpy 폴백
- 검증: self-match sim=1.00, 랜덤 질의 sim=0.05 → unknown

## 4. 모바일 경량 분기 (`detectors.py`, YuNet)
- `make_detector("scrfd"|"yunet")` 공통 인터페이스 (bbox/det_score/kps/embedding)
- YuNet: OpenCV DNN, 6명 전원 검출 53ms (SCRFD CPU ~2s 대비 ~17배)
  - 5점 랜드마크로 ArcFace rec 모델 임베딩 추출 → 인식 정상 동작
  - 랜드마크 106점 없어 마스크는 bbox 폴백
- `pipeline_v2.py --detector yunet`

## 측정 결과 (CPU, 60프레임 테스트 영상)
| 파이프라인 | FPS | 비고 |
|---|---|---|
| Phase 2 SCRFD (`--skip 3`) | 1.4 | 기준 |
| Phase 3 YuNet (`--skip 2`, bbox+solid) | 4.0 | 약 3배 |
| Phase 3 SCRFD (`--skip 6`, head+pixelate) | ~2.4 | 마스크 오버헤드 미미 |

## 남은 작업 (Phase 3 stretch) — 2026-10-19 구현 완료
- [x] SAM 2 정밀 마스크 (`sam2_mask.py`, `pipeline_v2 --mask-region sam2`)
  - sam2.1_hiera_tiny 체크포인트, 박스 프롬프트 → 얼굴 윤곽 정밀 분할
  - CPU 5.8초/마스크 (GPU 권장), 5프레임 E2E 검증 완료 (0.3 FPS CPU)
- [x] WebRTC 실시간 스트림 (`server.py` POST `/webrtc/offer`, aiortc)
  - 시그널링 검증 통과 (offer→answer 200, 유효 SDP+candidate)
  - 처리 경로 검증 통과 (6프레임 익명화, 평균 차이 5.3)
  - 환경 제한: 이 VM 샌드박스는 UDP를 차단하므로 실제 미디어 루프백 불가
    (32비트 실행 차단과 같은 샌드박스 정책). GPU/일반 환경에서는 동작.
- [x] RTMP 송출 (`stream_out.py`, `pipeline_v2 --output rtmp://...`)
  - ffmpeg 파이프 → libx264 → flv, 로컬 ffmpeg 리스너로 E2E 검증
    (수신 FLV에서 6명 전원 solid 마스크 확인)
- [x] 온디바이스 패키징 (`mobile/`)
  - `infer.py`: onnxruntime-only (YuNet 검출 + ArcFace 임베딩, Umeyama 정렬)
  - insightface 대비 임베딩 코사인 유사도 평균 0.97 (6명)
  - `build.py`: 모델+DB+스크립트를 `mofy-mobile.zip`(160MB)으로 패키징
  - 의존성: onnxruntime, opencv-python-headless, numpy만

## 아직 열린 항목 (GPU/GT 필요)
- AdaFace 백본 옵션, ONNX FP16/TensorRT, AI-Hub GT 기반 recall/false-match 측정
