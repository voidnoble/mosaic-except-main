# GPU 실측 런북 (draft — 2026-10-19)

이 VM에는 NVIDIA GPU가 없어서 진행하지 못한 항목을 CUDA 머신에서 실행하는 절차.
기준 코드: remote main `9f332d90`. CPU 실측값은 이 문서 하단 표 참조.

## 0. 환경 준비

```bash
nvidia-smi                      # CUDA 버전 확인 (예: 12.x)
python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt
# requirements.txt의 torch는 CUDA 빌드(CPU 호환). GPU 박스에서는 그대로 사용.
.venv/bin/pip install onnxruntime-gpu   # torch CUDA와 맞는 빌드인지 확인
.venv/bin/python -c "import onnxruntime as o; print(o.get_available_providers())"
# 기대: ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
```

## 1. CUDA 기준 FPS 재측정 (CPU 병목 검증)

```bash
# SCRFD + ArcFace (Phase 2 기준)
.venv/bin/python pipeline_v2.py --input <영상> --providers cuda --skip 3
# YuNet 경량 분기 (Phase 3 기준, CPU 4.0 FPS)
.venv/bin/python pipeline_v2.py --input <영상> --providers cuda --detector yunet --skip 3
# AdaFace 백본
.venv/bin/python pipeline_v2.py --input <영상> --providers cuda --rec-model adaface
```

기록할 것: 검출기별 FPS, `--skip` 값별 FPS, GPU 메모리 사용량(`nvidia-smi dmon`).

## 2. FP16 / TensorRT

주의: TensorRT는 박스의 CUDA·cuDNN 버전과 맞는 빌드 필요. 버전 불일치가 가장 흔한 실패 원인.

```bash
# (a) ORT CUDA EP + FP16: onnxruntime FP16 변환 후 CUDA 프로바이더로 실행
.venv/bin/python - <<'EOF'
from onnxruntime.tools.transformers.float16 import convert_float16
# 예: models/adaface_ir_101.onnx -> adaface_fp16.onnx
EOF

# (b) TensorRT 엔진 빌드 (검출 모델은 dynamic shape → optimization profile 필수)
trtexec --onnx=models/adaface_ir_101.onnx --saveEngine=adaface_fp16.trt \
  --fp16 --minShapes=input:1x3x112x112 --optShapes=input:1x3x112x112 \
  --maxShapes=input:8x3x112x112
# SCRFD/YuNet도 동일 패턴 (입력 shape는 각 모델에 맞게 조정)
```

측정 지표: FP32 CUDA 대비 latency/FPS, FP16 수치 오차(임베딩 cosine 유사도 ≥ 0.99 기준).
정직한 한계: TensorRT EP 경로 코드는 미작성 — 엔진 빌드 후
`rec_models.py`/`detectors.py`의 providers 리스트에 `TensorrtExecutionProvider`를
추가하는 작업이 별도로 필요.

## 3. SAM 2 GPU 실측

```bash
.venv/bin/python pipeline_v2.py --input <영상> --providers cuda --mask-region sam2
```

CPU 실측 5.8초/마스크(`sam2.1_hiera_tiny.pt`) 대비 GPU 시간을 기록.
메모리 부족 시 `sam2_mask.py`의 `size="tiny"` 유지, 여유 있으면 small로 교체
(`download_checkpoint("small")`, `sam2.1_hiera_small.pt`).

## 4. AI-Hub GT 재현율 측정

`eval_recall.py`는 현재 augmentation 기반 동등 지표(verification accuracy,
TAR@FAR=1%, 열화 강건성)를 측정. GT 연결 시 교체 지점:

- `verification_test(det, gallery_dir, ...)`: `gallery_dir/<person_id>/*.jpg` 구조를
  그대로 사용 — AI-Hub 데이터를 이 디렉토리 구조로 풀어 넣으면 코드 수정 없이 실행.
- 저화질 강건성(`degradation_test`): 갤러리=원본, 쿼리=`degrade()` 열화(x56/x28/blur/jpeg)
  rank-1 — 감시급 저화질 쿼리셋이 있으면 `degrade()` 대신 실제 저화질 이미지를
  쿼리로 투입하도록 해당 함수만 교체.

```bash
.venv/bin/python eval_recall.py --gallery <GT_디렉토리> --backbones arcface,adaface
```

AdaFace의 강점은 저화질에서 발현 — 고화질셋에서 차이 없었던 10-19 CPU 결과와
저화질셋 결과를 나란히 기록할 것.

## CPU 실측 기준값 (이 VM, 2026-10-18~19)

| 항목 | 값 |
|---|---|
| Phase 2 파이프라인 | 1.4 FPS (CPU) |
| YuNet 분기 | 4.0 FPS (CPU) |
| SAM 2 마스크 | 5.8초/마스크 (CPU, hiera_tiny) |
| AdaFace vs ArcFace | 차이 없음 (고화질 테스트셋, accuracy 100%) |
| WebRTC | 시그널링만 검증 (VM이 UDP 차단) |

---
*이 문서는 draft입니다. GPU 박스에서 검증된 뒤 확정판으로 갱신할 것.*
