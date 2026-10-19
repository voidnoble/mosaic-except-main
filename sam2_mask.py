"""Phase 3 stretch: SAM 2 기반 정밀 마스크 (sam2_mask.py)

검출 박스를 프롬프트로 SAM 2 이미지 예측을 수행해 정밀 얼굴 마스크를 얻는다.
랜드마크 폴리곤보다 경계가 정확하고 머리카락·턱선까지 포함한다.

사용법 (오프라인 전파는 향후 확장):
  seg = SAM2Segmenter(checkpoint, config)   # 1회 초기화
  mask = seg.mask_for_box(img_rgb, (x1, y1, x2, y2))  # (H, W) bool

pipeline_v2 --mask-region sam2 로 연동 (검출 프레임의 unknown 얼굴에 적용).
"""
import numpy as np

CONFIGS = {
    "tiny": ("configs/sam2.1/sam2.1_hiera_t.yaml",
             "sam2.1_hiera_tiny.pt"),
    "small": ("configs/sam2.1/sam2.1_hiera_s.yaml",
              "sam2.1_hiera_small.pt"),
}


def download_checkpoint(size="tiny", dest="models/"):
    """SAM 2.1 체크포인트 다운로드."""
    import os
    import urllib.request
    fname = CONFIGS[size][1]
    url = ("https://dl.fbaipublicfiles.com/segment_anything_2/092824/"
           + fname)
    path = os.path.join(dest, fname)
    if not os.path.exists(path):
        print(f"다운로드 중: {url}")
        urllib.request.urlretrieve(url, path)
    return path


class SAM2Segmenter:
    def __init__(self, checkpoint, config, device="cpu"):
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        model = build_sam2(config, checkpoint, device=device)
        self.predictor = SAM2ImagePredictor(model)
        self._img_id = None

    def mask_for_box(self, img_rgb, box):
        """단일 박스 → 최고 점수 마스크 (H, W) bool."""
        import torch
        if self._img_id != id(img_rgb):
            self.predictor.set_image(img_rgb)
            self._img_id = id(img_rgb)
        with torch.inference_mode():
            masks, scores, _ = self.predictor.predict(
                box=np.asarray(box, dtype=np.float32))
        return masks[int(np.argmax(scores))].astype(bool)
