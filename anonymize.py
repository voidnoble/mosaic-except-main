"""Phase 3: 마스크 기반 익명화 (anonymize.py)

bbox 박스 대신 106점 랜드마크 폴리곤/헤드 마스크로 익명화 영역을 지정한다.
- region="bbox": 기존 방식 (박스)
- region="face": 랜드마크 convex hull 폴리곤 (얼굴 윤곽에 밀착)
- region="head": 이마 위·귀 옆·턱 아래까지 확장한 헤드 폴리곤
  (헤어스타일·귀 노출로 인한 재식별 방지)
- mode="pixelate" / "blur" / "solid" (검은색 채움 — 논문 근거상 가장 안전)

사용법:
  regions = [{"bbox": (x1,y1,x2,y2), "lm": lm106_or_None}, ...]
  out = anonymize(img, regions, region="head", mode="pixelate")
"""
import cv2
import numpy as np


def face_polygon(lm):
    """106점 랜드마크 → 얼굴 폴리곤 (convex hull)."""
    hull = cv2.convexHull(lm.astype(np.float32).reshape(-1, 1, 2))
    return hull.reshape(-1, 2)


def head_polygon(lm):
    """얼굴 폴리곤을 머리 전체로 확장.

    - 위: 이마 상단보다 (턱~눈썹 거리)의 70% 위로 확장 (머리카락 포함)
    - 좌우: 15%씩 확장 (귀 포함)
    - 아래: 턱에서 5% 아래로
    """
    lm = np.asarray(lm, dtype=np.float32)
    x1, y1 = lm.min(axis=0)
    x2, y2 = lm.max(axis=0)
    w, h = x2 - x1, y2 - y1
    # 이마 상단(y1) 기준으로 위쪽 확장
    top = y1 - h * 0.70
    left = x1 - w * 0.15
    right = x2 + w * 0.15
    bottom = y2 + h * 0.05
    poly = face_polygon(lm)
    # 상단 정점들을 위로 밀어 올림 (머리 윗부분 확장)
    cy = (y1 + y2) / 2
    poly = poly.copy()
    upper = poly[:, 1] < cy
    poly[upper, 1] = np.minimum(poly[upper, 1], top)
    poly[:, 0] = np.clip(poly[:, 0], left, right)
    poly[:, 1] = np.clip(poly[:, 1], top, bottom)
    return poly


def _bbox_poly(bbox):
    x1, y1, x2, y2 = bbox
    return np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
                    dtype=np.float32)


def region_mask(shape, polys, feather=0):
    """폴리곤 리스트 → 0/1 마스크 (feather 픽셀만큼 가장자리 블러)."""
    mask = np.zeros(shape[:2], dtype=np.uint8)
    for poly in polys:
        cv2.fillPoly(mask, [poly.astype(np.int32)], 255)
    if feather > 0:
        mask = cv2.GaussianBlur(mask, (0, 0), feather)
    return (mask.astype(np.float32) / 255.0)[..., None]


def pixelate_roi(roi, n=12):
    h, w = roi.shape[:2]
    if h < n or w < n:
        return np.zeros_like(roi)
    small = cv2.resize(roi, (max(w // n, 1), max(h // n, 1)),
                       interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)


def apply_effect(img, mask, mode, n=12, blur_ksize=51):
    """마스크 영역에만 효과를 적용 (가장자리는 마스크 가중치로 블렌딩)."""
    if mode == "solid":
        eff = np.zeros_like(img)
    elif mode == "blur":
        k = blur_ksize if blur_ksize % 2 else blur_ksize + 1
        eff = cv2.GaussianBlur(img, (k, k), 0)
    else:  # pixelate — bbox 단위로 블록화한 뒤 마스크로 자름
        eff = img.copy()
    out = img.astype(np.float32)
    if mode == "pixelate":
        # 마스크 경계 박스 단위로 픽셀레이션 후 마스크 블렌딩
        ys, xs = np.nonzero(mask[..., 0] > 0.01)
        if len(xs):
            x1, x2, y1, y2 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
            roi = img[y1:y2, x1:x2]
            eff[y1:y2, x1:x2] = pixelate_roi(roi, n)
    blended = eff.astype(np.float32) * mask + out * (1.0 - mask)
    return blended.astype(np.uint8)


def anonymize(img, regions, region="head", mode="pixelate", n=12,
              feather=3):
    """regions: [{"bbox": (x1,y1,x2,y2), "lm": (106,2) or None}, ...]
    랜드마크가 없으면 bbox로 폴백."""
    polys = []
    for r in regions:
        lm = r.get("lm")
        if region == "bbox" or lm is None:
            polys.append(_bbox_poly(r["bbox"]))
        elif region == "face":
            polys.append(face_polygon(lm))
        else:  # head
            polys.append(head_polygon(lm))
    if not polys:
        return img
    mask = region_mask(img.shape, polys, feather=feather)
    return apply_effect(img, mask, mode, n=n)
