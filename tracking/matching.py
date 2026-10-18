"""IoU 계산 + 선형 할당 (scipy 있으면 Hungarian, 없으면 greedy)."""
import numpy as np

try:
    from scipy.optimize import linear_sum_assignment

    _HAS_SCIPY = True
except ImportError:  # pragma: no cover
    _HAS_SCIPY = False


def iou_batch(boxes1, boxes2):
    """boxes: (N,4) tlbr. → (N,M) IoU 행렬."""
    if len(boxes1) == 0 or len(boxes2) == 0:
        return np.zeros((len(boxes1), len(boxes2)), dtype=np.float32)
    b1 = np.expand_dims(boxes1, 1)  # (N,1,4)
    b2 = np.expand_dims(boxes2, 0)  # (1,M,4)
    xx1 = np.maximum(b1[..., 0], b2[..., 0])
    yy1 = np.maximum(b1[..., 1], b2[..., 1])
    xx2 = np.minimum(b1[..., 2], b2[..., 2])
    yy2 = np.minimum(b1[..., 3], b2[..., 3])
    w = np.maximum(0.0, xx2 - xx1)
    h = np.maximum(0.0, yy2 - yy1)
    inter = w * h
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    union = area1[:, None] + area2[None, :] - inter
    return inter / np.maximum(union, 1e-6)


def linear_assignment(cost_matrix, thresh):
    """cost 행렬에서 임계값 이하 매칭만 반환.

    Returns: matches (K,2), unmatched_rows, unmatched_cols
    """
    if cost_matrix.size == 0:
        return (
            np.empty((0, 2), dtype=int),
            tuple(range(cost_matrix.shape[0])),
            tuple(range(cost_matrix.shape[1])),
        )
    if _HAS_SCIPY:
        rows, cols = linear_sum_assignment(cost_matrix)
        matches = []
        for r, c in zip(rows, cols):
            if cost_matrix[r, c] <= thresh:
                matches.append([r, c])
        matches = np.array(matches, dtype=int) if matches else np.empty((0, 2), dtype=int)
    else:
        # greedy fallback: 비용 오름차순 탐욕 매칭
        matches = []
        used_r, used_c = set(), set()
        flat = np.argsort(cost_matrix, axis=None)
        for idx in flat:
            r, c = np.unravel_index(idx, cost_matrix.shape)
            if cost_matrix[r, c] > thresh or r in used_r or c in used_c:
                continue
            matches.append([r, c])
            used_r.add(r)
            used_c.add(c)
        matches = np.array(matches, dtype=int) if matches else np.empty((0, 2), dtype=int)

    matched_r = set(matches[:, 0]) if len(matches) else set()
    matched_c = set(matches[:, 1]) if len(matches) else set()
    unmatched_rows = tuple(i for i in range(cost_matrix.shape[0]) if i not in matched_r)
    unmatched_cols = tuple(i for i in range(cost_matrix.shape[1]) if i not in matched_c)
    return matches, unmatched_rows, unmatched_cols


def iou_distance(tracks, detections):
    """1 - IoU 비용 행렬. tracks/detections: tlbr 배열 리스트."""
    if len(tracks) == 0 or len(detections) == 0:
        return np.zeros((len(tracks), len(detections)), dtype=np.float32)
    t = np.asarray([tr.tlbr for tr in tracks], dtype=np.float32)
    d = np.asarray([dt.tlbr for dt in detections], dtype=np.float32)
    return 1.0 - iou_batch(t, d)
