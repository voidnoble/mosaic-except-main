"""ByteTrack (2단계 연관 매칭) — 얼굴 트래킹용 경량 구현.

1단계: 높은 신뢰도 검출과 기존 트랙 매칭
2단계: 낮은 신뢰도 검출로 남은 트랙 복원 (가려짐/흔들림에 강함)
"""
import numpy as np

from .matching import iou_distance, linear_assignment
from .track import FaceTrack, TrackState


class BYTETracker:
    def __init__(
        self,
        track_thresh=0.5,      # 1단계 매칭에 쓸 최소 검출 신뢰도
        low_thresh=0.1,        # 2단계 매칭에 쓸 최소 신뢰도
        match_thresh=0.7,      # IoU 거리 매칭 임계값 (1 - IoU)
        max_time_lost=30,      # Lost 유지 프레임 수 (30fps 기준 1초)
        ema_alpha=0.6,
        vote_to_lock=3,
    ):
        self.track_thresh = track_thresh
        self.low_thresh = low_thresh
        self.match_thresh = match_thresh
        self.max_time_lost = max_time_lost
        self.ema_alpha = ema_alpha
        self.vote_to_lock = vote_to_lock
        self.tracked_tracks = []   # 활성 트랙
        self.lost_tracks = []      # 일시 소실 트랙
        self.removed_tracks = []
        self.frame_id = 0
        FaceTrack.reset_id()

    def _make_track(self, tlwh, score):
        return FaceTrack(
            tlwh, score, ema_alpha=self.ema_alpha, vote_to_lock=self.vote_to_lock
        )

    def predict_only(self):
        """검출 스킵 프레임용: 연관 없이 Kalman 예측만 수행.

        검출을 건너뛴 프레임에서도 트랙 박스를 보간해 모자이크를 유지한다.
        Lost 트랙의 만료(max_time_lost)는 벽시간 기준으로 계속 적용된다.
        """
        self.frame_id += 1
        for t in self.tracked_tracks + self.lost_tracks:
            t.predict()
        # 오래 Lost된 트랙 제거
        for t in self.lost_tracks:
            if self.frame_id - t.end_frame > self.max_time_lost:
                t.mark_removed()
        self.lost_tracks = [t for t in self.lost_tracks
                            if t.state == TrackState.Lost]
        return self.tracked_tracks + self.lost_tracks

    def update(self, detections):
        """detections: [(tlwh(4,), score), ...] → 활성+Lost 트랙 리스트 반환."""
        self.frame_id += 1

        # --- 검출 분리 ---
        dets = [self._make_track(tlwh, s) for tlwh, s in detections]
        dets_high = [d for d in dets if d.score >= self.track_thresh]
        dets_low = [d for d in dets if self.low_thresh <= d.score < self.track_thresh]

        # --- 기존 트랙 예측 ---
        for t in self.tracked_tracks + self.lost_tracks:
            t.predict()

        # --- 1단계: Tracked + Lost vs 높은 신뢰도 검출 ---
        pool = [t for t in self.tracked_tracks + self.lost_tracks
                if t.state in (TrackState.Tracked, TrackState.Lost)]
        matches, u_track, u_det = linear_assignment(
            iou_distance(pool, dets_high), self.match_thresh
        )
        activated, refind = [], []
        for it, id_ in matches:
            trk, det = pool[it], dets_high[id_]
            if trk.state == TrackState.Tracked:
                trk.update(det, self.frame_id)
                activated.append(trk)
            else:  # Lost → 복귀
                trk.update(det, self.frame_id)
                refind.append(trk)
        remaining_tracks = [pool[i] for i in u_track
                            if pool[i].state == TrackState.Tracked]
        remaining_high = [dets_high[i] for i in u_det]

        # --- 2단계: 남은 Tracked vs 낮은 신뢰도 검출 (복원) ---
        matches2, u_track2, _ = linear_assignment(
            iou_distance(remaining_tracks, dets_low), 0.5
        )
        for it, id_ in matches2:
            trk, det = remaining_tracks[it], dets_low[id_]
            trk.update(det, self.frame_id)
            activated.append(trk)
        for i in u_track2:
            trk = remaining_tracks[i]
            if trk.state != TrackState.Lost:
                trk.mark_lost()
                if all(t is not trk for t in self.lost_tracks):
                    self.lost_tracks.append(trk)

        # --- 신규 트랙 ---
        for det in remaining_high:
            det.activate(self.frame_id)
            activated.append(det)

        # --- 오래 Lost된 트랙 제거 ---
        for trk in self.lost_tracks:
            if self.frame_id - trk.end_frame > self.max_time_lost:
                trk.mark_removed()
        # 중복 트랙 제거 (IoU>0.85, 짧은 쪽 제거)
        activated = self._remove_duplicate(activated + refind)

        self.tracked_tracks = [t for t in activated if t.state == TrackState.Tracked]
        # Lost 리스트: 상태가 Lost인 것만 유지, 객체 중복 제거
        seen = set()
        lost = []
        for t in self.lost_tracks:
            if t.state == TrackState.Lost and id(t) not in seen:
                seen.add(id(t))
                lost.append(t)
        self.lost_tracks = lost
        return self.tracked_tracks + self.lost_tracks

    @staticmethod
    def _remove_duplicate(tracks):
        if len(tracks) < 2:
            return tracks
        from .matching import iou_batch
        boxes = np.asarray([t.tlbr for t in tracks], dtype=np.float32)
        ious = iou_batch(boxes, boxes)
        dup = set()
        for i in range(len(tracks)):
            for j in range(i + 1, len(tracks)):
                if ious[i, j] > 0.85:
                    # 수명이 짧은 쪽을 중복으로
                    if tracks[i].frame_id - tracks[i].start_frame > \
                       tracks[j].frame_id - tracks[j].start_frame:
                        dup.add(j)
                    else:
                        dup.add(i)
        return [t for i, t in enumerate(tracks) if i not in dup]
