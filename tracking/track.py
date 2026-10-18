"""트랙 상태 관리: Kalman 예측/갱신 + 신원 투표 + EMA 박스 스무딩."""
import numpy as np

from .kalman_filter import KalmanFilter


class TrackState:
    New = 0
    Tracked = 1
    Lost = 2
    Removed = 3


class STrack:
    """단일 트랙. 내부는 tlwh, Kalman 상태는 (x, y, a, h)."""

    shared_kalman = KalmanFilter()

    def __init__(self, tlwh, score):
        self._tlwh = np.asarray(tlwh, dtype=np.float32)
        self.score = float(score)
        self.track_id = None
        self.state = TrackState.New
        self.is_activated = False
        self.frame_id = 0
        self.start_frame = 0
        self.end_frame = 0
        self.time_since_update = 0

    # ---- 좌표 변환 ----
    @property
    def tlwh(self):
        """Tracked 상태면 Kalman mean(x,y,a,h)에서, 아니면 마지막 관측값에서."""
        if getattr(self, "mean", None) is not None:
            x, y, a, h = self.mean[:4]
            w = a * h
            return np.array([x - w / 2, y - h / 2, w, h], dtype=np.float32)
        return self._tlwh.copy()

    @property
    def tlbr(self):
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @staticmethod
    def tlwh_to_xyah(tlwh):
        ret = np.asarray(tlwh, dtype=np.float32).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= max(ret[3], 1e-6)
        return ret

    # ---- Kalman ----
    def activate(self, frame_id):
        self.track_id = self.next_id()
        self.mean, self.covariance = self.shared_kalman.initiate(
            self.tlwh_to_xyah(self._tlwh)
        )
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id
        self.end_frame = frame_id

    def predict(self):
        if self.state != TrackState.Tracked:
            # Lost 상태에서도 예측은 유지 (가려짐 구간 커버)
            pass
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[7] = 0  # 속도 0으로 고정
        self.mean, self.covariance = self.shared_kalman.predict(
            mean_state, self.covariance
        )

    def update(self, new_track, frame_id):
        self.frame_id = frame_id
        self.end_frame = frame_id
        self.time_since_update = 0
        self.score = new_track.score
        self._tlwh = np.asarray(new_track.tlwh, dtype=np.float32)
        self.mean, self.covariance = self.shared_kalman.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_track.tlwh)
        )
        self.state = TrackState.Tracked
        self.is_activated = True

    def mark_lost(self):
        self.state = TrackState.Lost

    def mark_removed(self):
        self.state = TrackState.Removed

    @property
    def predicted_tlwh(self):
        """Kalman 예측 위치의 tlwh (Lost 트랙의 모자이크 유지용)."""
        if self.mean is None:
            return self._tlwh.copy()
        x, y, a, h = self.mean[:4]
        w = a * h
        return np.array([x - w / 2, y - h / 2, w, h], dtype=np.float32)

    # ---- ID 발급 ----
    _count = 0

    @classmethod
    def next_id(cls):
        cls._count += 1
        return cls._count

    @classmethod
    def reset_id(cls):
        cls._count = 0


class FaceTrack(STrack):
    """얼굴 전용 트랙: 신원 투표 + EMA 스무딩 박스."""

    def __init__(self, tlwh, score, ema_alpha=0.6, vote_to_lock=3):
        super().__init__(tlwh, score)
        self.votes = {}            # {name: count}
        self.identity = None       # 확정 신원 (None이면 미확정)
        self.identity_locked = False
        self._vote_to_lock = vote_to_lock
        self._ema_alpha = ema_alpha
        self._smooth_tlwh = np.asarray(tlwh, dtype=np.float32)

    def vote(self, name):
        """인식 결과를 1표로 기록. 과반+최소 표수 도달 시 신원 확정."""
        if self.identity_locked:
            return self.identity
        self.votes[name] = self.votes.get(name, 0) + 1
        total = sum(self.votes.values())
        best, best_n = max(self.votes.items(), key=lambda kv: kv[1])
        if best_n >= self._vote_to_lock and best_n > total / 2:
            self.identity = best
            self.identity_locked = True
        return self.identity

    def smooth(self, tlwh):
        """EMA 스무딩 박스 갱신 → 반환 (모자이크 렌더링용)."""
        tlwh = np.asarray(tlwh, dtype=np.float32)
        self._smooth_tlwh = (
            self._ema_alpha * tlwh + (1 - self._ema_alpha) * self._smooth_tlwh
        )
        return self._smooth_tlwh.copy()

    @property
    def render_tlbr(self):
        """렌더링용 박스: 스무딩된 tlwh → tlbr."""
        ret = self._smooth_tlwh.copy()
        ret[2:] += ret[:2]
        return ret
