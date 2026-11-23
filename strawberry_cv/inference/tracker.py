"""轻量 IoU 跟踪 + 虚拟线越线计数。

不依赖 ByteTrack / torch，便于在没有模型权重时先测通逻辑。
正式视频 Demo 可把 IoUTracker 替换为 Ultralytics 的 ByteTrack/BoT-SORT，
接口保持 update()/active_tracks 即可。
"""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, field

from ..utils.geometry import Box, center, iou, line_side


@dataclass
class Track:
    track_id: int
    box: Box
    state: str
    state_conf: float
    hits: int = 1
    age: int = 0
    missed: int = 0
    state_history: deque = field(default_factory=lambda: deque(maxlen=7))
    counted: bool = False

    def __post_init__(self) -> None:
        if not self.state_history:
            self.state_history.append((self.state, self.state_conf))

    @property
    def centroid(self) -> tuple[float, float]:
        return center(self.box)

    def update_state(self, state: str, conf: float) -> None:
        self.state_history.append((state, conf))
        counts = Counter(s for s, _ in self.state_history)
        best = max(counts.values())
        candidates = [s for s, c in counts.items() if c == best]
        # 平票时用最近一帧
        self.state = state if state in candidates else candidates[-1]
        self.state_conf = conf


class IoUTracker:
    def __init__(self, iou_threshold: float = 0.3, max_age: int = 30):
        self.iou_threshold = float(iou_threshold)
        self.max_age = int(max_age)
        self.tracks: list[Track] = []
        self._next_id = 1

    def reset(self) -> None:
        self.tracks = []
        self._next_id = 1

    def update(self, detections: list[dict]) -> list[Track]:
        """detections: [{"box": xyxy, "state": str, "state_conf": float}]"""
        unmatched_tracks = set(range(len(self.tracks)))
        matched_pairs: list[tuple[int, int]] = []

        for di, det in enumerate(detections):
            best_ti, best_iou = -1, 0.0
            for ti in unmatched_tracks:
                value = iou(det["box"], self.tracks[ti].box)
                if value > best_iou:
                    best_iou, best_ti = value, ti
            if best_ti >= 0 and best_iou >= self.iou_threshold:
                unmatched_tracks.discard(best_ti)
                matched_pairs.append((best_ti, di))

        for ti, di in matched_pairs:
            track = self.tracks[ti]
            det = detections[di]
            track.box = det["box"]
            track.hits += 1
            track.missed = 0
            track.age += 1
            track.update_state(det["state"], float(det.get("state_conf", 1.0)))

        for ti in list(unmatched_tracks):
            track = self.tracks[ti]
            track.missed += 1
            track.age += 1

        matched_di = {di for _, di in matched_pairs}
        for di, det in enumerate(detections):
            if di in matched_di:
                continue
            track = Track(
                track_id=self._next_id,
                box=det["box"],
                state=det["state"],
                state_conf=float(det.get("state_conf", 1.0)),
            )
            self._next_id += 1
            self.tracks.append(track)

        self.tracks = [t for t in self.tracks if t.missed <= self.max_age]
        return list(self.tracks)

    @property
    def active_tracks(self) -> list[Track]:
        return [t for t in self.tracks if t.missed == 0]


class LineCounter:
    """有向线段越线计数：每个轨迹只计一次。"""

    def __init__(
        self,
        p1: tuple[float, float],
        p2: tuple[float, float],
        min_hits: int = 3,
        debounce: int = 3,
    ):
        self.p1 = p1
        self.p2 = p2
        self.min_hits = int(min_hits)
        self.debounce = max(0, int(debounce))
        self.counts: Counter[str] = Counter()
        self._last_side: dict[int, float] = {}
        self._pending: dict[int, tuple[float, int]] = {}

    def reset(self) -> None:
        self.counts.clear()
        self._last_side.clear()
        self._pending.clear()

    @property
    def total(self) -> int:
        return int(sum(self.counts.values()))

    def update(self, tracks: list[Track], frame_index: int) -> Counter[str]:
        for track in tracks:
            side = line_side(track.centroid, self.p1, self.p2)
            sign = 1.0 if side >= 0 else -1.0
            last = self._last_side.get(track.track_id)

            if last is not None and sign != last and track.hits >= self.min_hits:
                if self.debounce == 0:
                    if not track.counted:
                        self.counts[track.state] += 1
                        track.counted = True
                else:
                    self._pending[track.track_id] = (sign, frame_index)
            if track.track_id in self._pending:
                pending_sign, start = self._pending[track.track_id]
                if sign != pending_sign:
                    self._pending.pop(track.track_id, None)
                elif frame_index - start >= self.debounce and not track.counted:
                    self.counts[track.state] += 1
                    track.counted = True
                    self._pending.pop(track.track_id, None)

            self._last_side[track.track_id] = sign
        return self.counts
