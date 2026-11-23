import unittest

from strawberry_cv.inference.tracker import IoUTracker, LineCounter, Track


class TrackerTests(unittest.TestCase):
    def test_iou_tracking_and_age(self):
        tracker = IoUTracker(iou_threshold=0.3, max_age=2)
        tracks = tracker.update([{"box": (10, 10, 50, 50), "state": "ripe", "state_conf": 0.9}])
        self.assertEqual(len(tracks), 1)
        first_id = tracks[0].track_id
        tracks = tracker.update([{"box": (12, 10, 52, 50), "state": "ripe", "state_conf": 0.8}])
        self.assertEqual(tracks[0].track_id, first_id)
        tracks = tracker.update([{"box": (100, 100, 140, 140), "state": "unripe", "state_conf": 0.8}])
        self.assertEqual(len(tracks), 2)
        self.assertEqual(len(tracker.active_tracks), 1)
        tracker.update([])
        tracker.update([])
        ids = {t.track_id for t in tracker.tracks}
        self.assertNotIn(first_id, ids)

    def test_line_counter_counts_once(self):
        counter = LineCounter((100, 0), (100, 150), min_hits=1, debounce=0)
        track = Track(1, (40, 50, 80, 110), "ripe", 0.9, hits=1)
        counter.update([track], 0)
        track.box = (90, 50, 130, 110)
        counter.update([track], 1)
        self.assertEqual(counter.counts["ripe"], 1)
        track.box = (100, 50, 140, 110)
        counter.update([track], 2)
        self.assertEqual(counter.counts["ripe"], 1)

    def test_line_counter_debounce(self):
        counter = LineCounter((100, 0), (100, 150), min_hits=1, debounce=2)
        track = Track(1, (40, 50, 80, 110), "half", 0.9, hits=1)
        counter.update([track], 0)
        track.box = (90, 50, 130, 110)
        counter.update([track], 1)   # pending
        counter.update([track], 2)   # 1 帧，未确认
        self.assertEqual(counter.counts["half"], 0)
        counter.update([track], 3)   # 2 帧，确认
        self.assertEqual(counter.counts["half"], 1)


if __name__ == "__main__":
    unittest.main()
