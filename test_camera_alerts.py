"""Local tests for temporal alert gates; no camera or external messages."""

import unittest

from camera_fire_temporal_zalo import TemporalVerifier


BOX = (20, 20, 40, 60)


class AlertConfirmationTests(unittest.TestCase):
    def observe(self, observations, frames=23):
        verifier = TemporalVerifier()
        states = []
        for index in range(frames):
            states = verifier.update(observations(index), index / 10)
        return verifier, states

    def test_first_frame_does_not_alert(self):
        _, states = self.observe(lambda _: [(2, 0.95, BOX)], frames=1)
        self.assertFalse(states[0]["alarm"])

    def test_persistent_confident_region_alerts_after_two_seconds(self):
        _, states = self.observe(lambda _: [(2, 0.90, BOX)])
        self.assertTrue(states[0]["alarm"])

    def test_low_confidence_flash_is_ignored(self):
        _, states = self.observe(lambda _: [(4, 0.425, BOX)])
        self.assertEqual(states, [])

    def test_mean_confidence_gate_rejects_weak_persistent_region(self):
        _, states = self.observe(lambda _: [(2, 0.61, BOX)])
        self.assertFalse(states[0]["alarm"])

    def test_intermittent_detections_do_not_alert(self):
        _, states = self.observe(lambda i: [] if i % 3 == 0 else [(2, 0.95, BOX)])
        self.assertFalse(any(state["alarm"] for state in states))

    def test_short_misses_can_recover(self):
        _, states = self.observe(lambda i: [] if i in (8, 14) else [(2, 0.90, BOX)])
        self.assertTrue(states[0]["alarm"])

    def test_two_distant_regions_do_not_combine_their_hits(self):
        distant = (200, 200, 220, 240)
        _, states = self.observe(lambda i: [(2, 0.95, BOX if i % 2 else distant)])
        self.assertFalse(any(state["alarm"] for state in states))

    def test_current_frame_must_still_contain_region(self):
        verifier, states = self.observe(lambda _: [(2, 0.90, BOX)])
        self.assertTrue(states[0]["alarm"])
        states = verifier.update([], 2.3)
        self.assertFalse(states[0]["alarm"])

    def test_camera_gap_restarts_confirmation(self):
        verifier, states = self.observe(lambda _: [(2, 0.90, BOX)])
        self.assertTrue(states[0]["alarm"])
        states = verifier.update([(2, 0.90, BOX)], 5.0)
        self.assertFalse(states[0]["alarm"])
        self.assertEqual(states[0]["age"], 0)

    def test_too_few_samples_do_not_alert(self):
        verifier = TemporalVerifier()
        for now in (0.0, 1.0, 2.0):
            states = verifier.update([(2, 0.95, BOX)], now)
        self.assertFalse(states[0]["alarm"])

    def test_all_five_numeric_classes_remain_eligible(self):
        _, states = self.observe(lambda _: [(class_id, 0.90, BOX) for class_id in range(5)])
        self.assertEqual(len(states), 5)
        self.assertTrue(all(state["alarm"] for state in states))


if __name__ == "__main__":
    unittest.main()
