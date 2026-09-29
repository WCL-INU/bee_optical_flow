import unittest

import cv2
import numpy as np

from src.detect_entrance import detect_frames


class DetectionTests(unittest.TestCase):
    def test_blank_scene_has_no_coordinates(self):
        frame = np.full((240, 320, 3), 120, np.uint8)
        result = detect_frames([frame] * 3)
        self.assertEqual(result["status"], "failed")
        self.assertIsNone(result["roi"])

    def test_upper_half_structure_is_ignored(self):
        frame = np.full((240, 320, 3), 120, np.uint8)
        cv2.rectangle(frame, (10, 30), (310, 60), (0, 0, 0), -1)
        self.assertEqual(detect_frames([frame] * 3)["candidates"], [])

    def test_gray_structure_needs_review_and_stays_in_lower_half(self):
        frame = np.full((241, 320, 3), 180, np.uint8)
        cv2.rectangle(frame, (10, 190), (310, 200), (30, 30, 30), -1)
        result = detect_frames([frame] * 3)
        self.assertEqual(result["status"], "needs_review")
        rx1, ry1, rx2, ry2 = result["roi"]
        x1, y1, x2, y2 = result["entrance"]
        self.assertTrue(0 <= rx1 <= x1 < x2 <= rx2 <= 320)
        self.assertTrue(121 <= ry1 <= y1 < y2 <= ry2 <= 241)

    def test_competing_structures_are_rejected(self):
        frame = np.full((400, 400, 3), 180, np.uint8)
        for y in (230, 340):
            cv2.rectangle(frame, (10, y), (390, y + 10), (30, 30, 30), -1)
        result = detect_frames([frame] * 3)
        self.assertEqual(result["status"], "failed")
        self.assertIsNone(result["entrance"])

    def test_insufficient_frames_are_rejected(self):
        with self.assertRaises(ValueError):
            detect_frames([])


if __name__ == "__main__":
    unittest.main()
