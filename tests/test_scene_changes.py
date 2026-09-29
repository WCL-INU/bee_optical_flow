import unittest
from unittest.mock import patch
from contextlib import redirect_stdout
import io
import json
import csv
from pathlib import Path
import tempfile

import cv2
import numpy as np

from src.detect_scene_changes import (Background, ChangeTracker, build_background,
                                      compare_backgrounds, cached_background, prepared_videos, main)
from src.scene_lines import compare_lines, extract_lines


def scene(seed=1):
    rng = np.random.default_rng(seed)
    image = np.full((360, 480), 130, np.uint8)
    for _ in range(600):
        x, y = rng.integers([5, 5], [460, 340])
        size = int(rng.integers(3, 12))
        cv2.rectangle(image, (int(x), int(y)), (int(x) + size, int(y) + size), int(rng.integers(25, 230)), -1)
    return image


def background(image):
    return build_background([image] * 7)


class BackgroundTests(unittest.TestCase):
    @staticmethod
    def structural_scene(seed, level=160):
        rng = np.random.default_rng(seed)
        image = np.clip(rng.normal(level, 8, (360, 480)), 0, 255).astype(np.uint8)
        for start, end in [((30, 90), (440, 90)), ((30, 270), (440, 270)),
                           ((60, 30), (60, 330)), ((410, 30), (410, 330))]:
            cv2.line(image, start, end, 240, 5)
        return image

    def test_surface_change_with_fixed_boundaries_is_same(self):
        dry = background(self.structural_scene(10, 170))
        wet = background(self.structural_scene(20, 65))
        result = compare_backgrounds(dry, wet)
        self.assertEqual(result["status"], "same")
        self.assertEqual(result["reason"], "long structural lines aligned")
        self.assertGreaterEqual(result["matched_lines"], 3)

    def test_moved_structural_boundaries_are_not_accepted_as_same(self):
        image = self.structural_scene(10)
        for matrix in [np.float32([[1, 0, 25], [0, 1, 22]]),
                       cv2.getRotationMatrix2D((240, 180), 8, 1)]:
            changed = cv2.warpAffine(image, matrix, (480, 360))
            self.assertEqual(compare_backgrounds(background(image), background(changed))["status"], "change")

    def test_parallel_lines_do_not_prove_full_alignment(self):
        image = np.zeros((360, 480), np.uint8)
        for y in (60, 180, 300):
            cv2.line(image, (20, y), (460, y), 200, 3)
        bg = background(image)
        _, stationary, _ = compare_lines(bg, bg)
        self.assertFalse(stationary)

    def test_lost_texture_without_geometry_is_unknown(self):
        metrics = dict(reference_lines=0, current_lines=0, matched_lines=0, line_match_ratio=0)
        with patch("src.detect_scene_changes.compare_lines", return_value=(metrics, False, False)):
            self.assertEqual(compare_backgrounds(background(scene(1)), background(scene(2)))["status"], "unknown")

    def test_dynamic_mask_excludes_lines(self):
        image = self.structural_scene(10)
        self.assertEqual(len(extract_lines(image, np.zeros_like(image))), 0)

    def test_moving_foreground_does_not_trigger_change(self):
        image = scene()
        rng = np.random.default_rng(4)
        frames = []
        for _ in range(21):
            frame = image.copy()
            for _ in range(35):
                x, y = rng.integers([0, 0], [480, 360])
                cv2.ellipse(frame, (int(x), int(y)), (5, 3), 0, 0, 360, 5, -1)
            frames.append(frame)
        result = build_background(frames)
        self.assertTrue(result.usable)
        self.assertLess(np.abs(result.image.astype(float) - image).mean(), 1)
        self.assertEqual(compare_backgrounds(background(image), result)["status"], "same")

    def test_brightness_offset_is_not_a_change(self):
        image = scene()
        brighter = np.clip(image.astype(float) + 20, 0, 255).astype(np.uint8)
        self.assertEqual(compare_backgrounds(background(image), background(brighter))["status"], "same")

    def test_shift_and_rotation_are_changes(self):
        image = scene()
        transforms = [np.float32([[1, 0, 25], [0, 1, 18]]),
                      cv2.getRotationMatrix2D((240, 180), 8, 1)]
        for matrix in transforms:
            moved = cv2.warpAffine(image, matrix, (480, 360))
            self.assertEqual(compare_backgrounds(background(image), background(moved))["status"], "change")

    def test_unrelated_scene_is_change_evidence(self):
        self.assertEqual(compare_backgrounds(background(scene(1)), background(scene(2)))["status"], "change")

    def test_dark_or_textureless_scene_is_unknown(self):
        blank = background(np.zeros((360, 480), np.uint8))
        self.assertEqual(compare_backgrounds(background(scene()), blank)["status"], "unknown")


class TrackerTests(unittest.TestCase):
    def setUp(self):
        self.a = Background(np.array([[1]]), None, None, None, True, "usable")
        self.b = Background(np.array([[2]]), None, None, None, True, "usable")
        self.unknown = Background(np.array([[0]]), None, None, None, False, "dark")

    @staticmethod
    def comparison(a, b, *args):
        return {"status": "same" if a is b else "change", "reason": "test"}

    @patch("src.detect_scene_changes.compare_backgrounds", side_effect=comparison)
    def test_confirm_first_changed_video_and_do_not_repeat(self, _):
        tracker = ChangeTracker()
        self.assertEqual(tracker.observe("a", self.a)[0]["status"], "baseline")
        self.assertEqual(tracker.observe("b", self.b)[0]["status"], "pending")
        self.assertEqual(tracker.observe("dark", self.unknown)[0]["status"], "unknown")
        check, event = tracker.observe("c", self.b)
        self.assertEqual(check["status"], "changed")
        self.assertEqual(event[0]["changed_from_video"], "b")
        self.assertEqual(event[0]["last_normal_video"], "a")
        self.assertIsNone(tracker.observe("d", self.b)[1])

    @patch("src.detect_scene_changes.compare_backgrounds", side_effect=comparison)
    def test_single_outlier_does_not_report_change(self, _):
        tracker = ChangeTracker()
        for name, bg in (("a", self.a), ("b", self.b), ("c", self.a)):
            self.assertIsNone(tracker.observe(name, bg)[1])
        self.assertIsNone(tracker.pending)


class CommandTests(unittest.TestCase):
    def test_grouping_change_report_and_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            clips = [("a_20260101_120000.avi", scene(1)),
                     ("a_20260102_120000.avi", scene(2)),
                     ("a_20260103_235959.avi", scene(2)),
                     ("b_20260101_120000.avi", scene(3))]
            for name, frame in clips:
                writer = cv2.VideoWriter(str(root / name), cv2.VideoWriter_fourcc(*"MJPG"), 5, (480, 360))
                self.assertTrue(writer.isOpened())
                try:
                    for _ in range(8):
                        writer.write(cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR))
                finally:
                    writer.release()
            output = root / "report"
            argv = ["detect_scene_changes", "--no-cache", "--video-dir", str(root), "--pattern", "*.avi",
                    "--samples", "7", "--output-dir", str(output)]
            with patch("sys.argv", argv), redirect_stdout(io.StringIO()):
                main()
            report = json.loads((output / "scene_changes.json").read_text())
            self.assertEqual(len(report["changes"]), 1)
            event = report["changes"][0]
            self.assertEqual(event["device"], "a")
            self.assertEqual(event["changed_from_video"], "a_20260102_120000.avi")
            self.assertTrue(Path(event["evidence"]).exists())
            self.assertTrue((output / "video_checks.csv").exists())
            # Inclusive end date must keep late-night recordings; out-of-range
            # files must not be decoded, and the new period gets its own baseline.
            (root / "a_20260104_000000.avi").touch()
            with patch("sys.argv", argv + ["--start-date", "2026-01-02", "--end-date", "2026-01-03"]), redirect_stdout(io.StringIO()):
                main()
            with (output / "video_checks.csv").open(encoding="utf-8-sig") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual([r["video"] for r in rows],
                             ["a_20260102_120000.avi", "a_20260103_235959.avi"])
            self.assertEqual(rows[0]["status"], "baseline")
            report = json.loads((output / "scene_changes.json").read_text())
            self.assertEqual(report["changes"], [])
            self.assertEqual(report["settings"]["end_date"], "2026-01-03")

    def test_invalid_date_ranges_rejected(self):
        for options in (["--start-date", "2026-02-30"],
                        ["--start-date", "2026-04-01", "--end-date", "2026-03-01"],
                        ["--start-datetime", "2026-03-01 25:00"],
                        ["--start-datetime", "2026-03-01 12:00:01", "--end-datetime", "2026-03-01 12:00"],
                        ["--start-date", "2026-03-02", "--end-datetime", "2026-03-01 23:59:59"],
                        ["--start-date", "2026-03-01", "--start-datetime", "2026-03-01 12:00"]):
            with patch("sys.argv", ["detect_scene_changes", *options]), patch("sys.stderr", io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    main()
                self.assertEqual(error.exception.code, 2)

    def test_datetime_filters_include_exact_endpoints(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for stamp in ("115959", "120000", "123000", "130015", "130016"):
                (root / f"a_20260301_{stamp}.mp4").touch()
            output = root / "output"
            argv = ["detect_scene_changes", "--no-cache", "--video-dir", str(root), "--output-dir", str(output),
                    "--start-datetime", "2026-03-01T12:00", "--end-datetime", "2026-03-01 13:00:15"]
            with patch("sys.argv", argv), patch("src.detect_scene_changes.sample_background", return_value=background(scene())) as sample, redirect_stdout(io.StringIO()):
                main()
            self.assertEqual([call.args[0].stem for call in sample.call_args_list],
                             ["a_20260301_120000", "a_20260301_123000", "a_20260301_130015"])
            settings = json.loads((output / "scene_changes.json").read_text())["settings"]
            self.assertEqual(settings["start_datetime"], "2026-03-01T12:00:00")
            self.assertEqual(settings["end_datetime"], "2026-03-01T13:00:15")


class CacheTests(unittest.TestCase):
    def test_cache_reuses_exact_features_and_invalidates_changed_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "video.mp4"
            path.write_bytes(b"original")
            cache = Path(directory) / "cache"
            bg = background(scene())
            with patch("src.detect_scene_changes.sample_background", return_value=bg) as sample:
                first, hit = cached_background(path, 21, 800, cache)
                self.assertFalse(hit)
                second, hit = cached_background(path, 21, 800, cache)
                self.assertTrue(hit)
                self.assertEqual(sample.call_count, 1)
                for field in ("image", "mask", "points", "descriptors"):
                    np.testing.assert_array_equal(getattr(first, field), getattr(second, field))
                self.assertFalse(cached_background(path, 15, 800, cache)[1])
                self.assertFalse(cached_background(path, 21, 640, cache)[1])
                path.write_bytes(b"modified file")
                self.assertFalse(cached_background(path, 21, 800, cache)[1])
                self.assertEqual(sample.call_count, 4)

    def test_damaged_cache_is_rebuilt_and_no_cache_bypasses_disk(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "video.mp4"
            path.touch()
            cache = Path(directory) / "cache"
            with patch("src.detect_scene_changes.sample_background", return_value=background(scene())) as sample:
                cached_background(path, 21, 800, cache)
                next(cache.glob("*.npz")).write_bytes(b"broken")
                self.assertFalse(cached_background(path, 21, 800, cache)[1])
                self.assertTrue(cached_background(path, 21, 800, cache)[1])
                self.assertFalse(cached_background(path, 21, 800, None)[1])
                self.assertEqual(sample.call_count, 3)

    def test_parallel_preparation_preserves_order_and_errors(self):
        videos = [(i, Path(str(i))) for i in range(6)]
        def prepare(path, *args):
            if str(path) == "0":
                import time
                time.sleep(0.02)
            return None, False, 0.0, "unreadable" if str(path) == "3" else None
        with patch("src.detect_scene_changes.prepare_background", side_effect=prepare):
            results = list(prepared_videos(videos, 21, 800, None, 2))
        self.assertEqual([row[0] for row in results], list(range(6)))
        self.assertEqual(results[3][2][-1], "unreadable")


if __name__ == "__main__":
    unittest.main()
