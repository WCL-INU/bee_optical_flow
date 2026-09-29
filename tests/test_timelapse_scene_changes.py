import csv
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from src.detect_scene_changes import main
from src.timelapse_scene_changes import read_manifest


class TimelapseSceneTests(unittest.TestCase):
    def fixture(self, root):
        path = root / "device1.mp4"
        rng = np.random.default_rng(30)
        image = np.full((240, 320, 3), 120, np.uint8)
        for _ in range(400):
            x, y = rng.integers([5, 30], [305, 225])
            cv2.rectangle(image, (int(x), int(y)), (int(x + 6), int(y + 6)),
                          (int(rng.integers(15, 240)),) * 3, -1)
        moved = cv2.warpAffine(image, np.float32([[1, 0, 25], [0, 1, 15]]), (320, 240))
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 24, (320, 240))
        self.assertTrue(writer.isOpened())
        try:
            for clip, frame in enumerate((image, moved, moved)):
                for sample in range(24):
                    actual = frame.copy()
                    actual[:25] = 0
                    cv2.putText(actual, f"{clip} {sample}", (0, 20), cv2.FONT_HERSHEY_SIMPLEX,
                                0.6, (255, 255, 255), 1)
                    writer.write(actual)
        finally:
            writer.release()
        fields = ["video", "video_start", "status", "output_start_sec", "output_frames"]
        with path.with_suffix(".csv").open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for clip in range(3):
                writer.writerow(dict(video=f"videos/ANU-25-summer-1_2026070{clip+1}_120000.mp4",
                                     video_start=f"2026-07-0{clip+1} 12:00:00", status="written",
                                     output_start_sec=clip, output_frames=24))
        return path

    def test_legacy_csv_maps_original_times_and_segment_boundaries(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.fixture(root)
            output = root / "reports"
            argv = ["detect_scene_changes", "--timelapse-dir", str(root), "--timelapse-ids", "1",
                    "--output-dir", str(output)]
            with patch("sys.argv", argv), redirect_stdout(io.StringIO()):
                main()
            result = json.loads((output / "device1/scene_changes.json").read_text())
            self.assertEqual(result["processed"], 3)
            self.assertEqual(len(result["changes"]), 1)
            event = result["changes"][0]
            self.assertEqual(event["changed_from_timestamp"], "2026-07-02T12:00:00")
            self.assertEqual(event["confirmed_at_timestamp"], "2026-07-03T12:00:00")
            self.assertEqual(event["timelapse_start_sec"], 1)
            with (output / "device1/video_checks.csv").open(encoding="utf-8-sig") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual([int(r["frame_start"]) for r in rows], [0, 24, 48])
            self.assertTrue(Path(event["evidence"]).exists())
            with patch("sys.argv", argv + ["--start-date", "2026-07-02", "--limit", "1"]), redirect_stdout(io.StringIO()):
                main()
            with (output / "device1/video_checks.csv").open(encoding="utf-8-sig") as stream:
                filtered = list(csv.DictReader(stream))
            self.assertEqual(len(filtered), 1)
            self.assertEqual(filtered[0]["timestamp"], "2026-07-02T12:00:00")
            self.assertEqual(filtered[0]["status"], "baseline")

    def test_mismatched_frame_mapping_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.fixture(Path(directory)).with_suffix(".csv")
            with self.assertRaisesRegex(ValueError, "CSV describes"):
                read_manifest(path, 24, 73)
            content = path.read_text(encoding="utf-8-sig")
            path.write_text(content.replace(",written,1,24", ",written,1.5,24"), encoding="utf-8-sig")
            with self.assertRaisesRegex(ValueError, "Non-contiguous"):
                read_manifest(path, 24, 72)

    def test_missing_manifest_reported_as_error(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.fixture(root).with_suffix(".csv").unlink()
            argv = ["detect_scene_changes", "--timelapse-dir", str(root), "--timelapse-ids", "1",
                    "--output-dir", str(root / "reports")]
            with patch("sys.argv", argv), redirect_stdout(io.StringIO()), self.assertRaises(SystemExit) as error:
                main()
            self.assertEqual(error.exception.code, 1)
            self.assertTrue((root / "reports/device1/error.json").exists())


if __name__ == "__main__":
    unittest.main()
