import csv
from contextlib import redirect_stdout
from datetime import datetime, timedelta
import io
from pathlib import Path
import tempfile
import shutil
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from src.make_timelapse import caption_frame, fit_frame, main, sample_frames, ordered_samples
from src.timelapse_ffmpeg import sample_frames_ffmpeg, extraction_command, check_backend


class TimelapseTests(unittest.TestCase):
    def test_first_sampling_reads_consecutively_without_seeking(self):
        cap = unittest.mock.MagicMock()
        cap.isOpened.return_value = True
        cap.get.side_effect = lambda prop: 2400 if prop == cv2.CAP_PROP_FRAME_COUNT else 24
        cap.read.return_value = (True, np.zeros((240, 320, 3), np.uint8))
        with patch("src.make_timelapse.cv2.VideoCapture", return_value=cap):
            frames, fps, failures = sample_frames(Path("clip.mp4"), 24, 320, "first")
        cap.set.assert_not_called()
        self.assertEqual(cap.read.call_count, 24)
        self.assertEqual([index for index, _ in frames], list(range(24)))
        cap.release.assert_called_once()

    @unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg not installed")
    def test_first_ffmpeg_sampling_stops_at_requested_frames(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source.avi"
            self.make_clip(path, frames=40)
            frames, fps, _ = sample_frames_ffmpeg(path, 24, 320, "ffmpeg", sampling="first")
            self.assertEqual([index for index, _ in frames], list(range(24)))
            expected, _, _ = sample_frames(path, 24, 320, "first")
            for (_, image), (_, other) in zip(frames, expected):
                self.assertLess(np.abs(image.astype(float) - other).mean(), 3)
        command = extraction_command(Path("source.mp4"), list(range(24)), 320, 240, "cuda", 0, "yuv420p")
        self.assertEqual(command[command.index("-frames:v") + 1], "24")
        self.assertIn("select=lte(n\\,23)", command[command.index("-vf") + 1])

    @unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg not installed")
    def test_ffmpeg_sampling_keeps_indices_and_timestamps(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source.avi"
            self.make_clip(path)
            frames, fps, failures = sample_frames_ffmpeg(path, 24, 320, "ffmpeg")
            expected, expected_fps, _ = sample_frames(path, 24, 320)
            self.assertEqual([i for i, _ in frames], [i for i, _ in expected])
            self.assertEqual(fps, expected_fps)
            self.assertEqual(failures, 0)
            for (_, image), (_, other) in zip(frames, expected):
                self.assertEqual(image.shape, other.shape)
                self.assertLess(np.abs(image.astype(float) - other).mean(), 3)

    def test_cuda_command_keeps_selected_frames_on_gpu_until_download(self):
        command = extraction_command(Path("some video.mp4"), [0, 12, 24], 320, 240, "cuda", 1, "yuv420p")
        self.assertEqual(command[command.index("-hwaccel") + 1], "cuda")
        self.assertEqual(command[command.index("-hwaccel_device") + 1], "1")
        filters = command[command.index("-vf") + 1]
        self.assertLess(filters.index("select="), filters.index("scale_cuda="))
        self.assertLess(filters.index("scale_cuda="), filters.index("hwdownload"))
        with self.assertRaises(ValueError):
            extraction_command(Path("video.mp4"), [0], 320, 240, "cuda", 0, "unsupported")

    def test_cuda_unavailable_is_reported_without_cpu_fallback(self):
        with patch("src.timelapse_ffmpeg.shutil.which", return_value="ffmpeg"), patch(
            "src.timelapse_ffmpeg.run_command", side_effect=[b"scale_cuda", ValueError("CUDA_ERROR_NO_DEVICE")]):
            with self.assertRaisesRegex(ValueError, "CUDA_ERROR_NO_DEVICE"):
                check_backend("cuda")

    def make_clip(self, path, size=(320, 240), frames=8):
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 2, size)
        self.assertTrue(writer.isOpened())
        try:
            for i in range(frames):
                writer.write(np.full((size[1], size[0], 3), (20 + (i % 20) * 10, 70, 110), np.uint8))
        finally:
            writer.release()

    def test_caption_timestamp_and_unknown_fps(self):
        stamp = datetime(2026, 7, 5, 23, 59, 59)
        frame = np.full((240, 320, 3), 80, np.uint8)
        result, captured = caption_frame(frame, stamp, 48, 24, "device-3")
        self.assertEqual(captured, stamp + timedelta(seconds=2))
        self.assertFalse(np.array_equal(result[:50], frame[:50]))
        np.testing.assert_array_equal(result[100:], frame[100:])
        self.assertIsNone(caption_frame(frame, stamp, 48, 0, "device-3")[1])

    def test_letterbox_preserves_aspect(self):
        frame = np.full((200, 100, 3), 255, np.uint8)
        result = fit_frame(frame, (320, 240))
        self.assertEqual(result.shape, (240, 320, 3))
        self.assertFalse(result[:, :100].any())
        self.assertTrue((result[:, 100:220] == 255).all())

    def test_period_selection_frame_count_and_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_clip(root / "a_20260705_060000.avi")
            self.make_clip(root / "a_20260705_070000.avi", (240, 320))
            # These must be excluded before attempting to decode.
            (root / "a_20260705_050000.avi").touch()
            (root / "a_20260705_070001.avi").touch()
            (root / "b_20260705_060000.avi").touch()
            output = root / "result.mp4"
            argv = ["make_timelapse", "--device", "a", "--video-dir", str(root),
                    "--workers", "2", "--pattern", "*.avi", "--start-datetime", "2026-07-05 06:00",
                    "--end-datetime", "2026-07-05 07:00", "--output", str(output)]
            with patch("sys.argv", argv), redirect_stdout(io.StringIO()):
                main()
            cap = cv2.VideoCapture(str(output))
            try:
                self.assertEqual(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 48)
                self.assertAlmostEqual(cap.get(cv2.CAP_PROP_FPS), 24)
                ok, frame = cap.read()
                self.assertTrue(ok)
                self.assertEqual(frame.shape, (240, 320, 3))
                self.assertTrue(np.any(np.all(frame[:60] > 200, axis=2)))
            finally:
                cap.release()
            with output.with_suffix(".csv").open(encoding="utf-8-sig") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 2)
            self.assertEqual([Path(row["video"]).name for row in rows],
                             ["a_20260705_060000.avi", "a_20260705_070000.avi"])
            self.assertEqual([row["output_frames"] for row in rows], ["24", "24"])
            self.assertEqual(rows[1]["output_start_sec"], "1.0")
            self.assertEqual(rows[0]["last_frame_time"], "2026-07-05 06:00:03.500000")
            sampled, fps, failures = sample_frames(root / "a_20260705_060000.avi", 24, 320)
            self.assertEqual(len(sampled), 24)
            self.assertEqual((sampled[0][0], sampled[-1][0]), (0, 7))
            self.assertEqual(failures, 0)

    def test_all_unreadable_keeps_existing_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "a_20260705_060000.mp4").touch()
            output = root / "result.mp4"
            output.write_bytes(b"existing output")
            argv = ["make_timelapse", "--device", "a", "--video-dir", str(root), "--output", str(output)]
            with patch("sys.argv", argv), patch("src.make_timelapse.sample_frames", side_effect=ValueError("bad video")), redirect_stdout(io.StringIO()):
                with self.assertRaises(RuntimeError):
                    main()
            self.assertEqual(output.read_bytes(), b"existing output")

    def test_parallel_order_errors_and_bounded_prefetch(self):
        consumed = []
        def source():
            for index in range(6):
                consumed.append(index)
                yield index, Path(str(index))
        def extract(path, *args):
            if str(path) == "0":
                import time
                time.sleep(0.02)
            return None, "bad video" if str(path) == "1" else None
        with patch("src.make_timelapse.extract_video", side_effect=extract):
            samples = ordered_samples(source(), 24, 1280, 2)
            try:
                first = next(samples)
                self.assertEqual(first[0], 0)
                self.assertEqual(consumed, [0, 1])
                rest = list(samples)
            finally:
                samples.close()
        self.assertEqual([item[0] for item in rest], [1, 2, 3, 4, 5])
        self.assertEqual(rest[0][2][1], "bad video")


if __name__ == "__main__":
    unittest.main()
