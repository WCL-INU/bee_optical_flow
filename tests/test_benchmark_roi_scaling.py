from argparse import Namespace
from pathlib import Path
import unittest

import numpy as np

from src.benchmark_optical_flow import resolve_config
from src.benchmark_roi_scaling import (
    PYRAMID_PIXEL_FACTOR,
    build_jobs,
    evenly_spaced_starts,
    fit_linear,
    fit_log_log,
    scaled_dimensions,
)
from src.bee_entrance_count import FARNEBACK_PARAMS


class BenchmarkRoiScalingTests(unittest.TestCase):
    def test_scaled_dimensions_and_validation(self):
        self.assertEqual(scaled_dimensions(1602, 273, 0.35), (561, 96))
        self.assertEqual(scaled_dimensions(1, 1, 0.01), (1, 1))
        for scale in (0, -1, float("nan"), float("inf")):
            with self.subTest(scale=scale), self.assertRaises(ValueError):
                scaled_dimensions(100, 50, scale)

    def test_evenly_spaced_segments_include_start_middle_and_end(self):
        self.assertEqual(evenly_spaced_starts(2880, 600, 3), [0, 1140, 2280])
        self.assertEqual(evenly_spaced_starts(600, 600, 1), [0])
        with self.assertRaises(ValueError):
            evenly_spaced_starts(599, 600, 1)

    def test_linear_and_log_models_recover_known_relationships(self):
        x = np.array([10.0, 20.0, 40.0, 80.0])
        linear = fit_linear(x, 3.0 + 2.0 * x)
        self.assertAlmostEqual(linear["intercept_ms"], 3.0)
        self.assertAlmostEqual(linear["slope_ms_per_pixel"], 2.0)
        self.assertAlmostEqual(linear["r2"], 1.0)

        logarithmic = fit_log_log(x, 4.0 * x**1.25)
        self.assertAlmostEqual(logarithmic["log_exponent"], 1.25)
        self.assertAlmostEqual(logarithmic["log_r2"], 1.0)

    def test_pyramid_pixel_factor_tracks_runtime_parameters(self):
        expected = sum(
            FARNEBACK_PARAMS["pyr_scale"] ** (2 * level)
            for level in range(FARNEBACK_PARAMS["levels"])
        )
        self.assertAlmostEqual(PYRAMID_PIXEL_FACTOR, expected)

    def test_jobs_apply_native_to_all_and_controlled_to_selected_device(self):
        videos = [Path("summer14.mp4"), Path("summer16.mp4")]
        records = {
            str(videos[0]): {"device": "ANU-25-summer-14", "roi": [0, 0, 100, 50]},
            str(videos[1]): {"device": "ANU-25-summer-16", "roi": [0, 0, 80, 40]},
        }
        args = Namespace(
            segment_frames=60,
            segments=2,
            experiments=["native", "controlled"],
            controlled_device="ANU-25-summer-14",
            scales=[0.5, 1.0],
        )
        jobs = build_jobs(args, videos, records, {video: 120 for video in videos})
        native = [job for job in jobs if job.experiment == "native"]
        controlled = [job for job in jobs if job.experiment == "controlled"]
        self.assertEqual(len(native), 4)
        self.assertEqual(len(controlled), 4)
        self.assertEqual({job.video for job in controlled}, {videos[0]})
        self.assertEqual({job.start_frame for job in controlled}, {0, 60})

    def test_full_pipeline_benchmark_accepts_yaml_resolved_coordinates(self):
        video = Path("summer14.mp4")
        args = Namespace(
            preset="selected",
            preview_stride=999999,
            roi_yaml=Path("regions.yaml"),
            region_records={
                str(video): {
                    "roi": [10, 20, 110, 70],
                    "entrance": [20, 30, 100, 60],
                }
            },
        )
        config = resolve_config(video, args)
        self.assertEqual(
            (config.roi_x1, config.roi_y1, config.roi_x2, config.roi_y2),
            (10, 20, 110, 70),
        )
        self.assertEqual(
            (config.ent_x1, config.ent_y1, config.ent_x2, config.ent_y2),
            (20, 30, 100, 60),
        )


if __name__ == "__main__":
    unittest.main()
