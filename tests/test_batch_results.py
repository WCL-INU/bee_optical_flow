from contextlib import redirect_stdout
from dataclasses import replace
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

import cv2
import numpy as np
import pandas as pd

from src.batch_results import paths_for, process_or_reuse, reusable_result
from src.bee_entrance_count import Config, process_video


class ResumeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.video = self.root / 'test.mp4'
        writer = cv2.VideoWriter(str(self.video), cv2.VideoWriter_fourcc(*'mp4v'), 24, (100, 100))
        for index in range(10):
            frame = np.zeros((100, 100, 3), np.uint8)
            cv2.circle(frame, (40+index, 60), 3, (255, 255, 255), -1)
            writer.write(frame)
        writer.release()
        self.config = Config(roi_x1=0, roi_y1=40, roi_x2=100, roi_y2=100,
                             ent_x1=20, ent_y1=50, ent_x2=80, ent_y2=90, preview_stride=3)
        self.output = self.root / 'results'
        self.paths = paths_for(self.video, self.output)

    def run_process(self, config=None, force=False, process=process_video):
        with redirect_stdout(io.StringIO()):
            return process_or_reuse(self.video, self.output, config or self.config, process, force)

    def test_verified_resume_setting_change_and_force(self):
        first = self.run_process()
        processor = Mock(wraps=process_video)
        second = self.run_process(process=processor)
        processor.assert_not_called()
        self.assertEqual(second['result_status'], 'reused_verified')
        self.assertEqual(first['total_filtered_in_flux'], second['total_filtered_in_flux'])
        self.run_process(replace(self.config, ent_x1=21), process=processor)
        self.assertEqual(processor.call_count, 1)
        self.run_process(replace(self.config, ent_x1=21), force=True, process=processor)
        self.assertEqual(processor.call_count, 2)

    def test_legacy_reuse_and_truncated_csv_rejected(self):
        process_video(self.video, self.output, self.config)
        result = self.run_process(process=Mock(side_effect=AssertionError('must reuse')))
        self.assertEqual(result['result_status'], 'reused_legacy_unverified')
        self.assertIsNone(result['roi_x1'])
        self.assertTrue(np.isnan(result['optical_flow_time_sec']))
        self.assertFalse(self.paths['completion'].exists())
        frames = pd.read_csv(self.paths['frame'])
        frames.iloc[:-1].to_csv(self.paths['frame'], index=False)
        self.assertIsNone(reusable_result(self.video, self.output, self.config)[0])

    def test_interrupted_marker_cannot_reuse_old_results(self):
        self.run_process()
        with self.assertRaises(RuntimeError):
            self.run_process(force=True, process=Mock(side_effect=RuntimeError('interrupted')))
        self.assertEqual(json.loads(self.paths['completion'].read_text())['state'], 'running')
        self.assertIsNone(reusable_result(self.video, self.output, self.config)[0])
        self.assertEqual(self.run_process()['result_status'], 'processed')

    def test_tampered_output_and_input_changes_rejected(self):
        self.run_process()
        with self.paths['window'].open('a') as stream:
            stream.write('\n')
        self.assertIsNone(reusable_result(self.video, self.output, self.config)[0])
        self.run_process()
        with self.video.open('ab') as stream:
            stream.write(b'changed')
        self.assertIsNone(reusable_result(self.video, self.output, self.config)[0])

    def test_partial_preview_and_bad_window_legacy_rejected(self):
        process_video(self.video, self.output, self.config)
        windows = pd.read_csv(self.paths['window'])
        windows.loc[0, 'raw_in_flux_sum'] += 123
        windows.to_csv(self.paths['window'], index=False)
        self.assertIsNone(reusable_result(self.video, self.output, self.config)[0])
        process_video(self.video, self.output, self.config)
        self.paths['preview'].write_bytes(b'incomplete')
        self.assertIsNone(reusable_result(self.video, self.output, self.config)[0])


if __name__ == '__main__':
    unittest.main()
