from datetime import datetime
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backend_bases import KeyEvent
import numpy as np

from src.edit_roi import (EditorState, check_period, discover_videos, edit_frame,
                          parse_boundary, read_records, save_record)


class RoiEditorTests(unittest.TestCase):
    def test_period_filter_uses_device_and_includes_end_day(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ['ANU-25-summer-1_20260701_000000.mp4',
                         'ANU-25-summer-1_20260705_235959.mp4',
                         'ANU-25-summer-1_20260706_000000.mp4',
                         'ANU-25-summer-10_20260705_120000.mp4']:
                (root / name).touch()
            videos = discover_videos(root, 'ANU-25-summer-1', parse_boundary('2026-07-01'),
                                     parse_boundary('2026-07-05', True))
            self.assertEqual(len(videos), 2)
            self.assertIn('235959', videos[-1].name)
            self.assertEqual(parse_boundary('2026-07-05 12:34'), datetime(2026, 7, 5, 12, 34))

    def test_edges_move_only_perpendicularly_and_preserve_containment(self):
        state = EditorState([0, 0, 100, 100], [20, 20, 80, 80], 100, 100)
        self.assertFalse(state.move('down'))
        state.cycle()  # ROI top
        self.assertFalse(state.move('left'))
        self.assertTrue(state.move('down', 20))
        self.assertFalse(state.move('down'))  # would exclude ENT
        for _ in range(4):
            state.cycle()
        self.assertTrue(state.move('down', 10))  # ENT top
        self.assertFalse(state.move('up', 20))
        self.assertEqual(state.boxes['entrance'], [20, 30, 80, 80])
        for _ in range(4):
            state.cycle()
        self.assertEqual(state.selected, 0)

    def test_yaml_roundtrip_update_and_overlap_preserve_existing(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'regions.yaml'
            record = dict(device='test', start='2026-07-01 00:00:00', end='2026-07-05 23:59:59',
                          roi=[0, 0, 100, 100], entrance=[20, 20, 80, 80])
            save_record(path, record)
            self.assertIn('roi: [0, 0, 100, 100]', path.read_text())
            record['entrance'] = [21, 20, 80, 80]
            save_record(path, record)
            self.assertEqual(read_records(path)['regions'], [record])
            before = path.read_bytes()
            with self.assertRaises(ValueError):
                save_record(path, dict(record, start='2026-07-05 12:00:00'))
            self.assertEqual(path.read_bytes(), before)
            save_record(path, dict(record, start='2026-07-06 00:00:00', end='2026-07-07 00:00:00'))
            self.assertEqual(len(read_records(path)['regions']), 2)

    def test_keyboard_frame_navigation_edit_save_and_cancel(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'test.avi'
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'MJPG'), 24, (100, 100))
            self.assertTrue(writer.isOpened())
            for value in [0, 60, 120]:
                writer.write(np.full((100, 100, 3), value, dtype=np.uint8))
            writer.release()
            existing = dict(roi=[0, 0, 100, 100], entrance=[20, 20, 80, 80], image_size=[100, 100])

            def operate():
                fig = plt.gcf()
                for key in [' ', 'down', 'n', 'n', 'n', 'b']:
                    fig.canvas.callbacks.process('key_press_event', KeyEvent('key_press_event', fig.canvas, key=key))
                self.assertIn('Frame 1/2', fig.axes[0].get_title())
                self.assertAlmostEqual(float(np.mean(fig.axes[0].images[0].get_array())), 60, delta=3)
                fig.canvas.draw()  # exercise overlay and help rendering
                fig.canvas.callbacks.process('key_press_event', KeyEvent('key_press_event', fig.canvas, key='enter'))

            with patch('matplotlib.pyplot.show', side_effect=operate):
                record = edit_frame(path, 'test', datetime(2026, 7, 1), datetime(2026, 7, 5), existing)
            self.assertEqual(record['roi'], [0, 1, 100, 100])
            self.assertEqual(record['reference_frame'], 1)

            def cancel():
                fig = plt.gcf()
                fig.canvas.callbacks.process('key_press_event', KeyEvent('key_press_event', fig.canvas, key='escape'))
            with patch('matplotlib.pyplot.show', side_effect=cancel):
                self.assertIsNone(edit_frame(path, 'test', datetime(2026, 7, 1), datetime(2026, 7, 5), existing))


if __name__ == '__main__':
    unittest.main()
