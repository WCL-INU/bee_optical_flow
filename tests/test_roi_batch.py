from contextlib import redirect_stdout
import csv
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np
import yaml

from src.main import (parse_args, select_videos, build_config, resolve_config_for_video,
                      run_batch, main)
from src.roi_batch import resolve_regions


class RoiBatchTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.videos = []
        for device, stamp in [('8', '20260731_235959'), ('8', '20260801_000000'), ('18', '20260801_000000')]:
            path = self.root / f'ANU-25-summer-{device}_{stamp}.mp4'
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), 24, (100, 100))
            self.assertTrue(writer.isOpened())
            for index in range(4):
                frame = np.zeros((100, 100, 3), np.uint8)
                cv2.circle(frame, (45 + index, 70), 3, (255, 255, 255), -1)
                writer.write(frame)
            writer.release()
            self.videos.append(path)
        self.records = [dict(device='ANU-25-summer-8', start='2026-07-01', end='2026-07-31',
                             roi=[0, 40, 100, 100], entrance=[20, 60, 80, 90], image_size=[100, 100]),
                        dict(device='ANU-25-summer-8', start='2026-08-01', end='2026-08-31',
                             roi=[10, 40, 100, 100], entrance=[30, 60, 80, 90], image_size=[100, 100])]
        self.yaml = self.root / 'regions.yaml'
        self.write_yaml()
        self.argv = ['--video-dir', str(self.root), '--roi-yaml', str(self.yaml), '--devices', '8',
                     '--start-date', '2026-07-31', '--end-date', '2026-08-01',
                     '--output-root', str(self.root / 'output')]

    def write_yaml(self):
        self.yaml.write_text(yaml.safe_dump(dict(version=1, regions=self.records)))

    def test_selection_and_real_batch_uses_changed_regions(self):
        args = parse_args(self.argv)
        videos = select_videos(args)
        self.assertEqual(videos, self.videos[:2])
        args.region_records = resolve_regions(self.yaml, videos)
        config = build_config(args)
        self.assertEqual(resolve_config_for_video(videos[0], config, args).ent_x1, 20)
        self.assertEqual(resolve_config_for_video(videos[1], config, args).ent_x1, 30)
        with redirect_stdout(io.StringIO()):
            args.workers = 2
            result = run_batch(videos, self.root / 'result', config, args)
        self.assertEqual(result['roi_x1'].tolist(), [0, 10])
        self.assertTrue((self.root / 'result' / 'batch_summary.csv').is_file())
        self.assertEqual(len(list((self.root / 'result').glob('*_preview.mp4'))), 2)
        args.workers = 1
        with patch('src.main.process_video', side_effect=AssertionError('completed video reprocessed')), redirect_stdout(io.StringIO()):
            resumed = run_batch(videos, self.root / 'result', config, args)
        self.assertEqual(resumed['result_status'].tolist(), ['reused_verified', 'reused_verified'])
        self.assertEqual(resumed['roi_x1'].tolist(), [0, 10])

    def test_overlap_and_resolution_fail_before_processing(self):
        for modification in ('overlap', 'size'):
            with self.subTest(modification=modification):
                original = self.records.copy()
                if modification == 'overlap':
                    self.records = self.records + [self.records[0]]
                else:
                    self.records = [dict(self.records[0], image_size=[200, 100]), self.records[1]]
                self.write_yaml()
                with patch('sys.argv', ['main'] + self.argv), patch('src.main.process_video') as process:
                    with self.assertRaises(ValueError):
                        main()
                    process.assert_not_called()
                self.records = original

    def test_missing_coordinates_skip_and_continue_real_analysis(self):
        self.records = self.records[1:]
        self.write_yaml()
        output = io.StringIO()
        with patch('sys.argv', ['main'] + self.argv), redirect_stdout(output):
            main()
        self.assertIn(f'[SKIP] {self.videos[0].name}', output.getvalue())
        self.assertIn('분석 대상 1개, 좌표 없음으로 건너뜀 1개', output.getvalue())
        result = self.root / 'output' / 'selected'
        self.assertFalse((result / f'{self.videos[0].stem}_preview.mp4').exists())
        self.assertTrue((result / f'{self.videos[1].stem}_preview.mp4').is_file())
        summary = (result / 'batch_summary.csv').read_text()
        self.assertIn(self.videos[1].name, summary)
        self.assertNotIn(self.videos[0].name, summary)
        with (result / 'skipped_videos.csv').open(encoding='utf-8-sig') as stream:
            skipped = list(csv.DictReader(stream))
        self.assertEqual(len(skipped), 1)
        self.assertEqual(skipped[0]['video'], self.videos[0].name)
        self.assertEqual(skipped[0]['device'], 'ANU-25-summer-8')
        self.assertEqual(skipped[0]['recorded_at'], '2026-07-31 23:59:59')
        self.assertIn('좌표 정보가 없습니다', skipped[0]['reason'])
        self.assertIn('skipped_videos.csv', output.getvalue().splitlines()[-1])

    def test_all_missing_exits_without_modifying_outputs(self):
        self.records = []
        self.write_yaml()
        result = self.root / 'output' / 'selected'
        result.mkdir(parents=True)
        summary = result / 'batch_summary.csv'
        summary.write_text('previous results')
        output = io.StringIO()
        with patch('sys.argv', ['main'] + self.argv), patch('src.main.process_video') as process, redirect_stdout(output):
            main()
            process.assert_not_called()
        self.assertEqual(output.getvalue().count('[SKIP]'), 2)
        self.assertIn('분석 대상 0개', output.getvalue())
        self.assertEqual(summary.read_text(), 'previous results')
        with (result / 'skipped_videos.csv').open(encoding='utf-8-sig') as stream:
            self.assertEqual(len(list(csv.DictReader(stream))), 2)

    def test_skipped_report_survives_processing_failure(self):
        self.records = self.records[1:]
        self.write_yaml()
        with patch('sys.argv', ['main'] + self.argv), patch('src.main.run_batch', side_effect=RuntimeError('failure')), redirect_stdout(io.StringIO()):
            with self.assertRaises(RuntimeError):
                main()
        report = self.root / 'output' / 'selected' / 'skipped_videos.csv'
        self.assertIn(self.videos[0].name, report.read_text(encoding='utf-8-sig'))

    def test_skip_report_replaced_and_dry_run_leaves_it_untouched(self):
        report = self.root / 'output' / 'selected' / 'skipped_videos.csv'
        report.parent.mkdir(parents=True)
        report.write_text('previous report')
        with patch('sys.argv', ['main'] + self.argv + ['--dry-run']), redirect_stdout(io.StringIO()):
            main()
        self.assertEqual(report.read_text(), 'previous report')
        with patch('sys.argv', ['main'] + self.argv), patch('src.main.run_batch'), redirect_stdout(io.StringIO()):
            main()
        with report.open(encoding='utf-8-sig') as stream:
            reader = csv.DictReader(stream)
            self.assertEqual(list(reader), [])
            self.assertIn('reason', reader.fieldnames)

    def test_dry_run_datetime_and_existing_preview(self):
        args = parse_args(self.argv)
        output = args.output_root / args.preset
        output.mkdir(parents=True)
        (output / f'{self.videos[0].stem}_preview.mp4').touch()
        self.assertEqual(len(select_videos(args)), 2)
        argv = ['--video-dir', str(self.root), '--devices', '8', '18', '--start-datetime',
                '2026-08-01 00:00:00', '--end-datetime', '2026-08-01 00:00:00']
        self.assertEqual(set(select_videos(parse_args(argv))), set(self.videos[1:]))
        with patch('sys.argv', ['main'] + self.argv + ['--dry-run']), patch('src.main.process_video') as process, redirect_stdout(io.StringIO()):
            main()
            process.assert_not_called()
        self.assertFalse((output / 'batch_summary.csv').exists())

    def test_conflicting_arguments(self):
        for extra in [['--roi', '0', '0', '100', '100'], ['--reuse-existing']]:
            with redirect_stdout(io.StringIO()), patch('sys.stderr', io.StringIO()), self.assertRaises(SystemExit):
                parse_args(self.argv + extra)

    def test_dry_run_never_opens_videos(self):
        with patch('sys.argv', ['main'] + self.argv + ['--dry-run']), patch('src.roi_batch.cv2.VideoCapture') as capture, redirect_stdout(io.StringIO()):
            main()
            capture.assert_not_called()
        self.assertFalse((self.root / 'output').exists())

    def test_dry_run_validates_geometry_without_video_inspection(self):
        self.records[0]['entrance'] = [0, 0, 100, 100]
        self.write_yaml()
        with patch('src.roi_batch.cv2.VideoCapture') as capture:
            with self.assertRaises(ValueError):
                resolve_regions(self.yaml, self.videos[:2], inspect_videos=False)
            capture.assert_not_called()


if __name__ == '__main__':
    unittest.main()
