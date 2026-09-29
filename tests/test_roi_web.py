import base64
import json
from pathlib import Path
import re
import tempfile
import threading
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import cv2
import numpy as np

from src.edit_roi import read_records
from src.roi_web import EditorApp, make_server


class WebEditorTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        self.name = 'ANU-25-summer-8_20260701_120000.mp4'
        writer = cv2.VideoWriter(str(root / self.name), cv2.VideoWriter_fourcc(*'mp4v'), 24, (100, 100))
        self.assertTrue(writer.isOpened())
        for value in (20, 100, 200):
            writer.write(np.full((100, 100, 3), value, np.uint8))
        writer.release()
        self.app = EditorApp(root, root / 'regions.yaml')
        self.data = dict(device='8', start='2026-07-01', end='2026-07-01', video=self.name)

    def test_list_open_navigate_save_reload(self):
        listing = self.app.dispatch('list', self.data)
        self.assertEqual(listing['videos'], [self.name])
        self.assertEqual(listing['end'], '2026-07-01 23:59:59')
        opened = self.app.dispatch('open', self.data)
        self.assertEqual(opened['total'], 3)
        frame = self.app.dispatch('frame', dict(self.data, frame=2))
        decoded = cv2.imdecode(np.frombuffer(base64.b64decode(frame['image'].split(',')[1]), np.uint8), cv2.IMREAD_COLOR)
        self.assertGreater(decoded.mean(), 180)
        roi, ent = [0, 40, 100, 100], [20, 60, 80, 90]
        saved = self.app.dispatch('save', dict(self.data, roi=roi, entrance=ent, image_size=[100, 100], frame=2))
        self.assertEqual(saved['record']['reference_frame'], 2)
        reopened = self.app.dispatch('open', self.data)
        self.assertEqual(reopened['roi'], roi)
        self.assertEqual(reopened['entrance'], ent)
        self.assertEqual(len(read_records(self.app.output)['regions']), 1)

    def test_reject_outside_period_path_and_bad_coordinates(self):
        for update in [dict(video='../' + self.name), dict(start='2026-07-02', end='2026-07-02'),
                       dict(frame=-1), dict(frame=3), dict(frame=True)]:
            with self.assertRaises(ValueError):
                self.app.dispatch('open', dict(self.data, **update))
        for roi, ent, size in [([0, 40, 100, 100], [0, 0, 100, 100], [100, 100]),
                               ([0, 40, 101, 100], [20, 60, 80, 90], [100, 100]),
                               ([0, 40, 100, 100], [20, 60, 80, 90], [200, 100])]:
            with self.assertRaises(ValueError):
                self.app.dispatch('save', dict(self.data, roi=roi, entrance=ent, image_size=size))
        self.assertFalse(self.app.output.exists())

    def test_http_page_token_and_roundtrip(self):
        server = make_server(self.app, 0)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(server.server_close)
        self.addCleanup(thread.join, 2)
        self.addCleanup(server.shutdown)
        base = f'http://127.0.0.1:{server.server_port}'
        with urlopen(base + '/', timeout=5) as response:
            html = response.read().decode()
        token = re.search(r"'X-Editor-Token':'([^']+)'", html)[1]
        self.assertNotEqual(token, '__TOKEN__')
        request = Request(base + '/api/list', data=json.dumps(self.data).encode(), headers={'Content-Type': 'application/json'})
        with self.assertRaises(HTTPError) as error:
            urlopen(request, timeout=5)
        self.assertEqual(error.exception.code, 403)
        request.add_header('X-Editor-Token', token)
        with urlopen(request, timeout=5) as response:
            self.assertEqual(json.load(response)['videos'], [self.name])
        request = Request(base + '/api/save', data=b'[]', headers={'X-Editor-Token': token})
        with self.assertRaises(HTTPError) as error:
            urlopen(request, timeout=5)
        self.assertEqual(error.exception.code, 400)


if __name__ == '__main__':
    unittest.main()
