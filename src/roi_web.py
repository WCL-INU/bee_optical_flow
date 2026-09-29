"""Local browser ROI editor, intended for SSH port forwarding."""
import argparse
import base64
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
from pathlib import Path
import secrets

import cv2
import yaml

try:
    from src.edit_roi import (check_period, discover_videos, initial_boxes, parse_boundary,
                              read_records, save_record, validate_boxes)
except ModuleNotFoundError:
    from edit_roi import (check_period, discover_videos, initial_boxes, parse_boundary,
                          read_records, save_record, validate_boxes)


class EditorApp:
    def __init__(self, video_dir, output):
        self.video_dir = Path(video_dir)
        self.output = Path(output)

    def context(self, data):
        device = str(data.get('device', '')).strip()
        if device.isdigit():
            device = f'ANU-25-summer-{int(device)}'
        start = parse_boundary(str(data.get('start', '')))
        end = parse_boundary(str(data.get('end', '')), True)
        if not device or start > end:
            raise ValueError('기기 및 시작·종료 일시를 확인하세요.')
        existing = check_period(read_records(self.output)['regions'], device, start, end)
        videos = discover_videos(self.video_dir, device, start, end)
        return device, start, end, existing, videos

    def dispatch(self, action, data):
        device, start, end, existing, videos = self.context(data)
        if action == 'list':
            return dict(videos=[p.name for p in videos], device=device,
                        start=str(start), end=str(end))
        matches = [p for p in videos if p.name == data.get('video')]
        if not matches:
            raise ValueError('선택한 기기와 기간 안의 영상을 지정하세요.')
        index = data.get('frame', 0)
        if type(index) is not int or index < 0:
            raise ValueError('프레임 번호는 0 이상의 정수여야 합니다.')
        cap = cv2.VideoCapture(str(matches[0]))
        try:
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if not cap.isOpened() or total < 1:
                raise ValueError('영상을 열 수 없습니다.')
            if index >= total:
                raise ValueError('영상의 마지막 프레임을 넘었습니다.')
            if index:
                cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = cap.read()
            if not ok:
                raise ValueError('해당 프레임을 읽을 수 없습니다.')
            height, width = frame.shape[:2]
            if action == 'save':
                if data.get('image_size') != [width, height]:
                    raise ValueError('기준 영상의 해상도가 일치하지 않습니다.')
                roi, entrance = data.get('roi'), data.get('entrance')
                validate_boxes(roi, entrance, width, height)
                record = dict(device=device, start=str(start), end=str(end), roi=roi,
                              entrance=entrance, image_size=[width, height],
                              reference_video=matches[0].name, reference_frame=index)
                save_record(self.output, record)
                return dict(message=f'저장 완료: {self.output}', record=record)
            ok, encoded = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
            if not ok:
                raise ValueError('프레임을 이미지로 변환할 수 없습니다.')
            result = dict(image='data:image/jpeg;base64,' + base64.b64encode(encoded).decode(),
                          width=width, height=height, total=total, frame=index)
            if action == 'open':
                roi, entrance = initial_boxes(device, width, height, existing)
                result.update(roi=list(roi), entrance=list(entrance))
            return result
        finally:
            cap.release()


def make_server(app, port):
    token = secrets.token_urlsafe(32)
    page = Path(__file__).with_name('roi_web.html').read_text(encoding='utf-8').replace('__TOKEN__', token)

    class Handler(BaseHTTPRequestHandler):
        def reply(self, status, body, content_type='application/json; charset=utf-8'):
            if not isinstance(body, bytes):
                body = json.dumps(body, ensure_ascii=False).encode()
            self.send_response(status)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.send_header('X-Frame-Options', 'DENY')
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == '/':
                self.reply(200, page.encode(), 'text/html; charset=utf-8')
            else:
                self.reply(404, {'error': 'Not found'})

        def do_POST(self):
            if self.headers.get('X-Editor-Token') != token:
                self.reply(403, {'error': '페이지를 새로고침한 뒤 다시 시도하세요.'})
                return
            action = self.path.removeprefix('/api/')
            if self.path != '/api/' + action or action not in ('list', 'open', 'frame', 'save'):
                self.reply(404, {'error': 'Not found'})
                return
            try:
                length = int(self.headers.get('Content-Length', '0'))
                if not 0 < length <= 16384:
                    raise ValueError('요청 크기가 올바르지 않습니다.')
                data = json.loads(self.rfile.read(length))
                if not isinstance(data, dict):
                    raise ValueError('잘못된 요청입니다.')
                self.reply(200, app.dispatch(action, data))
            except (ValueError, TypeError, OSError, cv2.error, yaml.YAMLError) as exc:
                self.reply(400, {'error': str(exc)})

    return HTTPServer(('127.0.0.1', port), Handler)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video-dir', type=Path, default=Path('videos'))
    parser.add_argument('--output', type=Path, default=Path('roi_regions.yaml'))
    parser.add_argument('--port', type=int, default=8000)
    args = parser.parse_args(argv)
    with make_server(EditorApp(args.video_dir, args.output), args.port) as server:
        print(f'ROI 편집기: http://127.0.0.1:{server.server_port}', flush=True)
        print(f'로컬 PC에서: ssh -L {server.server_port}:127.0.0.1:{server.server_port} 사용자명@서버주소', flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print('\n편집기를 종료합니다.')


if __name__ == '__main__':
    main()
