"""Keyboard-only ROI/entrance editor; records periods without changing analysis settings."""
import argparse
from datetime import datetime, time
import os
from pathlib import Path
import re
import tempfile

import cv2
import yaml


EDGES = [(box, edge) for box in ('roi', 'entrance')
         for edge in ('top', 'bottom', 'left', 'right')]


def parse_boundary(value, end=False):
    value = value.strip()
    try:
        if len(value) == 10:
            day = datetime.strptime(value, '%Y-%m-%d').date()
            return datetime.combine(day, time(23, 59, 59) if end else time())
        result = datetime.fromisoformat(value)
        if result.tzinfo is not None:
            raise ValueError('시간대 없이 촬영 파일명 기준 시각을 입력하세요.')
        return result
    except ValueError as exc:
        raise ValueError(f'일시 형식 오류: {value} (YYYY-MM-DD HH:MM:SS)') from exc


def discover_videos(directory, device, start, end):
    found = []
    for path in directory.glob('*.mp4'):
        match = re.fullmatch(r'(.+)_(\d{8}_\d{6})', path.stem)
        if not match or match[1] != device:
            continue
        stamp = datetime.strptime(match[2], '%Y%m%d_%H%M%S')
        if start <= stamp <= end:
            found.append((stamp, path))
    return [path for _, path in sorted(found)]


def validate_boxes(roi, entrance, width, height):
    for box in (roi, entrance):
        if not isinstance(box, (list, tuple)) or len(box) != 4 or any(type(x) is not int for x in box):
            raise ValueError('좌표는 정수 4개 [x1, y1, x2, y2]여야 합니다.')
        x1, y1, x2, y2 = box
        if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
            raise ValueError('영역은 영상 안에 있어야 하며 너비/높이가 양수여야 합니다.')
    if not (roi[0] <= entrance[0] < entrance[2] <= roi[2] and
            roi[1] <= entrance[1] < entrance[3] <= roi[3]):
        raise ValueError('ENT 영역은 ROI 안에 있어야 합니다.')


class EditorState:
    def __init__(self, roi, entrance, width, height):
        validate_boxes(roi, entrance, width, height)
        self.boxes = {'roi': list(roi), 'entrance': list(entrance)}
        self.width, self.height = width, height
        self.selected = -1

    def cycle(self):
        self.selected = (self.selected + 1) % len(EDGES)

    def move(self, direction, step=1):
        if self.selected < 0:
            return False
        name, edge = EDGES[self.selected]
        if edge in ('top', 'bottom') and direction not in ('up', 'down'):
            return False
        if edge in ('left', 'right') and direction not in ('left', 'right'):
            return False
        index = {'left': 0, 'top': 1, 'right': 2, 'bottom': 3}[edge]
        old = self.boxes[name][index]
        self.boxes[name][index] += step * (-1 if direction in ('up', 'left') else 1)
        try:
            validate_boxes(self.boxes['roi'], self.boxes['entrance'], self.width, self.height)
        except ValueError:
            self.boxes[name][index] = old
            return False
        return True


def read_records(path):
    if not path.exists():
        return {'version': 1, 'regions': []}
    document = yaml.safe_load(path.read_text(encoding='utf-8'))
    if not isinstance(document, dict) or document.get('version') != 1 or not isinstance(document.get('regions'), list):
        raise ValueError('지원하지 않는 YAML 형식입니다 (version: 1, regions 목록 필요).')
    for record in document['regions']:
        if not isinstance(record, dict) or not all(key in record for key in ('device', 'start', 'end', 'roi', 'entrance')):
            raise ValueError('YAML 영역 기록에 필수 항목이 없습니다.')
        a, b = parse_boundary(str(record['start'])), parse_boundary(str(record['end']), True)
        if a > b:
            raise ValueError('YAML에 역전된 기간이 있습니다.')
    return document


def check_period(records, device, start, end):
    exact = None
    for record in records:
        if record['device'] != device:
            continue
        a, b = parse_boundary(str(record['start'])), parse_boundary(str(record['end']), True)
        if start <= b and a <= end:
            if (a, b) != (start, end) or exact is not None:
                raise ValueError(f'기존 기간 {a} ~ {b}와 겹칩니다. 겹치지 않는 기간을 지정하세요. 동일 기간은 다시 편집할 수 있습니다.')
            exact = record
    return exact


class CompactDumper(yaml.SafeDumper):
    pass


def represent_list(dumper, value):
    return dumper.represent_sequence('tag:yaml.org,2002:seq', value,
                                     flow_style=all(type(item) is int for item in value))


CompactDumper.add_representer(list, represent_list)


def save_record(path, record):
    document = read_records(path)
    start, end = parse_boundary(record['start']), parse_boundary(record['end'], True)
    existing = check_period(document['regions'], record['device'], start, end)
    if existing is not None:
        existing.update(record)
    else:
        document['regions'].append(record)
    document['regions'].sort(key=lambda r: (r['device'], str(r['start'])))
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent, delete=False) as stream:
            temp_path = Path(stream.name)
            yaml.dump(document, stream, Dumper=CompactDumper, sort_keys=False, allow_unicode=True)
            stream.flush()
            os.fsync(stream.fileno())
        temp_path.replace(path)
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()


def choose_video(videos):
    page = 0
    while True:
        for index in range(page * 20, min((page + 1) * 20, len(videos))):
            print(f'{index + 1:5d}  {videos[index].name}')
        choice = input(f'영상 번호 (총 {len(videos)}개), 다음 목록 n / 이전 목록 b: ').strip()
        if choice == 'n':
            page = min(page + 1, (len(videos) - 1) // 20)
        elif choice == 'b':
            page = max(0, page - 1)
        elif choice.isdigit() and 1 <= int(choice) <= len(videos):
            return videos[int(choice) - 1]
        else:
            print('목록의 영상 번호를 입력하세요.')


def initial_boxes(device, width, height, existing):
    if existing:
        roi, ent = existing['roi'], existing['entrance']
        if existing.get('image_size', [width, height]) != [width, height]:
            raise ValueError('저장된 기준 영상과 해상도가 다릅니다. 같은 해상도의 영상을 선택하세요.')
        validate_boxes(roi, ent, width, height)
        return roi, ent
    try:
        from src.main import COORDINATE_PRESETS
    except ModuleNotFoundError:
        from main import COORDINATE_PRESETS
    for preset in COORDINATE_PRESETS.values():
        if preset['video_key'] == device:
            try:
                validate_boxes(preset['roi'], preset['entrance'], width, height)
                return preset['roi'], preset['entrance']
            except ValueError:
                break
    return [0, height // 2, width, height], [width // 4, height * 5 // 8, width * 3 // 4, height * 7 // 8]


def edit_frame(video, device, start, end, existing, frame_step=1):
    # Import pyplot only when opening the editor (keeps CLI/storage usable headlessly).
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    cap = cv2.VideoCapture(str(video))
    try:
        ok, frame = cap.read()
        if not ok:
            raise ValueError(f'영상을 읽을 수 없습니다: {video}')
        height, width = frame.shape[:2]
        state = EditorState(*initial_boxes(device, width, height, existing), width, height)
        total = max(1, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        frame_index = 0
        accepted = False
        fig, ax = plt.subplots(figsize=(13, 9))
        fig.canvas.mpl_disconnect(fig.canvas.manager.key_press_handler_id)
        image = ax.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        patches = {}
        for name, color in [('roi', 'lime'), ('entrance', 'cyan')]:
            patch = Rectangle((0, 0), 1, 1, fill=False, edgecolor=color, linewidth=1.8)
            ax.add_patch(patch)
            patches[name] = patch
        selected_line, = ax.plot([], [], color='yellow', linewidth=3)
        ax.set_axis_off()
        fig.subplots_adjust(top=.85, bottom=.12)
        footer = fig.text(.02, .02, '', fontsize=10)

        def redraw(message=''):
            for name, patch in patches.items():
                x1, y1, x2, y2 = state.boxes[name]
                patch.set_bounds(x1, y1, x2 - x1, y2 - y1)
            label = 'view'
            if state.selected >= 0:
                name, edge = EDGES[state.selected]
                x1, y1, x2, y2 = state.boxes[name]
                points = {'top': ([x1, x2], [y1, y1]), 'bottom': ([x1, x2], [y2, y2]),
                          'left': ([x1, x1], [y1, y2]), 'right': ([x2, x2], [y1, y2])}
                selected_line.set_data(*points[edge])
                label = f'{name} {edge}'
            ax.set_title(f'{video.name}\nPeriod: {start} ~ {end}\nFrame {frame_index}/{total - 1} | {label}')
            footer.set_text('Space: select edge | arrows: 1px | Shift+arrows: 10px | n/b: next/previous frame\n'
                            'Enter: finish and save | Esc: cancel | ROI: green | ENT: cyan | selected: yellow\n'
                            f"ROI {state.boxes['roi']}    ENT {state.boxes['entrance']}    {message}")
            fig.canvas.draw_idle()

        def on_key(event):
            nonlocal frame_index, accepted
            key = event.key or ''
            message = ''
            if key == ' ':
                state.cycle()
            elif key == 'enter':
                accepted = True
                plt.close(fig)
                return
            elif key == 'escape':
                plt.close(fig)
                return
            elif key in ('n', 'b'):
                target = max(0, min(total - 1, frame_index + (frame_step if key == 'n' else -frame_step)))
                if target != frame_index:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, target)
                    success, next_frame = cap.read()
                    if success:
                        frame_index = target
                        image.set_data(cv2.cvtColor(next_frame, cv2.COLOR_BGR2RGB))
                    else:
                        message = 'Cannot read requested frame'
            elif key.removeprefix('shift+') in ('up', 'down', 'left', 'right'):
                if not state.move(key.removeprefix('shift+'), 10 if key.startswith('shift+') else 1):
                    message = 'Select an edge / perpendicular movement only / rectangle boundary reached'
            redraw(message)

        fig.canvas.mpl_connect('key_press_event', on_key)
        redraw()
        plt.show()
        if accepted:
            return dict(device=device, start=start.isoformat(sep=' '), end=end.isoformat(sep=' '),
                        **state.boxes, image_size=[width, height], reference_video=video.name,
                        reference_frame=frame_index)
        return None
    finally:
        cap.release()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video-dir', type=Path, default=Path('videos'))
    parser.add_argument('--device', help='기기명 또는 번호 (예: 8)')
    parser.add_argument('--start', help='YYYY-MM-DD 또는 YYYY-MM-DD HH:MM:SS (포함)')
    parser.add_argument('--end', help='YYYY-MM-DD 또는 YYYY-MM-DD HH:MM:SS (포함)')
    parser.add_argument('--video', help='기간 안에서 편집할 대표 영상 파일명 (생략하면 목록 선택)')
    parser.add_argument('--output', type=Path, default=Path('roi_regions.yaml'))
    parser.add_argument('--frame-step', type=int, default=1, help='n/b 이동 프레임 수 (기본 1)')
    args = parser.parse_args(argv)
    try:
        if args.frame_step < 1:
            raise ValueError('--frame-step은 양수여야 합니다.')
        device = (args.device or input('기기명 또는 번호: ')).strip()
        if device.isdigit():
            device = f'ANU-25-summer-{int(device)}'
        start = parse_boundary(args.start or input('적용 시작 일시 (YYYY-MM-DD [HH:MM:SS]): '))
        end = parse_boundary(args.end or input('적용 종료 일시 (YYYY-MM-DD [HH:MM:SS]): '), True)
        if start > end:
            raise ValueError('종료 일시는 시작 일시보다 빠를 수 없습니다.')
        existing = check_period(read_records(args.output)['regions'], device, start, end)
        videos = discover_videos(args.video_dir, device, start, end)
        if not videos:
            raise ValueError('해당 기기/기간의 영상이 없습니다.')
        if args.video:
            matches = [p for p in videos if p.name == args.video]
            if not matches:
                raise ValueError('--video는 선택한 기기/기간 안의 영상 파일명이어야 합니다.')
            video = matches[0]
        else:
            video = choose_video(videos)
        print(f'적용: {device}, {start} ~ {end} ({len(videos)}개 영상). 대표 영상: {video.name}')
        print('Space: 경계 선택, 방향키: 이동, Shift+방향키: 10px, n/b: 프레임 이동, Enter: 저장, Esc: 취소')
        record = edit_frame(video, device, start, end, existing, args.frame_step)
        if record:
            save_record(args.output, record)
            print(f'저장 완료: {args.output} (분석 코드 반영은 별도)')
        else:
            print('취소: 저장하지 않았습니다.')
        return 0
    except (ValueError, OSError, yaml.YAMLError) as exc:
        parser.error(str(exc))
    except (KeyboardInterrupt, EOFError):
        print('\n취소: 저장하지 않았습니다.')
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
