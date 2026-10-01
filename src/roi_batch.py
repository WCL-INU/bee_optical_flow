"""Resolve editor YAML periods against original recording filenames."""
from datetime import datetime
import csv
from pathlib import Path
import re
import sys

import cv2

try:
    from src.edit_roi import parse_boundary, read_records, validate_boxes
except ModuleNotFoundError:
    from edit_roi import parse_boundary, read_records, validate_boxes


def video_identity(path):
    match = re.fullmatch(r'(.+)_(\d{8}_\d{6})', Path(path).stem)
    if not match:
        raise ValueError(f'촬영 일시를 읽을 수 없는 파일명: {path}')
    return match[1], datetime.strptime(match[2], '%Y%m%d_%H%M%S')


def normalize_device(value):
    return f'ANU-25-summer-{int(value)}' if value.isdigit() else value


def filter_recordings(videos, devices, start, end):
    selected = []
    for video in videos:
        try:
            device, stamp = video_identity(video)
        except ValueError:
            continue
        if (not devices or device in devices) and (start is None or start <= stamp) and (end is None or stamp <= end):
            selected.append((stamp, device, video))
    return [video for _, _, video in sorted(selected)]


def resolve_regions(path, videos, *, inspect_videos=True):
    path = Path(path)
    if not path.is_file():
        raise ValueError(f'ROI YAML 파일을 찾을 수 없습니다: {path}')
    records = read_records(path)['regions']
    periods = []
    for record in records:
        periods.append((record, parse_boundary(str(record['start'])), parse_boundary(str(record['end']), True)))
    resolved = {}
    # Validate every selected recording before starting expensive optical flow.
    for video in videos:
        device, stamp = video_identity(video)
        matches = [record for record, start, end in periods if record['device'] == device and start <= stamp <= end]
        if not matches:
            print(f'[SKIP] {video.name}: YAML에 해당 기기·촬영 일시의 좌표 정보가 없습니다. ({device}, {stamp})')
            continue
        if len(matches) != 1:
            raise ValueError(f'{video.name}: 일치하는 YAML 영역이 {len(matches)}개입니다. 각 영상에 정확히 하나의 기간이 필요합니다.')
        record = matches[0]
        if not inspect_videos:
            # Check YAML geometry without opening thousands of video containers.
            size = record.get('image_size', [sys.maxsize, sys.maxsize])
            if not isinstance(size, list) or len(size) != 2 or any(type(value) is not int or value < 1 for value in size):
                raise ValueError(f'{video.name}: YAML image_size는 양의 정수 [너비, 높이]여야 합니다.')
            validate_boxes(record['roi'], record['entrance'], *size)
            resolved[str(video)] = record
            continue
        cap = cv2.VideoCapture(str(video))
        try:
            width, height = (int(cap.get(prop)) for prop in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT))
            if not cap.isOpened() or width < 1 or height < 1:
                raise ValueError(f'영상 정보를 읽을 수 없습니다: {video}')
        finally:
            cap.release()
        if 'image_size' in record and record['image_size'] != [width, height]:
            raise ValueError(f'{video.name}: YAML image_size {record["image_size"]}와 영상 해상도 {[width, height]}가 다릅니다.')
        validate_boxes(record['roi'], record['entrance'], width, height)
        resolved[str(video)] = record
    return resolved


def save_skipped_report(output_dir, videos, resolved, yaml_path):
    """Persist this run's skipped recordings before optical-flow processing starts."""
    report = Path(output_dir) / 'skipped_videos.csv'
    report.parent.mkdir(parents=True, exist_ok=True)
    with report.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=[
            'video', 'video_path', 'device', 'recorded_at', 'roi_yaml', 'reason',
        ])
        writer.writeheader()
        for video in videos:
            if str(video) in resolved:
                continue
            device, stamp = video_identity(video)
            writer.writerow(dict(video=video.name, video_path=str(video.resolve()),
                                 device=device, recorded_at=str(stamp),
                                 roi_yaml=str(Path(yaml_path).resolve()),
                                 reason='YAML에 해당 기기·촬영 일시의 좌표 정보가 없습니다.'))
    return report
