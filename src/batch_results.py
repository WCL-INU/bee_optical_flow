"""Shared batch resume validation and per-video completion records."""
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import tempfile

import cv2
import numpy as np
import pandas as pd

try:
    from src.bee_entrance_count import (FARNEBACK_PARAMS, MAX_PROCESS_FRAMES, aggregate_window_counts,
                                      clamp_rect, load_video_info, summarize_result)
except ModuleNotFoundError:
    from bee_entrance_count import (FARNEBACK_PARAMS, MAX_PROCESS_FRAMES, aggregate_window_counts,
                                  clamp_rect, load_video_info, summarize_result)


def paths_for(video, output):
    base = Path(output) / Path(video).stem
    return {key: Path(str(base) + suffix) for key, suffix in {
        'preview': '_preview.mp4', 'frame': '_frame_flux.csv', 'window': '_window_3sec.csv',
        'completion': '_completion.json',
    }.items()}


def signature(video, config):
    stat = Path(video).stat()
    return dict(version=1, source=str(Path(video).resolve()), size=stat.st_size,
                mtime_ns=stat.st_mtime_ns, config=asdict(config),
                farneback=FARNEBACK_PARAMS, max_frames=MAX_PROCESS_FRAMES)


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_marker(path, document):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = None
    try:
        with tempfile.NamedTemporaryFile('w', dir=path.parent, encoding='utf-8', delete=False) as stream:
            temp = Path(stream.name)
            json.dump(document, stream, ensure_ascii=False)
            stream.flush()
            os.fsync(stream.fileno())
        temp.replace(path)
    finally:
        if temp is not None and temp.exists():
            temp.unlink()


def validate_files(video, output, config):
    paths = paths_for(video, output)
    if not all(paths[key].is_file() and paths[key].stat().st_size > 0 for key in ('preview', 'frame', 'window')):
        raise ValueError('산출물 누락 또는 빈 파일')
    cap, info = load_video_info(video)
    cap.release()
    count = min(info['frame_count'], MAX_PROCESS_FRAMES) - 1
    if count < 1:
        raise ValueError('원본의 예상 처리 프레임 수를 확인할 수 없음')
    frames = pd.read_csv(paths['frame'])
    required = ['frame', 'time_sec', 'raw_in_flux', 'raw_out_flux', 'filtered_in_flux',
                'filtered_out_flux', 'raw_candidate_pixels', 'persistent_candidate_pixels',
                'filtered_candidate_pixels', 'persistence_mean', 'persistence_max']
    if not set(required) <= set(frames.columns) or len(frames) != count:
        raise ValueError('프레임 CSV 열 또는 행 수 불일치')
    if not np.isfinite(frames.to_numpy(dtype=float)).all():
        raise ValueError('프레임 CSV에 비정상 값 존재')
    if not np.array_equal(frames['frame'], np.arange(1, count + 1)) or not np.allclose(frames['time_sec'], np.arange(1, count + 1) / info['fps']):
        raise ValueError('프레임 번호 또는 시각 불일치')
    windows = pd.read_csv(paths['window'])
    expected = aggregate_window_counts(frames, config)
    if list(windows.columns) != list(expected.columns) or windows.shape != expected.shape or not np.allclose(windows.to_numpy(dtype=float), expected.to_numpy(dtype=float), rtol=1e-6, atol=1e-8):
        raise ValueError('구간 CSV와 프레임 CSV 집계 불일치')
    roi = clamp_rect((config.roi_x1, config.roi_y1, config.roi_x2, config.roi_y2), info['width'], info['height'], 'ROI')
    ent = (config.ent_x1 - roi[0], config.ent_y1 - roi[1], config.ent_x2 - roi[0], config.ent_y2 - roi[1])
    preview_count = count // max(1, config.preview_stride)
    if preview_count == 0:
        # Very large preview_stride intentionally emits no frames (e.g. tune mode).
        return summarize_result(video, output, config, frames, windows, roi, ent)
    preview = cv2.VideoCapture(str(paths['preview']))
    try:
        if not preview.isOpened() or int(preview.get(cv2.CAP_PROP_FRAME_COUNT)) != preview_count or preview_count < 1:
            raise ValueError('프리뷰 프레임 수 불일치 또는 읽기 실패')
        # mp4v may truncate odd dimensions by one pixel.
        dimensions = ((roi[2] - roi[0] + max(1, int(config.preview_panel_width))) // 2 * 2,
                      (roi[3] - roi[1]) // 2 * 2)
        if (int(preview.get(cv2.CAP_PROP_FRAME_WIDTH)), int(preview.get(cv2.CAP_PROP_FRAME_HEIGHT))) != dimensions:
            raise ValueError('프리뷰 해상도 불일치')
        fps = max(1., info['fps'] / max(1, config.preview_stride))
        if not np.isclose(preview.get(cv2.CAP_PROP_FPS), fps, rtol=.01):
            raise ValueError('프리뷰 FPS 불일치')
        if not preview.read()[0]:
            raise ValueError('프리뷰 첫 프레임 읽기 실패')
        if preview_count > 1:
            preview.set(cv2.CAP_PROP_POS_FRAMES, preview_count - 1)
            if not preview.read()[0]:
                raise ValueError('프리뷰 마지막 프레임 읽기 실패')
    finally:
        preview.release()
    return summarize_result(video, output, config, frames, windows, roi, ent)


def reusable_result(video, output, config):
    paths = paths_for(video, output)
    try:
        marker = None
        if paths['completion'].exists():
            marker = json.loads(paths['completion'].read_text(encoding='utf-8'))
            if not isinstance(marker, dict) or not isinstance(marker.get('summary', {}), dict):
                raise ValueError('완료 기록 형식 오류')
            if marker.get('state') != 'complete':
                raise ValueError('이전 작업이 완료되지 않음')
            if marker.get('signature') != signature(video, config):
                raise ValueError('입력 영상 또는 연산 설정 변경')
            if marker.get('hashes') != {key: digest(paths[key]) for key in ('preview', 'frame', 'window')}:
                raise ValueError('완료 기록과 산출물 내용 불일치')
        summary = validate_files(video, output, config)
        if marker:
            summary = marker['summary']
            return dict(summary, result_status='reused_verified'), '완료 기록 및 파일 검사 통과'
        # Legacy CSVs do not prove which coordinates/settings produced the flux.
        for key in list(summary):
            if key.startswith(('roi_', 'ent_', 'entrance_roi_')) or key in asdict(config):
                summary[key] = None
        return dict(summary, result_status='reused_legacy_unverified'), '기존 파일 검사 통과 (원래 좌표·설정 확인 불가; 현재 설정으로 집계 검증)'
    except (OSError, ValueError, TypeError, KeyError, RuntimeError, cv2.error) as exc:
        return None, str(exc)


def process_or_reuse(video, output, config, process, force=False):
    if not force:
        reused, reason = reusable_result(video, output, config)
        if reused is not None:
            print(f'[REUSE] {Path(video).name}: {reason}', flush=True)
            return reused
        print(f'[RUN] {Path(video).name}: {reason}', flush=True)
    paths = paths_for(video, output)
    identity = signature(video, config)
    # Write before touching any result, so interrupted replacements never look legacy-complete.
    write_marker(paths['completion'], dict(state='running', signature=identity))
    result, _, _ = process(video, output, config)
    try:
        validate_files(video, output, config)
        if signature(video, config) != identity:
            raise ValueError('처리 도중 입력 영상 변경')
    except (OSError, ValueError, RuntimeError, cv2.error) as exc:
        raise RuntimeError(f'{Path(video).name}: 완료 검사 실패: {exc}') from exc
    result = dict(result, result_status='processed')
    write_marker(paths['completion'], dict(state='complete', signature=identity, summary=result,
                                          hashes={key: digest(paths[key]) for key in ('preview', 'frame', 'window')}))
    return result
