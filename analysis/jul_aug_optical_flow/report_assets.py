"""Copy report references and create file indexes for static web readers."""
from collections import deque
import hashlib
import html
import json
import re
import shutil
from pathlib import Path
from urllib.parse import quote, urlsplit

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
REFERENCES = HERE / 'references'
RAW_RESULTS = ROOT / 'bee_count_output/yaml_jul_aug_all'
LINK = re.compile(r'\]\(([^)]+)\)')


def prepare_report_assets(report):
    """Return Markdown whose local links only target the shared analysis folder."""
    copied = {}
    queue = deque()

    def copy_reference(source):
        source = source.resolve()
        assert source.is_file(), source
        relative = source.relative_to(ROOT)
        destination = HERE / source.name if source.suffix == '.xlsx' else REFERENCES / relative
        if relative.as_posix() not in copied:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            copied[relative.as_posix()] = {
                'source': relative.as_posix(),
                'copy': destination.relative_to(ROOT).as_posix(),
                'bytes': source.stat().st_size,
                'sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
            }
            if source.suffix == '.md':
                queue.append(source)
        return destination

    # Share the compact source summaries underlying the inventory audit.
    for name in ['batch_summary.csv', 'skipped_videos.csv']:
        copy_reference(RAW_RESULTS / name)
    copy_reference(ROOT / 'roi_regions.yaml')
    # The earlier feature report names its supporting files in inline code.
    for source in sorted((ROOT / 'analysis/video_features/final').rglob('*')):
        if source.is_file():
            copy_reference(source)

    def shared_link(match):
        target = match[1]
        parsed = urlsplit(target)
        if parsed.scheme or parsed.netloc or target.startswith('#'):
            return match[0]
        source = ROOT / parsed.path
        if source == HERE / 'report_reference_manifest.json':
            destination = source
        elif source == RAW_RESULTS:
            destination = REFERENCES / source.relative_to(ROOT) / 'index.html'
        elif source.is_relative_to(HERE):
            assert source.exists(), source
            destination = source / 'index.html' if source.is_dir() else source
        else:
            assert source.is_file(), source
            destination = copy_reference(source)
        suffix = f'#{parsed.fragment}' if parsed.fragment else ''
        return '](' + destination.relative_to(ROOT).as_posix() + suffix + ')'

    report = LINK.sub(shared_link, report)
    # Preserve relative image/document links inside copied Markdown references.
    while queue:
        source = queue.popleft()
        for target in LINK.findall(source.read_text(encoding='utf-8')):
            parsed = urlsplit(target)
            if parsed.scheme or parsed.netloc or target.startswith('#'):
                continue
            linked = (source.parent / parsed.path).resolve()
            assert linked.is_file(), linked
            copy_reference(linked)

    manifest = HERE / 'report_reference_manifest.json'
    manifest.write_text(json.dumps({'files': list(copied.values())}, ensure_ascii=False, indent=2) + '\n',
                        encoding='utf-8')

    # Directory URLs are not portable to static hosting; use explicit HTML indexes.
    directories = [HERE] + sorted(path for path in HERE.rglob('*')
                                 if path.is_dir() and path.name != '__pycache__')
    for directory in directories:
        entries = []
        if directory != HERE:
            entries.append('<li><a href="../index.html">상위 파일 목록</a></li>')
        else:
            entries.append('<li><a href="../../optical_flow_jul_aug_analysis_report.html">분석 보고서 (HTML)</a></li>')
        for path in sorted(directory.iterdir(), key=lambda path: (not path.is_dir(), path.name)):
            if path.name in ['index.html', '__pycache__'] or path.is_symlink():
                continue
            target = path.name + '/index.html' if path.is_dir() else path.name
            size = '' if path.is_dir() else f' <small>({path.stat().st_size:,} bytes)</small>'
            entries.append(f'<li><a href="{quote(target)}">{html.escape(path.name)}</a>{size}</li>')
        title = '7–8월 Optical flow 보고서 자료: ' + directory.relative_to(HERE).as_posix()
        document = ('<!doctype html><html lang="ko"><meta charset="utf-8">'
                    '<meta name="viewport" content="width=device-width, initial-scale=1">'
                    f'<title>{html.escape(title)}</title>'
                    '<style>body{max-width:1100px;margin:32px auto;padding:0 20px;font-family:sans-serif;'
                    'line-height:1.8}a{color:#166b94}small{color:#666}</style>'
                    f'<h1>{html.escape(title)}</h1><ul>' + ''.join(entries) + '</ul></html>')
        (directory / 'index.html').write_text(document, encoding='utf-8')
    return report
