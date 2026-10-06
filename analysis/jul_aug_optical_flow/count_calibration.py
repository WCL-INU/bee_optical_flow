"""Apply the saved March/April directional linear models to July/August flux."""
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
MODEL_SOURCE = ROOT / 'validation/output/regression_model_comparison.csv'
TRAINING_SOURCE = ROOT / 'validation/data/merged_data.xlsx'
BATCH_SOURCE = ROOT / 'bee_count_output/yaml_jul_aug_all/batch_summary.csv'
COUNT_COLUMNS = [f'linear_predicted_{direction}_count' for direction in ['in', 'out', 'traffic']]
RATE_COLUMNS = [f'{column}_per_min' for column in COUNT_COLUMNS]
DERIVED_COLUMNS = COUNT_COLUMNS + RATE_COLUMNS
FIGURE_NAME = '13_linear_predicted_count_hourly_comparison.png'


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_models():
    """Read saved coefficients and verify that they match the spring training data."""
    source = pd.read_csv(MODEL_SOURCE, float_precision='round_trip')
    training = pd.read_excel(TRAINING_SOURCE)
    dates = pd.to_datetime(training.datetime, errors='raise')
    assert set(dates.dt.year) == {2026} and set(dates.dt.month) == {3, 4}
    models = {}
    for direction in ['in', 'out']:
        flux = f'total_filtered_{direction}_flux'
        rows = source[(source.model == 'linear') & (source.y_col == direction) & (source.x_col == flux)]
        assert len(rows) == 1 and rows.iloc[0].status == 'ok'
        row = rows.iloc[0]
        pairs = training[[flux, direction]].apply(pd.to_numeric, errors='coerce').replace(
            [np.inf, -np.inf], np.nan).dropna()
        assert len(pairs) == int(row.n) == int(row.fit_n)
        x = pairs[flux].to_numpy()
        y = pairs[direction].to_numpy()
        # Verify the stored model; calculations below use its full saved precision.
        slope = np.sum((x - x.mean()) * (y - y.mean())) / np.sum((x - x.mean()) ** 2)
        intercept = y.mean() - slope * x.mean()
        np.testing.assert_allclose([row.slope, row.intercept], [slope, intercept], rtol=1e-12, atol=1e-12)
        residual = y - (float(row.slope) * x + float(row.intercept))
        np.testing.assert_allclose([row.mae, row.rmse], [np.abs(residual).mean(), np.sqrt((residual**2).mean())],
                                   rtol=1e-12, atol=1e-12)
        models[direction] = dict(x_col=flux, y_col=direction, n=int(row.n),
                                 slope=float(row.slope), intercept=float(row.intercept),
                                 r_squared=float(row.r_squared), mae=float(row.mae), rmse=float(row.rmse))
    return dict(model_source=MODEL_SOURCE.relative_to(ROOT).as_posix(),
                model_source_sha256=sha256(MODEL_SOURCE),
                training_source=TRAINING_SOURCE.relative_to(ROOT).as_posix(),
                training_source_sha256=sha256(TRAINING_SOURCE),
                training_rows=len(training), training_start=dates.min().isoformat(sep=' '),
                training_end=dates.max().isoformat(sep=' '),
                training_month_rows={str(int(month)): int(n) for month, n in dates.dt.month.value_counts().sort_index().items()},
                models=models, count_unit='model-predicted directional count per analyzed video',
                rate_unit='model-predicted directional count per minute of analyzed video',
                intercept_policy='Apply once per direction per video; preserve the saved intercept.',
                rounding_policy='No rounding or clipping; no area normalization.')


def apply_models(batch, metadata):
    """Append predictions without changing any original batch values."""
    result = batch.copy()
    assert np.isfinite(result.duration_sec).all() and result.duration_sec.gt(0).all()
    for direction in ['in', 'out']:
        model = metadata['models'][direction]
        flux = result[model['x_col']]
        assert np.isfinite(flux).all() and flux.ge(0).all()
        result[f'linear_predicted_{direction}_count'] = model['slope'] * flux + model['intercept']
    result['linear_predicted_traffic_count'] = result[COUNT_COLUMNS[:2]].sum(axis=1)
    for count, rate in zip(COUNT_COLUMNS, RATE_COLUMNS):
        result[rate] = result[count] / result.duration_sec * 60.0
    assert np.isfinite(result[DERIVED_COLUMNS].to_numpy()).all()
    pd.testing.assert_frame_equal(result[batch.columns], batch)
    return result


def aggregate(data, keys):
    rows = []
    grouped = [((), data)] if not keys else data.groupby(keys, sort=True)
    for key, group in grouped:
        if not isinstance(key, tuple):
            key = (key,)
        row = dict(zip(keys, key))
        row.update(n=len(group), duration_sec=float(group.duration_sec.sum()))
        for direction in ['in', 'out', 'traffic']:
            count = f'linear_predicted_{direction}_count'
            rate = f'{count}_per_min'
            row[f'total_{direction}_count'] = float(group[count].sum())
            row[f'mean_{direction}_count_per_video'] = float(group[count].mean())
            row[f'mean_{direction}_count_per_min'] = float(group[rate].mean())
            row[f'pooled_{direction}_count_per_min'] = float(group[count].sum() / group.duration_sec.sum() * 60)
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    metadata = load_models()
    batch = pd.read_csv(BATCH_SOURCE, float_precision='round_trip')
    predicted = apply_models(batch, metadata)
    parsed = predicted.video.str.extract(r'^ANU-25-summer-(\d+)_(\d{8})_(\d{6})\.mp4$')
    assert not parsed.isna().any().any() and not predicted.video.duplicated().any()
    predicted['device'] = parsed[0].astype(int)
    predicted['date'] = pd.to_datetime(parsed[1], format='%Y%m%d')
    predicted['month'] = predicted.date.dt.month
    predicted['hour'] = parsed[2].str[:2].astype(int)
    assert set(predicted.month) == {7, 8}
    video_columns = ['video', 'device', 'date', 'month', 'hour', 'duration_sec',
                     'total_filtered_in_flux', 'total_filtered_out_flux'] + DERIVED_COLUMNS
    tables = HERE / 'tables'
    predicted[video_columns].to_csv(tables / 'linear_predicted_video_counts.csv', index=False)
    month = aggregate(predicted, ['month'])
    device = aggregate(predicted, ['device'])
    device_month = aggregate(predicted, ['device', 'month'])
    for name, table in [('month', month), ('device', device), ('device_month', device_month)]:
        table.to_csv(tables / f'linear_predicted_{name}_counts.csv', index=False)
    per_cell = predicted.groupby(['device', 'hour', 'month'])[RATE_COLUMNS].mean()
    presence = per_cell.unstack('month')
    shared_index = presence.dropna().index
    shared = per_cell.reset_index().set_index(['device', 'hour']).loc[shared_index].reset_index()
    assert len(shared_index) == 323 and len(shared_index.get_level_values('device').unique()) == 19
    shared.to_csv(tables / 'linear_predicted_matched_device_hour_counts.csv', index=False)
    hourly = shared.groupby(['hour', 'month'])[RATE_COLUMNS].mean().reset_index()
    hourly.to_csv(tables / 'linear_predicted_matched_hourly_counts.csv', index=False)
    shared_month = shared.groupby('month')[RATE_COLUMNS].mean()
    matched = {str(int(month)): {direction: float(row[f'linear_predicted_{direction}_count_per_min'])
                               for direction in ['in', 'out', 'traffic']}
               for month, row in shared_month.iterrows()}
    changes = {direction: matched['8'][direction] / matched['7'][direction] - 1
               for direction in ['in', 'out', 'traffic']}
    checks = {}
    for direction in ['in', 'out']:
        model = metadata['models'][direction]
        count = f'linear_predicted_{direction}_count'
        algebraic_total = model['slope'] * batch[model['x_col']].sum() + len(batch) * model['intercept']
        checks[f'{direction}_aggregate_abs_error'] = float(abs(predicted[count].sum() - algebraic_total))
        np.testing.assert_allclose(predicted[count].sum(), algebraic_total, rtol=1e-12, atol=1e-8)
        zero = batch[model['x_col']].eq(0)
        assert predicted.loc[zero, count].eq(model['intercept']).all()
        checks[f'{direction}_zero_flux_videos'] = int(zero.sum())
    np.testing.assert_allclose(predicted[COUNT_COLUMNS[2]], predicted[COUNT_COLUMNS[0]] + predicted[COUNT_COLUMNS[1]], rtol=0, atol=0)
    summary = dict(metadata=metadata, source_csv=BATCH_SOURCE.relative_to(ROOT).as_posix(),
                   source_csv_sha256=sha256(BATCH_SOURCE), video_rows=len(batch), derived_columns=DERIVED_COLUMNS,
                   overall=aggregate(predicted, []).iloc[0].to_dict(), monthly=month.to_dict('records'),
                   matched_devices=19, matched_cells=len(shared_index), matched_monthly_count_per_min=matched,
                   matched_relative_changes=changes, checks=checks, figure=FIGURE_NAME)
    (HERE / 'linear_count_calibration.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    font = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
    font_manager.fontManager.addfont(font)
    plt.rcParams.update({'font.family': font_manager.FontProperties(fname=font).get_name(), 'axes.unicode_minus': False})
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.6), sharex=True)
    titles = ['IN 추정', 'OUT 추정', 'IN + OUT 추정']
    for ax, direction, title in zip(axes, ['in', 'out', 'traffic'], titles):
        for month, color in [(7, '#247BA0'), (8, '#E59E36')]:
            curve = hourly[hourly.month == month]
            ax.plot(curve.hour, curve[f'linear_predicted_{direction}_count_per_min'], 'o-',
                    label=f'{month}월', color=color, markersize=4, linewidth=2)
        ax.set(title=title, xlabel='파일명 촬영 시각', ylabel='선형 모델 추정 마리수 / 분', ylim=(0, None))
        ax.set_xticks(range(5, 22, 2))
        ax.grid(axis='y', alpha=.2)
        ax.spines[['top', 'right']].set_visible(False)
        ax.legend(frameon=False)
    fig.suptitle('공통 기기·시각의 회귀 추정 출입 곡선 — 7·8월', fontsize=18, y=.98)
    fig.text(.5, .085, '3·4월 3,822건의 기존 방향별 선형 모델 적용 · 공통 19개 기기 × 17개 시각 · 기기별 동일 비중',
             ha='center', fontsize=10, color='#52616B')
    fig.text(.5, .035, '영상별 추정치 = 기울기 × filtered 방향별 누적 flux + 절편; 분당 추정치 = 영상별 추정치 / 처리 초 × 60',
             ha='center', fontsize=9, color='#52616B')
    fig.tight_layout(rect=(0, .13, 1, .93))
    fig.savefig(HERE / 'figures' / FIGURE_NAME, dpi=200, facecolor='white')
    plt.close(fig)
    print(json.dumps({'rows': len(batch), 'monthly': summary['monthly'], 'matched_relative_changes': changes,
                      'figure': FIGURE_NAME}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
