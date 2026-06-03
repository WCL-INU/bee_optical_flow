# Validation Data Viewer

`data_viewer.html` is a static browser viewer for checking the relationship between
algorithm output and manually counted truth values. The build step now calculates
the regression models directly from `merged_data.xlsx`, then embeds both the data
and model results into the HTML.

Build or refresh the viewer after updating `merged_data.xlsx`:

```powershell
python validation\build_data_viewer.py
```

Open:

```text
validation/data_viewer.html
```

The viewer uses:

- `validation/merged_data.xlsx` for per-sample output and truth rows.
- The built-in regression calculation in `build_data_viewer.py` for linear and
  flat-exponential prediction models.
- `validation/regression_model_comparison.csv` as a refreshed summary export of
  the models used by the viewer.

Main checks:

- Switch `IN` / `OUT` to inspect each direction independently.
- Switch model type to compare linear vs flat-exponential prediction behavior.
- Filter by `device` and `time` to find systematic error patterns.
- Use `Error percentile` to focus the feature summary on large-error samples.
- Click a scatter point to highlight the same sample in the error table and
  scroll to it.
- `Video date` is taken from the spreadsheet `datetime` column. `Video time` is
  formatted from the `time` code, so `100000` is shown as `10:00:00`.
- Inspect `filtered`, `raw`, and `ratio` columns to see whether the improved
  extraction keeps real movement while suppressing noisy raw flux.

To compare a newly improved algorithm, regenerate `merged_data.xlsx` from the new
results, rerun `build_data_viewer.py`, and reopen or refresh `data_viewer.html`.
The regression model summary is recalculated during the viewer build.
