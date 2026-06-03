import argparse
import csv
import json
import math
from datetime import datetime, timedelta
from pathlib import Path
from zipfile import ZipFile
from xml.etree import ElementTree as ET


DEFAULT_INPUT = Path("validation/merged_data.xlsx")
DEFAULT_OUTPUT = Path("validation/data_viewer.html")
DEFAULT_MODELS_OUTPUT = Path("validation/regression_model_comparison.csv")

EXP_FLUX_SCALE = 1_000_000.0
FLAT_EXP_MIN_LOG_SLOPE = 0.001
FLAT_EXP_MAX_LOG_SLOPE = 0.05
FLAT_EXP_GRID_SIZE = 600
MODEL_JOBS = [
    {
        "label": "in count model comparison",
        "x_col": "total_filtered_in_flux",
        "y_col": "in",
    },
    {
        "label": "out count model comparison",
        "x_col": "total_filtered_out_flux",
        "y_col": "out",
    },
]

MAIN_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
REL_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
PKG_REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"


def column_index(cell_ref):
    letters = "".join(ch for ch in cell_ref if ch.isalpha())
    index = 0
    for ch in letters.upper():
        index = index * 26 + ord(ch) - 64
    return index - 1


def convert_value(value):
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return value
    if math.isfinite(number) and number.is_integer():
        return int(number)
    return number


def excel_date_to_iso(value):
    if not isinstance(value, (int, float)):
        return value
    # Excel's Windows epoch includes a historical leap-year bug; this offset
    # matches common spreadsheet readers for modern dates.
    date = datetime(1899, 12, 30) + timedelta(days=float(value))
    return date.isoformat(timespec="seconds")


def read_shared_strings(zip_file):
    if "xl/sharedStrings.xml" not in zip_file.namelist():
        return []
    root = ET.fromstring(zip_file.read("xl/sharedStrings.xml"))
    strings = []
    for item in root.findall(f"{{{MAIN_NS}}}si"):
        strings.append("".join(node.text or "" for node in item.findall(f".//{{{MAIN_NS}}}t")))
    return strings


def workbook_first_sheet_path(zip_file):
    workbook = ET.fromstring(zip_file.read("xl/workbook.xml"))
    first_sheet = workbook.find(f".//{{{MAIN_NS}}}sheet")
    if first_sheet is None:
        return "xl/worksheets/sheet1.xml"

    rel_id = first_sheet.attrib.get(f"{{{REL_NS}}}id")
    if not rel_id or "xl/_rels/workbook.xml.rels" not in zip_file.namelist():
        return "xl/worksheets/sheet1.xml"

    rels = ET.fromstring(zip_file.read("xl/_rels/workbook.xml.rels"))
    for rel in rels.findall(f"{{{PKG_REL_NS}}}Relationship"):
        if rel.attrib.get("Id") == rel_id:
            target = rel.attrib["Target"]
            return f"xl/{target}" if not target.startswith("xl/") else target
    return "xl/worksheets/sheet1.xml"


def read_xlsx_rows(path):
    with ZipFile(path) as zip_file:
        shared_strings = read_shared_strings(zip_file)
        sheet_path = workbook_first_sheet_path(zip_file)
        sheet = ET.fromstring(zip_file.read(sheet_path))

    rows = []
    for row in sheet.findall(f".//{{{MAIN_NS}}}row"):
        values = []
        for cell in row.findall(f"{{{MAIN_NS}}}c"):
            ref = cell.attrib.get("r", "A1")
            index = column_index(ref)
            while len(values) <= index:
                values.append(None)

            cell_type = cell.attrib.get("t")
            raw_value = cell.find(f"{{{MAIN_NS}}}v")
            if cell_type == "inlineStr":
                value = "".join(
                    node.text or "" for node in cell.findall(f".//{{{MAIN_NS}}}t")
                )
            elif raw_value is None:
                value = None
            elif cell_type == "s":
                value = shared_strings[int(raw_value.text)]
            else:
                value = raw_value.text
            values[index] = convert_value(value)
        rows.append(values)

    return rows


def read_data(path):
    rows = read_xlsx_rows(path)
    if not rows:
        return []

    headers = [str(value) for value in rows[0]]
    records = []
    for row in rows[1:]:
        record = {}
        for index, header in enumerate(headers):
            record[header] = row[index] if index < len(row) else None
        if isinstance(record.get("datetime"), (int, float)):
            record["datetime_iso"] = excel_date_to_iso(record["datetime"])
        records.append(record)
    return records


def finite_number(value):
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def clean_regression_rows(rows, x_col, y_col):
    clean = []
    for row in rows:
        x = finite_number(row.get(x_col))
        y = finite_number(row.get(y_col))
        if x is None or y is None:
            continue
        clean.append((x, y))
    return clean


def calculate_prediction_metrics(y_values, predictions, num_params):
    residuals = [y - y_pred for y, y_pred in zip(y_values, predictions)]
    n = len(y_values)
    y_mean = sum(y_values) / n
    ss_res = sum(residual * residual for residual in residuals)
    ss_yy = sum((y - y_mean) ** 2 for y in y_values)
    mse = ss_res / n
    df_resid = n - num_params
    metrics = {
        "r_squared": None,
        "adjusted_r_squared": None,
        "mae": sum(abs(residual) for residual in residuals) / n,
        "mse": mse,
        "rmse": math.sqrt(mse),
        "residual_std_error": None,
    }
    if ss_yy > 0:
        metrics["r_squared"] = 1 - (ss_res / ss_yy)
        if df_resid > 0:
            metrics["adjusted_r_squared"] = 1 - (
                (1 - metrics["r_squared"]) * (n - 1) / df_resid
            )
    if df_resid > 0:
        metrics["residual_std_error"] = math.sqrt(ss_res / df_resid)
    return metrics


def base_model_result(label, model, x_col, y_col, n):
    return {
        "label": label,
        "model": model,
        "x_col": x_col,
        "y_col": y_col,
        "status": "ok",
        "n": n,
        "fit_n": n,
        "slope": None,
        "intercept": None,
        "r": None,
        "r_squared": None,
        "adjusted_r_squared": None,
        "mae": None,
        "mse": None,
        "rmse": None,
        "residual_std_error": None,
        "x_mean": None,
        "y_mean": None,
        "x_scale": None,
        "log_slope": None,
        "a": None,
        "b": None,
        "equation": "",
    }


def calculate_linear_regression(rows, x_col, y_col, label):
    clean = clean_regression_rows(rows, x_col, y_col)
    n = len(clean)
    result = base_model_result(label, "linear", x_col, y_col, n)
    result["x_scale"] = 1.0
    if n < 2:
        result["status"] = "insufficient_data"
        return result

    x_values = [x for x, _ in clean]
    y_values = [y for _, y in clean]
    x_mean = sum(x_values) / n
    y_mean = sum(y_values) / n
    ss_xx = sum((x - x_mean) ** 2 for x in x_values)
    ss_yy = sum((y - y_mean) ** 2 for y in y_values)
    ss_xy = sum((x - x_mean) * (y - y_mean) for x, y in clean)
    if ss_xx == 0:
        result["status"] = "zero_x_variance"
        return result

    slope = ss_xy / ss_xx
    intercept = y_mean - slope * x_mean
    predictions = [intercept + slope * x for x in x_values]
    result.update(
        {
            "slope": slope,
            "intercept": intercept,
            "r": ss_xy / math.sqrt(ss_xx * ss_yy) if ss_yy > 0 else None,
            "x_mean": x_mean,
            "y_mean": y_mean,
            "equation": f"{y_col} = {slope:.10g} * {x_col} + {intercept:.10g}",
        }
    )
    result.update(calculate_prediction_metrics(y_values, predictions, num_params=2))
    return result


def linear_fit_for_curve(curve_values, y_values):
    n = len(y_values)
    curve_mean = sum(curve_values) / n
    y_mean = sum(y_values) / n
    ss_curve = sum((value - curve_mean) ** 2 for value in curve_values)
    if ss_curve == 0:
        return None
    ss_curve_y = sum(
        (curve - curve_mean) * (y - y_mean)
        for curve, y in zip(curve_values, y_values)
    )
    amplitude = ss_curve_y / ss_curve
    intercept = y_mean - amplitude * curve_mean
    return amplitude, intercept


def calculate_flat_exponential_regression(rows, x_col, y_col, label):
    clean = clean_regression_rows(rows, x_col, y_col)
    n = len(clean)
    result = base_model_result(label, "flat_exponential", x_col, y_col, n)
    result["x_scale"] = EXP_FLUX_SCALE
    if n < 3:
        result["status"] = "insufficient_data"
        return result

    x_values = [x for x, _ in clean]
    y_values = [y for _, y in clean]
    scaled_x = [x / EXP_FLUX_SCALE for x in x_values]
    best = None
    search_low = FLAT_EXP_MIN_LOG_SLOPE
    search_high = FLAT_EXP_MAX_LOG_SLOPE

    for _ in range(4):
        step = (search_high - search_low) / max(FLAT_EXP_GRID_SIZE - 1, 1)
        for index in range(FLAT_EXP_GRID_SIZE):
            log_slope = search_low + step * index
            curve = [math.expm1(log_slope * x) for x in scaled_x]
            fit = linear_fit_for_curve(curve, y_values)
            if fit is None:
                continue
            amplitude, intercept = fit
            if amplitude < 0:
                continue
            predictions = [intercept + amplitude * value for value in curve]
            sse = sum((y - y_pred) ** 2 for y, y_pred in zip(y_values, predictions))
            if best is None or sse < best["sse"]:
                best = {
                    "sse": sse,
                    "log_slope": log_slope,
                    "amplitude": amplitude,
                    "intercept": intercept,
                    "curve": curve,
                    "predictions": predictions,
                }

        if best is None:
            break
        search_low = max(FLAT_EXP_MIN_LOG_SLOPE, best["log_slope"] - step)
        search_high = min(FLAT_EXP_MAX_LOG_SLOPE, best["log_slope"] + step)

    if best is None:
        result["status"] = "no_positive_amplitude_fit"
        return result

    curve = best["curve"]
    curve_mean = sum(curve) / n
    y_mean = sum(y_values) / n
    ss_curve = sum((value - curve_mean) ** 2 for value in curve)
    ss_y = sum((y - y_mean) ** 2 for y in y_values)
    ss_curve_y = sum(
        (curve_value - curve_mean) * (y - y_mean)
        for curve_value, y in zip(curve, y_values)
    )
    result.update(
        {
            "slope": best["amplitude"] * best["log_slope"] / EXP_FLUX_SCALE,
            "intercept": best["intercept"],
            "log_slope": best["log_slope"],
            "a": best["amplitude"],
            "b": best["log_slope"] / EXP_FLUX_SCALE,
            "r": ss_curve_y / math.sqrt(ss_curve * ss_y) if ss_curve > 0 and ss_y > 0 else None,
            "x_mean": sum(x_values) / n,
            "y_mean": y_mean,
            "equation": (
                f"{y_col} = {best['intercept']:.10g} + {best['amplitude']:.10g} * "
                f"(exp({best['log_slope']:.10g} * ({x_col} / {EXP_FLUX_SCALE:.10g})) - 1)"
            ),
        }
    )
    result.update(calculate_prediction_metrics(y_values, best["predictions"], num_params=3))
    return result


def calculate_models(rows):
    models = []
    for job in MODEL_JOBS:
        models.append(
            calculate_linear_regression(rows, job["x_col"], job["y_col"], job["label"])
        )
        models.append(
            calculate_flat_exponential_regression(
                rows, job["x_col"], job["y_col"], job["label"]
            )
        )
    return models


def write_models_csv(path, models):
    fields = [
        "label",
        "model",
        "status",
        "x_col",
        "y_col",
        "n",
        "fit_n",
        "slope",
        "intercept",
        "r",
        "r_squared",
        "adjusted_r_squared",
        "mae",
        "mse",
        "rmse",
        "residual_std_error",
        "x_mean",
        "y_mean",
        "x_scale",
        "log_slope",
        "a",
        "b",
        "equation",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(models)


def normalize_models(models):
    normalized = []
    numeric_fields = [
        "slope",
        "intercept",
        "r",
        "r_squared",
        "adjusted_r_squared",
        "mae",
        "mse",
        "rmse",
        "residual_std_error",
        "x_mean",
        "y_mean",
        "x_scale",
        "log_slope",
        "a",
        "b",
    ]
    for model in models:
        item = dict(model)
        for field in numeric_fields:
            value = item.get(field)
            if value in (None, ""):
                item[field] = None
                continue
            try:
                item[field] = float(value)
            except ValueError:
                item[field] = None
        normalized.append(item)
    return normalized


def build_payload(input_path):
    rows = read_data(input_path)
    return {
        "source": str(input_path).replace("\\", "/"),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "rows": rows,
        "models": normalize_models(calculate_models(rows)),
    }


def render_html(payload):
    payload_json = json.dumps(payload, ensure_ascii=False, allow_nan=False)
    return f"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Bee Count Data Viewer</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f7f8fb;
      --ink: #18202f;
      --muted: #667085;
      --line: #d7dce5;
      --panel: #ffffff;
      --blue: #2563eb;
      --green: #0f9f6e;
      --red: #d92d20;
      --amber: #b7791f;
      --violet: #7c3aed;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font: 14px/1.45 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    header {{
      padding: 18px 24px 12px;
      border-bottom: 1px solid var(--line);
      background: #fff;
    }}
    h1 {{ margin: 0 0 6px; font-size: 22px; font-weight: 720; letter-spacing: 0; }}
    .meta {{ color: var(--muted); font-size: 12px; }}
    main {{ padding: 16px 24px 28px; }}
    .toolbar {{
      display: grid;
      grid-template-columns: repeat(6, minmax(120px, 1fr));
      gap: 10px;
      margin-bottom: 16px;
      align-items: end;
    }}
    label {{ display: grid; gap: 5px; color: var(--muted); font-size: 12px; font-weight: 650; }}
    select, input {{
      min-height: 34px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #fff;
      color: var(--ink);
      padding: 6px 8px;
      font: inherit;
    }}
    .segmented {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      border: 1px solid var(--line);
      border-radius: 6px;
      overflow: hidden;
      background: #fff;
      min-height: 34px;
    }}
    .segmented button {{
      border: 0;
      background: transparent;
      color: var(--muted);
      font: inherit;
      font-weight: 700;
      cursor: pointer;
    }}
    .segmented button.active {{ background: var(--ink); color: #fff; }}
    .stats {{
      display: grid;
      grid-template-columns: repeat(5, minmax(130px, 1fr));
      gap: 10px;
      margin-bottom: 16px;
    }}
    .stat {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 10px 12px;
      min-height: 72px;
    }}
    .stat span {{ display: block; color: var(--muted); font-size: 12px; }}
    .stat strong {{ display: block; margin-top: 6px; font-size: 22px; line-height: 1; }}
    .grid {{
      display: grid;
      grid-template-columns: minmax(0, 1.4fr) minmax(360px, 0.9fr);
      gap: 16px;
      align-items: start;
    }}
    section {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      overflow: hidden;
    }}
    section h2 {{
      margin: 0;
      padding: 11px 12px;
      border-bottom: 1px solid var(--line);
      font-size: 14px;
      letter-spacing: 0;
    }}
    .plot-wrap {{ height: 480px; padding: 10px; }}
    svg {{ width: 100%; height: 100%; display: block; }}
    .side {{ display: grid; gap: 16px; }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 12px;
    }}
    th, td {{
      padding: 7px 8px;
      border-bottom: 1px solid #edf0f5;
      text-align: right;
      white-space: nowrap;
    }}
    th {{
      color: var(--muted);
      background: #fafbfc;
      font-weight: 700;
      position: sticky;
      top: 0;
      cursor: pointer;
    }}
    td:first-child, th:first-child, td:nth-child(2), th:nth-child(2) {{ text-align: left; }}
    tr.selected {{ background: #fff4e6; }}
    .table-wrap {{ max-height: 390px; overflow: auto; }}
    .insights {{ padding: 12px; display: grid; gap: 8px; }}
    .insight {{ display: flex; justify-content: space-between; gap: 10px; border-bottom: 1px solid #edf0f5; padding-bottom: 8px; }}
    .insight:last-child {{ border-bottom: 0; padding-bottom: 0; }}
    .insight span {{ color: var(--muted); }}
    .empty {{ padding: 18px; color: var(--muted); }}
    @media (max-width: 1000px) {{
      .toolbar, .stats, .grid {{ grid-template-columns: 1fr; }}
      main, header {{ padding-left: 14px; padding-right: 14px; }}
      .plot-wrap {{ height: 380px; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>Bee Count Data Viewer</h1>
    <div class="meta" id="meta"></div>
  </header>
  <main>
    <div class="toolbar">
      <label>Direction
        <div class="segmented">
          <button id="metric-in" class="active" type="button">IN</button>
          <button id="metric-out" type="button">OUT</button>
        </div>
      </label>
      <label>Model
        <select id="model"></select>
      </label>
      <label>Device
        <select id="device"></select>
      </label>
      <label>Time
        <select id="time"></select>
      </label>
      <label>Error percentile
        <input id="percentile" type="range" min="50" max="99" value="90">
      </label>
      <label>Sort table
        <select id="sort">
          <option value="absError">Absolute error</option>
          <option value="actual">Actual count</option>
          <option value="predicted">Predicted count</option>
          <option value="filteredFlux">Filtered flux</option>
          <option value="retention">Filtered/raw ratio</option>
        </select>
      </label>
    </div>
    <div class="stats" id="stats"></div>
    <div class="grid">
      <section>
        <h2>Output vs Actual</h2>
        <div class="plot-wrap"><svg id="scatter" role="img" aria-label="Output versus actual scatter plot"></svg></div>
      </section>
      <div class="side">
        <section>
          <h2>Large Error Features</h2>
          <div class="insights" id="insights"></div>
        </section>
        <section>
          <h2>Error Rows</h2>
          <div class="table-wrap"><table id="rows"></table></div>
        </section>
      </div>
    </div>
  </main>
  <script>
    const payload = {payload_json};
    const state = {{ metric: "in", selectedIndex: null, scrollSelectedIntoView: false }};
    const controls = {{
      model: document.getElementById("model"),
      device: document.getElementById("device"),
      time: document.getElementById("time"),
      percentile: document.getElementById("percentile"),
      sort: document.getElementById("sort"),
    }};

    const fmt = new Intl.NumberFormat("en-US", {{ maximumFractionDigits: 2 }});
    const compact = new Intl.NumberFormat("en-US", {{ notation: "compact", maximumFractionDigits: 2 }});

    function num(value) {{
      const parsed = Number(value);
      return Number.isFinite(parsed) ? parsed : null;
    }}

    function formatTimeCode(value) {{
      if (value === null || value === undefined || value === "") return "";
      const digits = String(value).replace(/\\D/g, "").padStart(6, "0");
      if (digits.length < 6) return String(value);
      return `${{digits.slice(0, 2)}}:${{digits.slice(2, 4)}}:${{digits.slice(4, 6)}}`;
    }}

    function modelFor(metric) {{
      const selected = controls.model.value;
      return payload.models.find(m => m.y_col === metric && `${{m.model}}:${{m.x_col}}` === selected)
        || payload.models.find(m => m.y_col === metric && m.model === "flat_exponential")
        || payload.models.find(m => m.y_col === metric);
    }}

    function predict(model, x) {{
      if (!model || model.status !== "ok" || x === null) return null;
      if (model.model === "flat_exponential") {{
        return model.intercept + model.a * (Math.exp(model.log_slope * (x / model.x_scale)) - 1);
      }}
      return model.intercept + model.slope * x;
    }}

    function enrichedRows() {{
      const metric = state.metric;
      const model = modelFor(metric);
      const filteredCol = metric === "in" ? "total_filtered_in_flux" : "total_filtered_out_flux";
      const rawCol = metric === "in" ? "total_raw_in_flux" : "total_raw_out_flux";
      return payload.rows.map((row, index) => {{
        const actual = num(row[metric]);
        const filteredFlux = num(row[filteredCol]);
        const rawFlux = num(row[rawCol]);
        const predicted = predict(model, filteredFlux);
        const error = predicted === null || actual === null ? null : predicted - actual;
        return {{
          index,
          device: row.device == null ? "" : String(row.device),
          time: row.time == null ? "" : String(row.time),
          datetime: row.datetime_iso || row.datetime || "",
          date: row.datetime_iso ? row.datetime_iso.slice(0, 10) : "",
          videoTime: formatTimeCode(row.time),
          actual,
          filteredFlux,
          rawFlux,
          predicted,
          error,
          absError: error === null ? null : Math.abs(error),
          retention: rawFlux && rawFlux !== 0 && filteredFlux !== null ? filteredFlux / rawFlux : null,
        }};
      }}).filter(row => row.actual !== null && row.filteredFlux !== null);
    }}

    function filteredRows() {{
      let rows = enrichedRows();
      if (controls.device.value !== "all") rows = rows.filter(row => row.device === controls.device.value);
      if (controls.time.value !== "all") rows = rows.filter(row => row.time === controls.time.value);
      return rows;
    }}

    function percentile(values, pct) {{
      const sorted = values.filter(Number.isFinite).sort((a, b) => a - b);
      if (!sorted.length) return null;
      const index = Math.min(sorted.length - 1, Math.max(0, Math.ceil((pct / 100) * sorted.length) - 1));
      return sorted[index];
    }}

    function mean(values) {{
      const valid = values.filter(Number.isFinite);
      return valid.length ? valid.reduce((sum, value) => sum + value, 0) / valid.length : null;
    }}

    function setupControls() {{
      document.getElementById("meta").textContent =
        `${{payload.source}} - rows ${{payload.rows.length}} - generated ${{payload.generated_at}}`;

      for (const metric of ["in", "out"]) {{
        document.getElementById(`metric-${{metric}}`).onclick = () => {{
          state.metric = metric;
          document.querySelectorAll(".segmented button").forEach(button => button.classList.remove("active"));
          document.getElementById(`metric-${{metric}}`).classList.add("active");
          refreshModelOptions();
          render();
        }};
      }}

      for (const control of Object.values(controls)) control.oninput = render;
      refreshFilterOptions();
      refreshModelOptions();
    }}

    function refreshFilterOptions() {{
      const devices = [...new Set(payload.rows.map(row => row.device).filter(v => v !== null && v !== undefined).map(String))].sort();
      const times = [...new Set(payload.rows.map(row => row.time).filter(v => v !== null && v !== undefined).map(String))].sort();
      controls.device.innerHTML = `<option value="all">All</option>${{devices.map(v => `<option>${{v}}</option>`).join("")}}`;
      controls.time.innerHTML = `<option value="all">All</option>${{times.map(v => `<option>${{v}}</option>`).join("")}}`;
    }}

    function refreshModelOptions() {{
      const metricModels = payload.models.filter(model => model.y_col === state.metric && model.status === "ok");
      controls.model.innerHTML = metricModels.map(model => {{
        const label = `${{model.model}} - ${{model.x_col.replace("total_", "").replace("_flux", "")}}`;
        return `<option value="${{model.model}}:${{model.x_col}}">${{label}}</option>`;
      }}).join("");
      const preferred = metricModels.find(model => model.model === "flat_exponential") || metricModels[0];
      if (preferred) controls.model.value = `${{preferred.model}}:${{preferred.x_col}}`;
    }}

    function renderStats(rows) {{
      const threshold = percentile(rows.map(row => row.absError), Number(controls.percentile.value));
      const large = rows.filter(row => row.absError !== null && row.absError >= threshold);
      const model = modelFor(state.metric);
      const stats = [
        ["Rows", rows.length],
        ["R squared", model?.r_squared],
        ["Mean abs error", mean(rows.map(row => row.absError))],
        [`P${{controls.percentile.value}} abs error`, threshold],
        ["Large error rows", large.length],
      ];
      document.getElementById("stats").innerHTML = stats.map(([label, value]) => `
        <div class="stat"><span>${{label}}</span><strong>${{value == null ? "n/a" : fmt.format(value)}}</strong></div>
      `).join("");
      return {{ threshold, large }};
    }}

    function scale(domainMin, domainMax, rangeMin, rangeMax) {{
      if (domainMax <= domainMin) return () => (rangeMin + rangeMax) / 2;
      return value => rangeMin + ((value - domainMin) / (domainMax - domainMin)) * (rangeMax - rangeMin);
    }}

    function renderScatter(rows, threshold) {{
      const svg = document.getElementById("scatter");
      const width = svg.clientWidth || 800;
      const height = svg.clientHeight || 460;
      const margin = {{ top: 18, right: 20, bottom: 48, left: 64 }};
      const plotWidth = width - margin.left - margin.right;
      const plotHeight = height - margin.top - margin.bottom;
      const xs = rows.map(row => row.filteredFlux).filter(Number.isFinite);
      const ys = rows.map(row => row.actual).filter(Number.isFinite);
      if (!xs.length || !ys.length) {{
        svg.innerHTML = `<text x="24" y="40" fill="#667085">No rows match current filters.</text>`;
        return;
      }}
      const xMin = Math.min(...xs);
      const xMax = Math.max(...xs);
      const yMin = Math.min(0, Math.min(...ys));
      const yMax = Math.max(...ys) * 1.08 + 1;
      const x = scale(xMin, xMax, margin.left, margin.left + plotWidth);
      const y = scale(yMin, yMax, margin.top + plotHeight, margin.top);
      const model = modelFor(state.metric);
      const linePoints = [];
      for (let i = 0; i <= 80; i++) {{
        const xv = xMin + ((xMax - xMin) * i / 80);
        const yv = predict(model, xv);
        if (Number.isFinite(yv)) linePoints.push(`${{x(xv)}},${{y(yv)}}`);
      }}
      const maxError = Math.max(...rows.map(row => row.absError || 0), 1);
      const points = rows.map(row => {{
        const isLarge = row.absError >= threshold;
        const color = row.error > 0 ? "var(--red)" : "var(--blue)";
        const opacity = 0.25 + 0.65 * Math.min(1, (row.absError || 0) / maxError);
        const selected = state.selectedIndex === row.index;
        return `<circle cx="${{x(row.filteredFlux)}}" cy="${{y(row.actual)}}" r="${{selected ? 7 : isLarge ? 5 : 3.2}}"
          fill="${{color}}" fill-opacity="${{opacity}}" stroke="${{selected ? "var(--amber)" : isLarge ? "#111827" : "none"}}"
          stroke-width="${{selected ? 3 : isLarge ? 1 : 0}}" data-index="${{row.index}}">
          <title>device ${{row.device}}, date ${{row.date}}, time ${{row.videoTime || row.time}}\\nactual ${{fmt.format(row.actual)}} - predicted ${{fmt.format(row.predicted)}}\\nerror ${{fmt.format(row.error)}}</title>
        </circle>`;
      }}).join("");
      const ticks = [0, 0.25, 0.5, 0.75, 1];
      const xTicks = ticks.map(t => {{
        const value = xMin + (xMax - xMin) * t;
        return `<g><line x1="${{x(value)}}" x2="${{x(value)}}" y1="${{margin.top}}" y2="${{margin.top + plotHeight}}" stroke="#edf0f5"/><text x="${{x(value)}}" y="${{height - 18}}" text-anchor="middle" fill="#667085" font-size="11">${{compact.format(value)}}</text></g>`;
      }}).join("");
      const yTicks = ticks.map(t => {{
        const value = yMin + (yMax - yMin) * t;
        return `<g><line x1="${{margin.left}}" x2="${{margin.left + plotWidth}}" y1="${{y(value)}}" y2="${{y(value)}}" stroke="#edf0f5"/><text x="${{margin.left - 8}}" y="${{y(value) + 4}}" text-anchor="end" fill="#667085" font-size="11">${{fmt.format(value)}}</text></g>`;
      }}).join("");
      svg.innerHTML = `
        <rect x="0" y="0" width="${{width}}" height="${{height}}" fill="#fff"/>
        ${{xTicks}}${{yTicks}}
        <line x1="${{margin.left}}" y1="${{margin.top + plotHeight}}" x2="${{margin.left + plotWidth}}" y2="${{margin.top + plotHeight}}" stroke="#98a2b3"/>
        <line x1="${{margin.left}}" y1="${{margin.top}}" x2="${{margin.left}}" y2="${{margin.top + plotHeight}}" stroke="#98a2b3"/>
        <polyline points="${{linePoints.join(" ")}}" fill="none" stroke="var(--green)" stroke-width="2.5"/>
        ${{points}}
        <text x="${{margin.left + plotWidth / 2}}" y="${{height - 3}}" text-anchor="middle" fill="#475467" font-size="12">filtered flux</text>
        <text x="16" y="${{margin.top + plotHeight / 2}}" text-anchor="middle" fill="#475467" font-size="12" transform="rotate(-90 16 ${{margin.top + plotHeight / 2}})">actual count</text>
      `;
      svg.querySelectorAll("circle").forEach(point => {{
        point.onclick = () => {{
          state.selectedIndex = Number(point.dataset.index);
          state.scrollSelectedIntoView = true;
          render();
        }};
      }});
    }}

    function groupSummary(rows, key) {{
      const groups = new Map();
      for (const row of rows) {{
        const value = row[key] || "blank";
        if (!groups.has(value)) groups.set(value, []);
        groups.get(value).push(row.absError);
      }}
      return [...groups.entries()]
        .map(([value, errors]) => [value, mean(errors), errors.length])
        .sort((a, b) => (b[1] || 0) - (a[1] || 0))[0];
    }}

    function renderInsights(rows, large) {{
      const device = groupSummary(large, "device");
      const time = groupSummary(large, "time");
      const actualMean = mean(large.map(row => row.actual));
      const rawMean = mean(large.map(row => row.rawFlux));
      const filteredMean = mean(large.map(row => row.filteredFlux));
      const retentionMean = mean(large.map(row => row.retention));
      const html = [
        ["Worst device", device ? `${{device[0]}} - MAE ${{fmt.format(device[1])}} - n=${{device[2]}}` : "n/a"],
        ["Worst time", time ? `${{time[0]}} - MAE ${{fmt.format(time[1])}} - n=${{time[2]}}` : "n/a"],
        ["Large-error actual mean", actualMean],
        ["Large-error raw flux mean", rawMean == null ? null : compact.format(rawMean)],
        ["Large-error filtered flux mean", filteredMean == null ? null : compact.format(filteredMean)],
        ["Filtered/raw ratio mean", retentionMean == null ? null : fmt.format(retentionMean)],
      ].map(([label, value]) => `<div class="insight"><span>${{label}}</span><strong>${{value == null ? "n/a" : value}}</strong></div>`).join("");
      document.getElementById("insights").innerHTML = html || `<div class="empty">No large-error rows.</div>`;
    }}

    function renderTable(rows) {{
      const sortKey = controls.sort.value;
      const sorted = [...rows].sort((a, b) => (b[sortKey] ?? -Infinity) - (a[sortKey] ?? -Infinity));
      const body = sorted.map(row => `
        <tr class="${{state.selectedIndex === row.index ? "selected" : ""}}" data-index="${{row.index}}">
          <td>${{row.device}}</td>
          <td>${{row.date}}</td>
          <td>${{row.videoTime || row.time}}</td>
          <td>${{fmt.format(row.actual)}}</td>
          <td>${{fmt.format(row.predicted)}}</td>
          <td>${{fmt.format(row.error)}}</td>
          <td>${{fmt.format(row.absError)}}</td>
          <td>${{compact.format(row.filteredFlux)}}</td>
          <td>${{compact.format(row.rawFlux || 0)}}</td>
          <td>${{row.retention == null ? "n/a" : fmt.format(row.retention)}}</td>
        </tr>
      `).join("");
      document.getElementById("rows").innerHTML = `
        <thead><tr><th>device</th><th>date</th><th>video time</th><th>actual</th><th>pred</th><th>error</th><th>abs</th><th>filtered</th><th>raw</th><th>ratio</th></tr></thead>
        <tbody>${{body}}</tbody>
      `;
      document.querySelectorAll("#rows tbody tr").forEach(row => {{
        row.onclick = () => {{
          state.selectedIndex = Number(row.dataset.index);
          render();
        }};
      }});
      if (state.scrollSelectedIntoView) {{
        const selected = document.querySelector("#rows tbody tr.selected");
        if (selected) selected.scrollIntoView({{ block: "center" }});
        state.scrollSelectedIntoView = false;
      }}
    }}

    function render() {{
      const rows = filteredRows();
      const {{ threshold, large }} = renderStats(rows);
      renderScatter(rows, threshold);
      renderInsights(rows, large);
      renderTable(rows);
    }}

    setupControls();
    render();
    window.addEventListener("resize", render);
  </script>
</body>
</html>
"""


def main():
    parser = argparse.ArgumentParser(
        description="Build a static validation data viewer with regression models."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--models-output",
        type=Path,
        default=DEFAULT_MODELS_OUTPUT,
        help="Write the regression model summary calculated for the viewer.",
    )
    args = parser.parse_args()

    payload = build_payload(args.input)
    if args.models_output:
        write_models_csv(args.models_output, payload["models"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render_html(payload), encoding="utf-8")
    print(
        f"Wrote {args.output} with {len(payload['rows'])} rows "
        f"and {len(payload['models'])} calculated models"
    )


if __name__ == "__main__":
    main()
