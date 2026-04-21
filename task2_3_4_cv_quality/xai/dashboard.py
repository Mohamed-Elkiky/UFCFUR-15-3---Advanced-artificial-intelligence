"""Admin XAI dashboard showing model decisions and reasoning (AA-49).

Mount with::

    from task2_3_4_cv_quality.xai.dashboard import build_dashboard
    build_dashboard(app)   # registers /admin/xai routes on the Flask app
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

from flask import Blueprint, Flask, jsonify, render_template_string

from api.database import get_all_interactions


# ---------------------------------------------------------------------------
# HTML template (inline – no template folder needed)
# ---------------------------------------------------------------------------

_DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>XAI Admin Dashboard</title>
  <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
  <style>
    body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
    h1 { color: #2c3e50; }
    h2 { color: #34495e; border-bottom: 2px solid #3498db; padding-bottom: 6px; }
    table { border-collapse: collapse; width: 100%; background: white; }
    th, td { border: 1px solid #ddd; padding: 8px 12px; text-align: left; }
    th { background: #3498db; color: white; }
    tr:nth-child(even) { background: #f9f9f9; }
    .card { background: white; border-radius: 6px; padding: 16px;
            margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,.1); }
    .grade-A { color: #27ae60; font-weight: bold; }
    .grade-B { color: #f39c12; font-weight: bold; }
    .grade-C { color: #e74c3c; font-weight: bold; }
    .flagged  { background: #ffe5e5; }
  </style>
</head>
<body>
  <h1>XAI Admin Dashboard</h1>

  <div class="card">
    <h2>Recent Predictions</h2>
    <table id="predictions-table">
      <thead>
        <tr>
          <th>#</th><th>Timestamp</th><th>Endpoint</th>
          <th>Prediction</th><th>Confidence</th><th>Grade</th><th>Overridden</th>
        </tr>
      </thead>
      <tbody id="predictions-body"></tbody>
    </table>
  </div>

  <div class="card">
    <h2>Grade Distribution</h2>
    <div id="grade-pie" style="height:350px;"></div>
  </div>

  <div class="card">
    <h2>Override Rate Over Time</h2>
    <div id="override-chart" style="height:350px;"></div>
  </div>

  <div class="card">
    <h2>Producer Bias Metrics</h2>
    <div id="bias-table-container">Loading…</div>
  </div>

  <script>
  async function loadData() {
    const res = await fetch('/admin/xai/api/data');
    const d = await res.json();

    // Predictions table
    const tbody = document.getElementById('predictions-body');
    (d.interactions || []).slice(0, 50).forEach((r, i) => {
      const gradeClass = r.grade ? 'grade-' + r.grade : '';
      const overrideClass = r.was_overridden ? 'flagged' : '';
      tbody.innerHTML += `<tr class="${overrideClass}">
        <td>${i+1}</td>
        <td>${r.timestamp || ''}</td>
        <td>${r.endpoint || ''}</td>
        <td>${r.prediction || ''}</td>
        <td>${r.confidence != null ? (r.confidence*100).toFixed(1)+'%' : 'N/A'}</td>
        <td class="${gradeClass}">${r.grade || 'N/A'}</td>
        <td>${r.was_overridden ? 'Yes' : 'No'}</td>
      </tr>`;
    });

    // Grade pie chart
    const gradeCounts = d.grade_counts || {};
    Plotly.newPlot('grade-pie', [{
      type: 'pie',
      labels: Object.keys(gradeCounts),
      values: Object.values(gradeCounts),
      marker: { colors: ['#27ae60', '#f39c12', '#e74c3c', '#95a5a6'] },
      textinfo: 'label+percent',
    }], { title: 'Grade Distribution', margin: {t:40} });

    // Override rate chart
    const ov = d.override_weekly || [];
    Plotly.newPlot('override-chart', [{
      type: 'scatter', mode: 'lines+markers',
      x: ov.map(r => r.date || r.week),
      y: ov.map(r => r.override_rate),
      name: 'Override rate',
      line: { color: '#e74c3c' },
    }], {
      title: 'Weekly Override Rate',
      yaxis: { tickformat: '.0%', title: 'Override rate' },
      xaxis: { title: 'Week' },
      shapes: [{ type: 'line', x0: 0, x1: 1, xref: 'paper',
                 y0: 0.2, y1: 0.2, line: { color: 'orange', dash: 'dash' } }],
      margin: {t:40},
    });

    // Producer bias table
    const producers = d.producer_bias || [];
    let html = '<table><thead><tr>'
             + '<th>Producer ID</th><th>Samples</th><th>Precision</th>'
             + '<th>Rec. Rate</th><th>Avg Confidence</th><th>Flagged</th>'
             + '</tr></thead><tbody>';
    producers.forEach(p => {
      const flagClass = p.flagged ? 'flagged' : '';
      html += `<tr class="${flagClass}">
        <td>${p.producer_id}</td>
        <td>${p.sample_count}</td>
        <td>${(p.precision*100).toFixed(1)}%</td>
        <td>${(p.recommendation_rate*100).toFixed(1)}%</td>
        <td>${p.avg_confidence != null ? (p.avg_confidence*100).toFixed(1)+'%' : 'N/A'}</td>
        <td>${p.flagged ? '⚠ Yes' : 'No'}</td>
      </tr>`;
    });
    html += '</tbody></table>';
    document.getElementById('bias-table-container').innerHTML = html;
  }

  loadData();
  </script>
</body>
</html>
"""

# ---------------------------------------------------------------------------
# Blueprint factory
# ---------------------------------------------------------------------------

def _build_grade_counts(interactions: List[Dict]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for row in interactions:
        g = row.get("grade") or "N/A"
        counts[g] = counts.get(g, 0) + 1
    return counts


def _build_override_weekly(interactions: List[Dict]) -> List[Dict]:
    """Aggregate override rate by ISO week from logged interactions."""
    from collections import defaultdict

    week_data: Dict[str, Dict[str, int]] = defaultdict(lambda: {"total": 0, "overrides": 0})
    for row in interactions:
        ts = row.get("timestamp", "")
        week = ts[:10] if ts else "unknown"
        week_data[week]["total"] += 1
        if row.get("was_overridden"):
            week_data[week]["overrides"] += 1

    result = []
    for date, counts in sorted(week_data.items()):
        total = counts["total"]
        overrides = counts["overrides"]
        result.append({
            "date": date,
            "override_rate": overrides / total if total > 0 else 0.0,
        })
    return result


def _build_producer_bias() -> List[Dict]:
    """Attempt to load producer bias from evaluate.py. Returns empty list on failure."""
    try:
        from task1_purchase_prediction.src.evaluate import (
            bias_audit,
            build_predictions_df,
        )

        predictions_df = build_predictions_df()
        audit = bias_audit(predictions_df)
        per_producer = audit["per_producer"]
        return per_producer.to_dict(orient="records")
    except Exception:
        return []


def build_dashboard(app: Flask) -> None:
    """Register the /admin/xai blueprint on an existing Flask app.

    Parameters
    ----------
    app : Flask
        The Flask application instance.
    """
    xai_bp = Blueprint("xai_admin", __name__, url_prefix="/admin/xai")

    @xai_bp.route("/", methods=["GET"])
    def dashboard_index():
        return render_template_string(_DASHBOARD_HTML)

    @xai_bp.route("/api/data", methods=["GET"])
    def dashboard_data():
        interactions = get_all_interactions()
        return jsonify({
            "interactions": interactions,
            "grade_counts": _build_grade_counts(interactions),
            "override_weekly": _build_override_weekly(interactions),
            "producer_bias": _build_producer_bias(),
        })

    app.register_blueprint(xai_bp)
