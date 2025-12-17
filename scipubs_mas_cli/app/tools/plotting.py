from __future__ import annotations

import base64
import io
from typing import List, Dict, Any
import matplotlib
matplotlib.use('Agg')  # Используем неинтерактивный бэкенд
import matplotlib.pyplot as plt


def line_chart_base64(
    rows: List[Dict[str, Any]],
    x_key: str,
    y_key: str,
    group_key: str | None = None,
) -> str:
    """Build a line chart and return it as base64-encoded PNG.

    If `group_key` is present (e.g. compare_groups returns `group_topic`), builds
    a multi-series plot with a legend and fills missing x values with zeros.
    """
    if not rows:
        raise ValueError("No data to plot.")

    if group_key is None and isinstance(rows[0], dict) and "group_topic" in rows[0]:
        group_key = "group_topic"
    # If we have merged results from multiple sources (SQL vs OpenAlex), allow multi-series
    # plots by grouping on `source`.
    if group_key is None and isinstance(rows[0], dict) and "source" in rows[0]:
        group_key = "source"

    fig, ax = plt.subplots()

    if group_key and group_key in rows[0]:
        # Multi-series case
        series: dict[str, dict[Any, float]] = {}
        xs: list[Any] = []
        for r in rows:
            x = r.get(x_key)
            y = r.get(y_key)
            g = r.get(group_key)
            if x is None or y is None or g is None:
                continue
            xs.append(x)
            gk = str(g)
            series.setdefault(gk, {})
            series[gk][x] = series[gk].get(x, 0) + float(y)

        if not xs or not series:
            raise ValueError("No valid data to plot.")

        # Determine X axis domain
        if all(isinstance(v, int) for v in xs):
            x_min = int(min(xs))
            x_max = int(max(xs))
            x_values = list(range(x_min, x_max + 1))
        else:
            x_values = sorted(set(xs))

        for gk in sorted(series.keys()):
            y_values = [series[gk].get(x, 0.0) for x in x_values]
            ax.plot(x_values, y_values, label=gk)
        ax.legend()
    else:
        # Single series case
        points = [
            (r.get(x_key), r.get(y_key))
            for r in rows
            if r.get(x_key) is not None and r.get(y_key) is not None
        ]
        # sort by x when possible
        try:
            points.sort(key=lambda p: p[0])
        except Exception:
            pass
        x = [p[0] for p in points]
        y = [p[1] for p in points]
        ax.plot(x, y)

    ax.set_xlabel(x_key)
    ax.set_ylabel(y_key)
    ax.grid(True)

    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


def bar_chart_base64(
    rows: List[Dict[str, Any]],
    x_key: str,
    y_key: str,
    *,
    top_n: int = 20,
    max_label_len: int = 48,
    include_other: bool = True,
    other_label: str = "Other",
) -> str:
    """Build a categorical bar chart and return it as base64-encoded PNG.

    For wide categorical axes (e.g. journals/conferences), the function:
    - sorts by y descending,
    - keeps top-N categories,
    - optionally aggregates the rest into "Other",
    - chooses horizontal bars automatically for readability.
    """
    if not rows:
        raise ValueError("No data to plot.")

    # Clean and normalize
    clean: list[tuple[str, float]] = []
    for r in rows:
        x = r.get(x_key)
        y = r.get(y_key)
        if x is None or y is None:
            continue
        try:
            yf = float(y)
        except Exception:
            continue
        clean.append((str(x), yf))
    if not clean:
        raise ValueError("No valid data to plot.")

    clean.sort(key=lambda t: t[1], reverse=True)

    # Cap categories for legibility
    if top_n > 0 and len(clean) > top_n:
        head = clean[:top_n]
        tail = clean[top_n:]
        if include_other and tail:
            other_sum = sum(v for _, v in tail)
            clean = head + [(other_label, other_sum)]
        else:
            clean = head

    labels = []
    values = []
    for name, val in clean:
        if max_label_len > 0 and len(name) > max_label_len:
            name = name[: max_label_len - 1] + "…"
        labels.append(name)
        values.append(val)

    # Decide orientation
    long_labels = any(len(s) > 18 for s in labels)
    many_cats = len(labels) > 12
    horizontal = many_cats or long_labels

    if horizontal:
        # Height scales with number of categories
        height = max(4.0, min(0.35 * len(labels) + 1.5, 12.0))
        fig, ax = plt.subplots(figsize=(10.0, height))
        ax.barh(labels[::-1], values[::-1])
        ax.set_ylabel(x_key)
        ax.set_xlabel(y_key)
        ax.grid(True, axis="x")
    else:
        fig, ax = plt.subplots(figsize=(10.0, 5.0))
        ax.bar(labels, values)
        ax.set_xlabel(x_key)
        ax.set_ylabel(y_key)
        ax.grid(True, axis="y")
        ax.tick_params(axis="x", labelrotation=45)

    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")
