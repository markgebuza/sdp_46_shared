"""Score cached forecasts, write metric tables and figures.

Usage: python -m bench.analyze
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

from bench.data import Dataset, dataset_names, load_dataset
from bench.metrics import coverage, crps, gmean, mase, seasonal_error
from bench.models import MODELS
from bench.run import PRED_DIR
from bench.transforms import (
    cap_to_history,
    clip_nonneg,
    flip_average,
    is_integer_valued,
    round_counts,
    scale_spread,
    sort_quantiles,
)

RESULTS = PRED_DIR.parent
FIG_DIR = RESULTS / "figures"
CALIB_GRID = np.round(np.arange(0.5, 2.51, 0.05), 2)

# Validated categorical slots 1-3 (dataviz reference palette, light mode).
COLORS = {"Chronos-2": "#2a78d6", "TimesFM-3.0": "#eb6834", "Toto-2.0": "#1baf7a"}
INK, INK_2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"
DIVERGING = LinearSegmentedColormap.from_list("div", ["#1c5cab", "#86b6ef", "#f0efec", "#ee8f8f", "#b42e2e"])

VARIANTS = {
    "raw": ("Raw", "none"),
    "impute": ("Linear imputation", "pre"),
    "log1p": ("log1p transform", "pre"),
    "log1p_cap": ("log1p + cap at 10x max", "pre+post"),
    "hampel": ("Hampel outlier clip", "pre"),
    "ctx256": ("Context 256 (vs 1024)", "pre"),
    "ctx2048": ("Context 2048 (vs 1024)", "pre"),
    "flip": ("Sign-flip averaging", "pre+post"),
    "sort": ("Sort quantiles", "post"),
    "clip0": ("Clip at zero", "post"),
    "round": ("Round counts", "post"),
    "calib": ("Interval calibration", "post"),
    "post_all": ("Sort + clip0 + calib", "post"),
}


def seasonal_naive(ds: Dataset, tasks) -> np.ndarray:
    out = np.empty((len(tasks), ds.horizon, 9))
    for k, task in enumerate(tasks):
        hist = pd.Series(ds.history(task)).ffill().fillna(0.0).to_numpy()
        m = ds.season_for(task)
        last = hist[-m:]
        out[k] = np.tile(last, -(-ds.horizon // m))[: ds.horizon, None]
    return out


def best_spread(val_q: np.ndarray, val_y: np.ndarray) -> float:
    scores = [crps(val_y, scale_spread(sort_quantiles(val_q), f)) for f in CALIB_GRID]
    return float(CALIB_GRID[int(np.nanargmin(scores))])


def build_variants(model: str, ds: Dataset, nonneg, integer, hist_max, val_y) -> tuple[dict, float]:
    def load(cfg, key="test"):
        return np.load(PRED_DIR / cfg / model / f"{ds.name}.npz")[key]

    raw = load("raw")
    v = {"raw": raw}
    for cfg in ("impute", "log1p", "hampel", "ctx256", "ctx2048"):
        v[cfg] = load(cfg)
    v["log1p_cap"] = np.where(nonneg[:, None, None], cap_to_history(v["log1p"], hist_max), v["log1p"])
    v["flip"] = flip_average(raw, load("negate"))
    v["sort"] = sort_quantiles(raw)
    v["clip0"] = clip_nonneg(raw, nonneg)
    v["round"] = round_counts(raw, integer)
    factor = best_spread(load("raw", "val"), val_y) if ds.val_tasks else 1.0
    v["calib"] = scale_spread(sort_quantiles(raw), factor)
    v["post_all"] = clip_nonneg(scale_spread(sort_quantiles(raw), factor), nonneg)
    return v, factor


def score_dataset(ds: Dataset) -> tuple[list[dict], dict]:
    y = np.stack([ds.target(t) for t in ds.tasks])
    scales = np.array([seasonal_error(ds.history(t), ds.season_for(t)) for t in ds.tasks])
    nonneg = np.array([np.nanmin(ds.history(t), initial=0.0) >= 0 for t in ds.tasks])
    integer = np.array([is_integer_valued(ds.history(t)) for t in ds.tasks])
    hist_max = np.array([np.nanmax(ds.history(t), initial=0.0) for t in ds.tasks])
    val_y = np.stack([ds.target(t) for t in ds.val_tasks]) if ds.val_tasks else None

    def row(model, variant, q, **extra):
        task_mase = np.array([mase(y[k], q[k, :, 4], scales[k]) for k in range(len(y))])
        return dict(
            dataset=ds.name, model=model, variant=variant,
            mase=np.nanmean(task_mase), crps=crps(y, q), cov80=coverage(y, q),
            crossing=float((np.diff(q, axis=-1) < -1e-9).any(-1).mean()), **extra,
        )

    rows = [row("Seasonal Naive", "raw", seasonal_naive(ds, ds.tasks))]
    examples = {"y": y, "scales": scales, "q": {}}
    for model in MODELS:
        variants, factor = build_variants(model, ds, nonneg, integer, hist_max, val_y)
        examples["q"][model] = variants["raw"]
        for name, q in variants.items():
            rows.append(row(model, name, q, calib_factor=factor if name in ("calib", "post_all") else np.nan))
    return rows, examples


def relative(df: pd.DataFrame) -> pd.DataFrame:
    base = df[df.model == "Seasonal Naive"].set_index("dataset")[["mase", "crps"]]
    out = df[df.model != "Seasonal Naive"].copy()
    out["rel_mase"] = out.mase / out.dataset.map(base.mase)
    out["rel_crps"] = out.crps / out.dataset.map(base.crps)
    return out


def summarize(rel: pd.DataFrame) -> pd.DataFrame:
    g = rel.groupby(["model", "variant"])
    summary = pd.DataFrame({
        "rel_mase": g.rel_mase.apply(gmean),
        "rel_crps": g.rel_crps.apply(gmean),
        "cov80": g.cov80.mean(),
        "crossing": g.crossing.mean(),
    }).reset_index()
    raw = summary[summary.variant == "raw"].set_index("model")
    summary["d_mase_pct"] = 100 * (summary.rel_mase / summary.model.map(raw.rel_mase) - 1)
    summary["d_crps_pct"] = 100 * (summary.rel_crps / summary.model.map(raw.rel_crps) - 1)
    order = {v: i for i, v in enumerate(VARIANTS)}
    return summary.sort_values(["variant", "model"], key=lambda s: s.map(order) if s.name == "variant" else s)


def style(ax, grid_axis="y"):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=8, length=0)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def fig_dataset_forecasts(datasets: dict, examples: dict):
    names = list(datasets)
    ncols = 3
    nrows = -(-len(names) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 3.0 * nrows), facecolor=SURFACE)
    for ax, name in zip(axes.flat, names):
        ds, ex = datasets[name], examples[name]
        rel = np.mean([
            [mase(ex["y"][k], ex["q"][m][k, :, 4], ex["scales"][k]) for k in range(len(ex["y"]))] for m in MODELS
        ], axis=0)
        rel = np.where(np.isfinite(rel), rel, np.inf)
        k = int(np.argsort(rel)[len(rel) // 2])  # task with median error: a typical case, not a cherry-pick
        task = ds.tasks[k]
        hist = ds.history(task)[-3 * ds.horizon :]
        t_hist = np.arange(-len(hist), 0)
        t_fut = np.arange(ds.horizon)
        ax.plot(t_hist, hist, color=INK, linewidth=1.0)
        ax.plot(t_fut, ex["y"][k], color=INK, linewidth=1.0, label="Actual")
        for model, color in COLORS.items():
            q = ex["q"][model][k]
            ax.fill_between(t_fut, q[:, 0], q[:, 8], color=color, alpha=0.18, linewidth=0)
            ax.plot(t_fut, q[:, 4], color=color, linewidth=1.6, label=f"{model} median, 10-90% band")
        ax.axvline(0, color=MUTED, linewidth=0.8)
        ax.set_title(f"{name}  ({ds.freq}, h={ds.horizon})", fontsize=9, color=INK, loc="left")
        style(ax)
    for ax in axes.flat[len(names) :]:
        ax.axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.suptitle("Raw forecasts on a typical test window per dataset (vertical line = forecast start)",
                 y=0.997, fontsize=13, color=INK, x=0.01, ha="left")
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.01, 0.985), ncol=4, frameon=False,
               fontsize=9, labelcolor=INK_2)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    fig.savefig(FIG_DIR / "dataset_forecasts.png", dpi=130, facecolor=SURFACE)
    plt.close(fig)


def fig_raw_scores(summary: pd.DataFrame):
    raw = summary[summary.variant == "raw"].set_index("model").loc[list(MODELS)]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2), facecolor=SURFACE)
    for ax, col, title in zip(axes, ("rel_mase", "rel_crps"), ("MASE", "CRPS")):
        vals = raw[col].to_numpy()
        ax.bar(range(3), vals, width=0.55, color=[COLORS[m] for m in raw.index])
        ax.axhline(1.0, color=MUTED, linewidth=1)
        ax.text(2.45, 1.0, "Seasonal Naive = 1.0", fontsize=7, color=INK_2, va="bottom", ha="right")
        for i, v in enumerate(vals):
            ax.text(i, v + 0.015, f"{v:.3f}", ha="center", fontsize=8, color=INK)
        ax.set_xticks(range(3), raw.index)
        ax.set_ylim(0, 1.12)
        ax.set_title(f"Relative {title} (geo. mean, lower is better)", fontsize=10, color=INK, loc="left")
        style(ax)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "raw_relative_scores.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)


def heatmap(ax, mat: pd.DataFrame, norm, fmt):
    values = mat.to_numpy(dtype=float)
    im = ax.imshow(np.nan_to_num(values, posinf=1e12, neginf=-1e12), cmap=DIVERGING, norm=norm, aspect="auto")
    ax.set_xticks(range(mat.shape[1]), mat.columns, rotation=35, ha="right", fontsize=8)
    ax.set_yticks(range(mat.shape[0]), mat.index, fontsize=8)
    for (i, j), v in np.ndenumerate(values):
        label = "inf" if np.isinf(v) else (">999" if abs(v) >= 1000 else fmt(v))
        on_dark = abs(norm(v) - 0.5) > 0.38 if np.isfinite(v) else True
        ax.text(j, i, label, ha="center", va="center", fontsize=6.5, color=SURFACE if on_dark else INK)
    ax.tick_params(length=0, colors=INK_2)
    for s in ax.spines.values():
        s.set_visible(False)
    return im


def fig_raw_heatmap(rel: pd.DataFrame):
    raw = rel[rel.variant == "raw"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 7), facecolor=SURFACE)
    for ax, col, title in zip(axes, ("rel_mase", "rel_crps"), ("MASE", "CRPS")):
        mat = raw.pivot(index="dataset", columns="model", values=col)[list(MODELS)]
        im = heatmap(ax, mat, TwoSlopeNorm(vcenter=1.0, vmin=0.2, vmax=1.8), lambda v: f"{v:.2f}")
        if ax is axes[1]:
            ax.set_yticklabels([])
        ax.set_title(f"Relative {title} vs Seasonal Naive", fontsize=10, color=INK, loc="left")
    cb = fig.colorbar(im, ax=axes, shrink=0.6, label="< 1 better than baseline  |  > 1 worse")
    cb.outline.set_visible(False)
    fig.savefig(FIG_DIR / "raw_per_dataset_heatmap.png", dpi=150, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def fig_sensitivity(summary: pd.DataFrame):
    variants = [v for v in VARIANTS if v != "raw"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2), facecolor=SURFACE, sharey=True)
    y = np.arange(len(variants))
    lo, hi = -15, 30
    for ax, col, title in zip(axes, ("d_mase_pct", "d_crps_pct"), ("MASE", "CRPS")):
        for j, model in enumerate(MODELS):
            vals = summary[summary.model == model].set_index("variant").loc[variants, col].to_numpy()
            pos = y + (j - 1) * 0.26
            ax.barh(pos, np.clip(vals, lo, hi), height=0.24, color=COLORS[model], label=model)
            for p, v in zip(pos, vals):
                if not lo <= v <= hi:  # bar runs off the axis: print the real value at its end
                    ax.text(hi, p, f" {v:+.0f}%" if np.isfinite(v) else " inf", va="center", fontsize=7, color=INK)
        ax.set_xlim(lo, hi + 9)
        ax.axvline(0, color=MUTED, linewidth=1)
        ax.set_yticks(y, [VARIANTS[v][0] for v in variants], fontsize=8)
        ax.set_title(f"Change in relative {title} vs raw (%), negative = better", fontsize=10, color=INK, loc="left")
        style(ax, "x")
    axes[0].invert_yaxis()
    axes[0].legend(frameon=False, fontsize=8, loc="lower right", labelcolor=INK_2)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "sensitivity_summary.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)


def fig_sensitivity_heatmap(rel: pd.DataFrame):
    variants = [v for v in VARIANTS if v != "raw"]
    fig, axes = plt.subplots(1, 3, figsize=(17, 7), facecolor=SURFACE, sharey=True)
    for ax, model in zip(axes, MODELS):
        sub = rel[rel.model == model].pivot(index="dataset", columns="variant", values="rel_crps")
        mat = 100 * (sub[variants].div(sub["raw"], axis=0) - 1)
        mat.columns = [VARIANTS[v][0] for v in variants]
        im = heatmap(ax, mat, TwoSlopeNorm(vcenter=0, vmin=-30, vmax=30), lambda v: f"{v:+.0f}" if abs(v) >= 0.5 else "")
        ax.set_title(f"{model}: CRPS change vs raw (%)", fontsize=10, color=INK, loc="left")
    cb = fig.colorbar(im, ax=axes, shrink=0.6, label="blue = technique helps, red = hurts (clipped at +/-30%)")
    cb.outline.set_visible(False)
    fig.savefig(FIG_DIR / "sensitivity_per_dataset.png", dpi=140, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def fig_coverage(rel: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(9, 6), facecolor=SURFACE)
    names = sorted(rel.dataset.unique())
    y = np.arange(len(names))
    for j, model in enumerate(MODELS):
        for variant, marker, fill in (("raw", "o", True), ("calib", "o", False)):
            sub = rel[(rel.model == model) & (rel.variant == variant)].set_index("dataset").loc[names]
            ax.scatter(sub.cov80, y + (j - 1) * 0.22, s=36, marker=marker,
                       color=COLORS[model] if fill else "none", edgecolors=COLORS[model], linewidths=1.4,
                       label=f"{model} ({'raw' if fill else 'calibrated'})", zorder=3)
    ax.axvline(0.8, color=MUTED, linewidth=1)
    ax.text(0.8, -0.9, " nominal 80%", fontsize=8, color=INK_2)
    ax.set_yticks(y, names, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_title("Share of actuals inside the 10-90% band (target 0.80)", fontsize=10, color=INK, loc="left")
    style(ax, "x")
    ax.legend(frameon=False, fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.06), ncol=3, labelcolor=INK_2)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "interval_coverage.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)


def runtime_table() -> pd.DataFrame:
    records = [json.loads(line) for line in (PRED_DIR / "timing.jsonl").read_text().splitlines()]
    t = pd.DataFrame(records)
    t = t[t.config == "raw"].groupby("model")[["n_run", "seconds"]].sum()
    t["ms_per_series"] = 1000 * t.seconds / t.n_run
    return t.reset_index()


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    datasets = {name: load_dataset(name) for name in dataset_names()}
    rows, examples = [], {}
    for name, ds in datasets.items():
        r, examples[name] = score_dataset(ds)
        rows += r
        print("scored", name, flush=True)
    df = pd.DataFrame(rows)
    rel = relative(df)
    summary = summarize(rel)
    df.to_csv(RESULTS / "metrics_per_dataset.csv", index=False)
    rel.to_csv(RESULTS / "relative_per_dataset.csv", index=False)
    summary.to_csv(RESULTS / "summary.csv", index=False)
    runtime_table().to_csv(RESULTS / "runtime.csv", index=False)

    fig_dataset_forecasts(datasets, examples)
    fig_raw_scores(summary)
    fig_raw_heatmap(rel)
    fig_sensitivity(summary)
    fig_sensitivity_heatmap(rel)
    fig_coverage(rel)
    with pd.option_context("display.width", 200, "display.max_rows", 100):
        print(summary.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
