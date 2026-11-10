"""Build the short key-takeaways PDF from results/summary.csv.

Usage: python -m bench.report
"""

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from bench.analyze import FIG_DIR, RESULTS, VARIANTS
from bench.models import MODELS

TAKEAWAYS = [
    "<b>TimesFM-3.0 is the most accurate on raw data</b>: relative MASE 0.629 / CRPS 0.470, vs Toto-2.0 "
    "0.648 / 0.486 and Chronos-2 0.654 / 0.504. It has the best CRPS on 12 of 18 datasets (Chronos-2 4, Toto-2.0 2; "
    "Toto wins solar and hospital). All three beat Seasonal Naive on every dataset by roughly 50% in CRPS, so the "
    "gaps between models (3-7%) are small next to their gap to the baseline.",
    "<b>Intervals:</b> TimesFM-3.0 (79.8%) and Toto-2.0 (79.5%) cover close to the nominal 80%; Chronos-2 is "
    "slightly too narrow (76.4%). Toto-2.0 never outputs crossed quantiles; TimesFM-3.0 does on 8.5% of forecast "
    "steps and Chronos-2 on 3.8%, but sorting them changes CRPS by under 0.1%.",
    "<b>Missing values are Chronos-2's weak spot.</b> On bitbrains (13% missing) its native NaN handling gives "
    "0.94x baseline CRPS; linearly imputing the context first cuts CRPS by 30%. TimesFM-3.0 already interpolates "
    "internally and Toto-2.0 masks missing points, so imputation barely moves them, except that back-filling "
    "<i>leading</i> gaps hurt TimesFM on electricity (+14%), since it normally drops them. Impute interior gaps only.",
    "<b>A blanket log1p transform is harmful for every model</b>: exponentiating the forecast back amplifies the "
    "upper tail. TimesFM-3.0 on covid_deaths and Chronos-2 on bitbrains blow up by >1000%, and Toto-2.0 overflows "
    "to infinity on one sparse car_parts series. Capping at 10x the history maximum contains the damage "
    "(Chronos +1.7%, Toto +5%, TimesFM +16% CRPS) but still does not beat raw. The models scale inputs internally; "
    "leave them alone.",
    "<b>Outlier clipping (Hampel) hurts all three by 5-7% CRPS.</b> It removes real signal: with the daily "
    "seasonal period of 1 used by GIFT-Eval, it erased the weekly pattern in us_births (+120-170%). It helped only "
    "on spiky bitbrains (Chronos -8%, Toto -5%).",
    "<b>Context length is the biggest lever.</b> Cutting context from 1,024 to 256 costs about +10% CRPS for "
    "every model (up to +100% on bizitobs and +30-50% on us_births). Going to 2,048 gives a further ~1% "
    "(largest for Chronos-2 and Toto-2.0). Give the models as much history as you can afford.",
    "<b>Cheap post-processing is safe but nearly free of benefit.</b> Clipping at zero and rounding count data "
    "improve CRPS by under 0.3%. Sign-flip averaging (forecast x and -x, average) helps Chronos-2 and "
    "TimesFM-3.0 by ~0.6% at 2x compute and is neutral for Toto. Rescaling interval width using one validation "
    "window made things <i>worse</i> (+0.7-1.4%; Toto on m4_hourly +17%): the raw intervals are already well "
    "calibrated and one window is too noisy to tune on.",
    "<b>Overall sensitivity:</b> Toto-2.0 is the most robust to sensible preprocessing (its worst case is the "
    "log transform), Chronos-2 is the most sensitive to how missing data is presented, and TimesFM-3.0 is the most "
    "sensitive to the log transform. All three models share the same three harmful steps: short context, log "
    "transform, outlier clipping.",
    "<b>Caveats:</b> up to 600 sampled windows per dataset, one seed, univariate mode only, CPU inference "
    "(Chronos-2 40 ms, Toto-2.0 49 ms, TimesFM-3.0 53 ms per series). Differences under ~1-2% should be read as "
    "ties. TimesFM-3.0 weights are licensed for non-commercial use only.",
]
SETUP = (
    "18 GIFT-Eval datasets (hourly to monthly, 7,300+ univariate series). GIFT-Eval protocol: horizon 48 (H), "
    "30 (D), 8 (W), 12 (M); rolling non-overlapping test windows over the last 10% of each series (max 20). "
    "Multivariate sets are split into univariate series. Up to 600 test windows per dataset (fixed seed). "
    "Context = last 1,024 points. Models: Chronos-2 (120M), TimesFM-3.0 (331M), Toto-2.0-313m, CPU, zero-shot. "
    "Scores are divided by Seasonal Naive per dataset, then combined with a geometric mean (lower is better; "
    "1.0 = baseline). CRPS = mean weighted quantile loss over the 0.1-0.9 deciles; MASE uses the median."
)

INK = colors.HexColor("#0b0b0b")
GRID = colors.HexColor("#e1e0d9")
HEAD = colors.HexColor("#f0efec")


def pct(v: float) -> str:
    if v != v or abs(v) == float("inf"):
        return "inf"
    return "0.0%" if abs(v) < 0.05 else f"{v:+.1f}%"


def table(rows, col_widths):
    t = Table(rows, colWidths=col_widths, hAlign="LEFT")
    t.setStyle(TableStyle([
        ("FONT", (0, 0), (-1, -1), "Helvetica", 8),
        ("FONT", (0, 0), (-1, 0), "Helvetica-Bold", 8),
        ("BACKGROUND", (0, 0), (-1, 0), HEAD),
        ("LINEBELOW", (0, 0), (-1, -1), 0.4, GRID),
        ("ALIGN", (1, 0), (-1, -1), "RIGHT"),
        ("TEXTCOLOR", (0, 0), (-1, -1), INK),
        ("TOPPADDING", (0, 0), (-1, -1), 2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
    ]))
    return t


def main():
    summary = pd.read_csv(RESULTS / "summary.csv")
    runtime = pd.read_csv(RESULTS / "runtime.csv").set_index("model")
    models = list(MODELS)
    styles = getSampleStyleSheet()
    body = ParagraphStyle("body", parent=styles["Normal"], fontSize=9, leading=12)
    small = ParagraphStyle("small", parent=body, fontSize=7.5, leading=10, textColor=colors.HexColor("#52514e"))
    h2 = ParagraphStyle("h2", parent=styles["Heading2"], fontSize=12, spaceBefore=8, spaceAfter=4)

    story = [
        Paragraph("Zero-shot forecasting: TimesFM-3.0 vs Chronos-2 vs Toto-2.0", styles["Title"]),
        Paragraph(SETUP, small),
        Paragraph("Key takeaways", h2),
    ]
    story += [Paragraph(f"&bull; {t}", body) for t in TAKEAWAYS]

    raw = summary[summary.variant == "raw"].set_index("model")
    rows = [["Model", "Rel. MASE", "Rel. CRPS", "80% coverage", "Crossed quantiles", "CPU ms / series"]]
    for m in models:
        r = raw.loc[m]
        rows.append([m, f"{r.rel_mase:.3f}", f"{r.rel_crps:.3f}", f"{r.cov80:.1%}", f"{r.crossing:.1%}",
                     f"{runtime.loc[m, 'ms_per_series']:.0f}"])
    story += [Paragraph("Raw results (lower is better; Seasonal Naive = 1.000)", h2), table(rows, [1.3 * inch] + [1.05 * inch] * 5)]

    rows = [["Technique", "Type"] + [f"{m}\nMASE" for m in models] + [f"{m}\nCRPS" for m in models]]
    for v, (label, kind) in VARIANTS.items():
        if v == "raw":
            continue
        sub = summary[summary.variant == v].set_index("model")
        rows.append([label, kind] + [pct(sub.loc[m, "d_mase_pct"]) for m in models]
                    + [pct(sub.loc[m, "d_crps_pct"]) for m in models])
    story += [
        Paragraph("Sensitivity: change vs raw in relative score (negative = better)", h2),
        table(rows, [1.45 * inch, 0.55 * inch] + [0.88 * inch] * 6),
        Spacer(1, 6),
        Image(str(FIG_DIR / "sensitivity_summary.png"), width=7.2 * inch, height=3.4 * inch),
        Paragraph(
            "Separate figures in results/figures: dataset_forecasts.png (each dataset with each model's "
            "10-90% prediction range), raw_relative_scores.png, raw_per_dataset_heatmap.png, "
            "sensitivity_summary.png, sensitivity_per_dataset.png, interval_coverage.png. "
            "Full numbers: results/summary.csv, results/relative_per_dataset.csv.", small),
    ]
    doc = SimpleDocTemplate(str(RESULTS / "key_takeaways.pdf"), pagesize=letter,
                            leftMargin=0.6 * inch, rightMargin=0.6 * inch, topMargin=0.5 * inch, bottomMargin=0.5 * inch,
                            title="TSFM evaluation: key takeaways")
    doc.build(story)


if __name__ == "__main__":
    main()
