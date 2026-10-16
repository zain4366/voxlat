"""Assemble every paper figure (F01-F40) and the main tables into one captioned PDF gallery.

    python scripts/experiments/make_gallery.py                     # results/figure_gallery.pdf
    python scripts/experiments/make_gallery.py --dpi 60 --out x.pdf # small preview

One A4 page per figure (portrait or landscape to fit), caption from scripts/experiments/catalog.py, producer
script and the source data files under it; then the paper tables. Matplotlib + Pillow only (core dependencies).
Missing figures get a placeholder page that says which command makes them.
"""

from __future__ import annotations

import argparse
import sys
import textwrap
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402
from PIL import Image  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
from catalog import ITEMS, PAPER_DESIGN, step  # noqa: E402

A4 = (8.27, 11.69)

#: columns shown in the gallery for each table (the CSVs hold all columns)
TABLE_VIEW = {
    "T05": ["family", "flow_lpm", "R_mK_W", "R_active_mK_W", "pump_power_W", "mass_kg", "min_margin"],
    "T06": ["layout", "method", "hv_50_median", "hv_100_median", "hv_200_median", "n_evals_final_median",
            "hv_final_median", "evals_to_reach_qnehvi_hv200_median"],
    "T07": ["layout", "design", "cap_W", "family", "R_baseline_mK_W", "P_baseline_W", "R_opt_mK_W", "P_opt_W",
            "R_change_pct"],
    "T08": ["layout", "design", "R_mK_W", "R_active_mK_W", "pump_power_W", "mass_kg", "min_margin", "rho_min",
            "rho_max", "w_max", "L_min_mm", "L_max_mm"],
    "T09": ["design", "geometry", "source", "voxel_mm", "min_wall_mm", "throat_mm", "connected", "pockets",
            "loose_solid", "all_pass", "iou_vs_numpy"],
    "T02": ["model", "split", "group", "R2_median", "MAPE_median", "cov95_median"],
    "T01": ["quantity", "w", "rho", "case", "p_observed", "err_raw_n32", "err_raw_n64", "err_R32_64"],
}


#: row filters for the gallery view (pandas query strings)
TABLE_FILTER = {"T07": "cap_W > 0.3 and cap_W < 0.31", "T05": "flow_lpm == 3.0 or family == 'B1'"}


def _load(path: Path, max_px: int) -> np.ndarray:
    im = Image.open(path)
    im = im.convert("RGBA") if im.mode in ("P", "LA") else im
    if max(im.size) > max_px:
        s = max_px / max(im.size)
        im = im.resize((max(1, int(im.width * s)), max(1, int(im.height * s))), Image.LANCZOS)
    return np.asarray(im)


def _caption_page_text(fig, it, y_top: float, width_chars: int) -> None:
    s = step(it.step)
    lines = textwrap.wrap(f"{it.id}. {it.caption}", width_chars)
    meta = [f"File: {it.path}", f"Made by: {s.script} (full run: {s.est_full})"]
    if it.extra:
        meta += textwrap.wrap("Data: " + ", ".join(it.extra), width_chars + 10)
    y = y_top
    for ln in lines:
        fig.text(0.07, y, ln, fontsize=9.5, va="top", family="DejaVu Sans")
        y -= 0.018
    y -= 0.006
    for ln in meta:
        fig.text(0.07, y, ln, fontsize=7.5, va="top", color="0.35", family="DejaVu Sans Mono")
        y -= 0.014


def figure_page(pdf: PdfPages, it, max_px: int) -> bool:
    p = ROOT / it.path
    if not p.exists():
        fig = plt.figure(figsize=A4)
        fig.text(0.5, 0.6, f"{it.id}: {it.path}\nnot found", ha="center", fontsize=14)
        fig.text(0.5, 0.5, f"python scripts/experiments/{it.id}_{it.name}.py", ha="center",
                 family="DejaVu Sans Mono", fontsize=9)
        pdf.savefig(fig)
        plt.close(fig)
        return False
    img = _load(p, max_px)
    h, w = img.shape[:2]
    landscape = w / h > 1.25
    size = (A4[1], A4[0]) if landscape else A4
    fig = plt.figure(figsize=size)
    cap_h = 0.20 if landscape else 0.16
    fig.text(0.07, 0.975, f"{it.id}  ·  {it.section}", fontsize=8, color="0.4", va="top")
    # image area: [left, bottom, width, height] in figure fractions, keep the aspect ratio
    box_w, box_h = 0.88, 1 - cap_h - 0.06
    page_ar = size[0] / size[1]
    img_ar = w / h
    if img_ar * box_h / page_ar > box_w:      # width-limited
        iw, ih = box_w, box_w / img_ar * page_ar
    else:
        iw, ih = box_h * img_ar / page_ar, box_h
    ax = fig.add_axes([(1 - iw) / 2, cap_h + (box_h - ih) / 2 + 0.01, iw, ih])
    ax.imshow(img, interpolation="none")
    ax.axis("off")
    _caption_page_text(fig, it, cap_h - 0.01, 150 if landscape else 105)
    pdf.savefig(fig)
    plt.close(fig)
    return True


def table_page(pdf: PdfPages, it) -> bool:
    p = ROOT / it.path
    if not p.exists():
        return False
    df = pd.read_csv(p)
    if it.id in TABLE_FILTER:
        df = df.query(TABLE_FILTER[it.id])
    cols = [c for c in TABLE_VIEW.get(it.id, list(df.columns)[:10]) if c in df.columns]
    df = df[cols]
    chunks = [df.iloc[i:i + 34] for i in range(0, len(df), 34)] or [df]
    for k, ch in enumerate(chunks):
        fig = plt.figure(figsize=(A4[1], A4[0]))
        fig.text(0.04, 0.96, f"{it.id}  ·  {it.section}" + (f"  (part {k + 1}/{len(chunks)})" if len(chunks) > 1
                                                          else ""), fontsize=8, color="0.4", va="top")
        for j, ln in enumerate(textwrap.wrap(f"{it.id}. {it.caption}", 170)):
            fig.text(0.04, 0.93 - 0.022 * j, ln, fontsize=9.5, va="top")
        fig.text(0.04, 0.04, f"Full table: {it.path} (+ .md). Made by scripts/experiments/tables.py {it.id}"
                 + (f". Rows shown: {TABLE_FILTER[it.id]}" if it.id in TABLE_FILTER else ""),
                 fontsize=7.5, color="0.35", family="DejaVu Sans Mono")

        def fmt(v):
            if isinstance(v, (float, np.floating)):
                return "" if not np.isfinite(v) else f"{v:.4g}"
            return str(v)[:48]
        ax = fig.add_axes([0.03, 0.08, 0.94, 0.78])
        ax.axis("off")
        tb = ax.table(cellText=[[fmt(v) for v in r] for r in ch.values], colLabels=cols, loc="upper center",
                      cellLoc="left", colLoc="left")
        tb.auto_set_font_size(False)
        tb.set_fontsize(6.5 if len(cols) > 8 else 7.5)
        tb.auto_set_column_width(list(range(len(cols))))
        for (r, _), cell in tb.get_celld().items():
            cell.set_linewidth(0.3)
            if r == 0:
                cell.set_facecolor("#e8e8e8")
        pdf.savefig(fig)
        plt.close(fig)
    return True


def title_page(pdf: PdfPages, n_fig: int, n_tab: int) -> None:
    fig = plt.figure(figsize=A4)
    fig.text(0.5, 0.80, "VoxLat — figure and table gallery", ha="center", fontsize=20, weight="bold")
    fig.text(0.5, 0.75, "Graded TPMS lattice cooling jacket: homogenization, finite-gap correction,\n"
                        "device model, multi-objective optimization, PicoGK export", ha="center", fontsize=11)
    fig.text(0.5, 0.69, f"{n_fig} figures, {n_tab} tables · built "
                        f"{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}", ha="center", fontsize=9,
             color="0.4")
    fig.text(0.5, 0.66, f"Manufacturable slots optimum: {PAPER_DESIGN}", ha="center", fontsize=8,
             family="DejaVu Sans Mono", color="0.4")
    y = 0.60
    sec = None
    for it in ITEMS:
        if it.section != sec:
            sec = it.section
            fig.text(0.10, y, sec, fontsize=9, weight="bold", va="top")
            y -= 0.015
        fig.text(0.13, y, f"{it.id}  {it.name.replace('_', ' ')}", fontsize=7.5, va="top")
        fig.text(0.55, y, it.path.split("/")[-1], fontsize=6.5, va="top", color="0.4", family="DejaVu Sans Mono")
        y -= 0.0115
        if y < 0.04:
            pdf.savefig(fig)
            plt.close(fig)
            fig = plt.figure(figsize=A4)
            y = 0.95
    fig.text(0.10, max(y - 0.02, 0.015), "Reproduce: python scripts/experiments/run_all.py --replot  (see README, "
                                          "'Reproduce the paper')", fontsize=7.5, color="0.35")
    pdf.savefig(fig)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(ROOT / "results" / "figure_gallery.pdf"))
    ap.add_argument("--dpi", type=int, default=150, help="image resolution on the page (default 150)")
    a = ap.parse_args(argv)
    max_px = int(a.dpi * 11.0)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    figs = [i for i in ITEMS if i.kind == "figure"]
    tabs = [i for i in ITEMS if i.kind == "table" and i.id in TABLE_VIEW]
    missing = []
    with PdfPages(out) as pdf:
        title_page(pdf, len(figs), len(tabs))
        for it in figs:
            if not figure_page(pdf, it, max_px):
                missing.append(it.id)
        for it in tabs:
            if not table_page(pdf, it):
                missing.append(it.id)
        d = pdf.infodict()
        d["Title"] = "VoxLat figure and table gallery"
        d["Author"] = "VoxLat"
    print(f"gallery: {out} ({out.stat().st_size / 1e6:.1f} MB); missing: {missing or 'none'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
