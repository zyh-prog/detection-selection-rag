"""gen_all_figures.py — publication figures for the RIV detection-selection paper.

academic-plotting style: "Ocean Dusk" palette, ACL sizing, serif fonts, value
labels, vector PDF + 300 dpi PNG. Architecture diagram is code-based (full-width).
Data verbatim from verified caches (selection_ci/selector/by_level/grounding) and
logs/f01_grounding_results.md + logs/f02_situatedqa_results.md.
"""
import os
os.environ.setdefault("MPLBACKEND", "Agg")
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

OUT = Path(__file__).resolve().parent
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.labelsize": 10, "legend.fontsize": 8.5, "legend.frameon": False,
    "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.15, "lines.linewidth": 1.8, "lines.markersize": 5,
})

TEAL_D, TEAL, GOLD, ORANGE, CORAL = "#264653", "#2A9D8F", "#E9C46A", "#F4A261", "#E76F51"
BLUE, GRAY = "#0072B2", "#8C8C8C"
FIG_SINGLE, FIG_FULL = (3.3, 2.6), (6.8, 2.9)
# LNCS \linewidth is 347pt = 4.82in. Drawing a figure wider than the width it is
# included at shrinks every label by that ratio: fig1 was drawn 5.93in wide and shown
# at 0.6\linewidth = 2.89in, so its 7pt value labels reached the page at 3.4pt. These
# two are therefore drawn at the width they are displayed at, unchanged page footprint.
LW_IN = 4.82
# Both are now drawn at full/three-quarter column width but proportionally shorter, so
# the block of page they occupy is unchanged while every label keeps its nominal size.
FIG1_DISPLAY = (LW_IN, 1.45)
FIG2_DISPLAY = (0.75 * LW_IN, 1.30)


def save(name, pad=0.1):
    for ext in ("pdf", "png"):
        plt.savefig(OUT / f"{name}.{ext}", dpi=300, bbox_inches="tight", pad_inches=pad)
    plt.close()


def fig_architecture():
    # Redesigned (2026-06-12): semantically faithful to Sec. 4.
    #  - pool is produced ONLY by answer generation;
    #  - Direct / Perfect detection / Perfect selection form a rising LADDER
    #    read off the same pool, with the rung gaps labeled $\\Delta_det$/$\\Delta_sel$
    #    (same notation as Eq. 2 / Table 1);
    #  - Gold interpretation BYPASSES the pool (dashed path from the pipeline);
    #  - the two deployable selectors are a separate group, not children of
    #    any bound, noted as capped by perfect selection.
    # Designed at LNCS text width (scale ~1 when included at \\textwidth).
    fig = plt.figure(figsize=(4.8, 2.40))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 100); ax.set_ylim(0, 50); ax.axis("off")

    fit = []  # (artist, max_w_pt_units, ) handled via data box sizes

    def section(x, y, w, h, fc):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.2,rounding_size=1.2",
                                    linewidth=0, facecolor=fc, zorder=0))

    def box(x, y, w, h, main, sub=None, fc="white", ec="#CBD5E1", tc="#1F2937",
            fs=7.5, sfs=5.8, bold=False, dashed=False):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=1.1",
                                    linewidth=1.0, edgecolor=ec, facecolor=fc, zorder=2,
                                    linestyle=(0, (3, 2)) if dashed else "solid"))
        if sub:
            tm = ax.text(x + w/2, y + h/2 + 1.25, main, ha="center", va="center", fontsize=fs,
                         color=tc, zorder=3, weight=("bold" if bold else "normal"))
            ts = ax.text(x + w/2, y + h/2 - 1.55, sub, ha="center", va="center", fontsize=sfs,
                         color="#52525B", zorder=3)
            fit.append((tm, w, h/2)); fit.append((ts, w, h/2))
        else:
            tm = ax.text(x + w/2, y + h/2, main, ha="center", va="center", fontsize=fs,
                         color=tc, zorder=3, weight=("bold" if bold else "normal"), linespacing=1.25)
            fit.append((tm, w, h))

    def arrow(x1, y1, x2, y2, color="#6B7280", lw=1.0, ls="-", ms=8, sa=1.5, sb=2.0):
        # zorder above the boxes (z2): heads landing on a box edge stay visible.
        # sa/sb retract tail/head along the shaft so the endpoints clear the
        # 0.3-unit FancyBboxPatch pad (each box visually overruns its (x,y,w,h)).
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=ms,
                                     linewidth=lw, color=color, linestyle=ls, zorder=4,
                                     shrinkA=sa, shrinkB=sb))

    # ---- Band A: fixed pipeline ----
    section(0.5, 39.2, 99, 10.3, "#E8EDF2")
    ax.text(2.0, 47.4, "FIXED REVERSE-INTENT PIPELINE (held constant; one cached run)",
            fontsize=7, color="#475569", weight="bold")
    stages = ["Question", "Retrieval", "Interpretation\ngeneration (\u00d74)", "Answer\ngeneration"]
    xs = [2.5, 27, 51.5, 76]; w = 21.5; yA = 40.0; hA = 6.3
    for x, s in zip(xs, stages):
        box(x, yA, w, hA, s, fc="white", ec="#94A3B8", fs=7.5)
    for i in range(3):
        arrow(xs[i] + w, yA + hA/2, xs[i+1], yA + hA/2)

    # ---- Shared candidate pool (produced by answer generation only) ----
    # the shared control object: emphasized via bold label + a more saturated
    # fill/border (NOT a thicker line, which would eat the arrow-head gaps above/below)
    box(6, 31.0, 58, 5.2, "Candidate pool: direct + 4 interpretation answers",
        fc="#FBE6BE", ec="#C68A2E", fs=7.2, bold=True)
    # answer generation feeds the pool. This arrow is near-horizontal (~4 deg), so
    # the arrowHEAD's lower back-corner sits ~1.7 pt BELOW the tip; placing the tip
    # by gap alone overlaps the box. Raise the tip (37.45) and shrink the head
    # (ms=6) so the corner -- not just the tip -- clears the pool's top border.
    arrow(84, 39.32, 53, 37.45, ms=6, sa=0, sb=0)

    # ---- Gold interpretation: bypasses the pool ----
    box(72, 28.2, 27, 6.6, "Gold interpretation", "(gold sub-questions)",
        fc="#FBEAE3", ec=CORAL, fs=7.5, sfs=5.8, dashed=True)
    # bypass path into the gold box. SOLID coral (a dashed linestyle is applied to
    # the filled arrowHEAD too, shredding it into a non-arrow blob -- the box's own
    # dashed outline + the caption already carry the "bypass" meaning). 2026-06-18:
    # gold box lowered (30.2->28.2) to soak up the whitespace left by the removed
    # annotation; tip lowered to 35.85 so the head corner still clears the new top.
    arrow(88, 39.33, 94, 35.85, color=CORAL, ms=7, sa=0, sb=0)
    # (annotation removed 2026-06-18 to de-clutter: "generation-side bound; outside
    #  the detection-selection pair" duplicated the Fig.1 caption; the box's dashed
    #  outline + "(gold sub-questions)" sublabel + the caption already carry it.)

    # ---- Band B: ladder + deployable selectors ----
    section(0.5, 0.5, 99, 25.0, "#E8F2EE")
    ax.text(2.0, 23.2, "SYSTEMS THAT READ ONE SHARED POOL",
            fontsize=7, color="#2f6f5e", weight="bold")
    # pool feeds Band B: tail ~0.8 pt below the pool bottom border, head tip ~0.9 pt
    # above the Band-B panel top edge (panel has no border line, so gap the fill)
    arrow(35, 30.32, 35, 25.9, sa=0, sb=0)

    # rising ladder: Direct -> Perfect detection -> Perfect selection
    # 2026-06-18: whole ladder lowered ~1.5 (Direct 4.5->3.0, Perf-det 8.5->7.0,
    # Perf-sel 13->11.5) so Perf-sel's top drops to 18.5, opening room above it for
    # +Delta_sel below the band title (y23.2); the ceiling line drops to 18.5 too so
    # it stays tied to Perf-sel's top. Right-column selectors stay put.
    box(2, 3.0, 13, 7, "Direct", "(answer as-is)", fc="#FFFFFF", ec="#94A3B8", fs=7.5, sfs=5.6)
    box(18.5, 7.0, 21, 7, "Perfect detection", "(gold ambiguity label)",
        fc="#FDEBD0", ec=GOLD, fs=7, sfs=5.6)
    box(43, 11.5, 21, 7, "Perfect selection", "(best gold subset)",
        fc="#F8D7C4", ec=ORANGE, fs=7, sfs=5.6)
    # ladder rungs: tips kept ~0.7 pt off each box border (sa=sb=0, exact endpoints)
    # 2026-06-18: steepened both rungs so each arrowHEAD rises into the upper box's
    # upper-left corner (near its Delta label), keeping the label-to-arrow gap ~2;
    # heads follow the lowered boxes (Perf-det top 14.0, Perf-sel top 18.5).
    arrow(15.4, 6.5, 17.9, 13.4, color="#8a6d3b", lw=1.1, sa=0, sb=0)
    arrow(40.0, 10.5, 42.4, 17.9, color="#B45309", lw=1.1, sa=0, sb=0)
    # label boxes: det raised clear of the Perf-Det top edge; sel shifted left so its
    # subscript stays out of the Perf-Sel corner (x>=43) and below the band title (y<=22.4)
    # the two measured quantities are the figure's main characters: enlarged (fs10).
    # NOTE va='baseline' (default), so a fs10 label extends ~2 units ABOVE its y;
    # keep the baseline low enough that the cap-top clears the band title (y=23.2).
    # 2026-06-18: each Delta centered over its rung arrow, baseline ~1.6 above the
    # (lowered) upper box top so the subscript clears the box and the cap clears the
    # band title: Perf-det top 14.0 -> det at 15.7; Perf-sel top 18.5 -> sel at 20.2.
    ax.text(16.6, 15.7, "$+\\Delta_{\\mathrm{det}}$", fontsize=10, color="#8a6d3b", ha="center")
    ax.text(41.0, 20.2, "$+\\Delta_{\\mathrm{sel}}$", fontsize=10, color="#B45309", ha="center")
    # (annotation removed 2026-06-18 to de-clutter: "constructed upper bounds (gold
    #  information, evaluation only)" duplicated the caption; the box sublabels
    #  "(gold ...)" + the right group's "deployable selectors (no gold info)" convey it.)

    # divider between the oracle ladder (left) and the deployable selectors (right)
    ax.plot([65.5, 65.5], [2.5, 21.5], color="#9CA3AF", linewidth=0.8,
            linestyle=(0, (3, 2)), zorder=1)
    # ---- deployable selectors: STACKED, sitting under a perfect-selection ceiling ----
    # ceiling drawn at y=18.5 = the top of the (lowered) Perfect-selection box, so
    # "capped by perfect selection" is shown by shared height (not only stated in
    # prose); the vertical stack fills the right column so nothing is crammed below.
    ax.plot([67.5, 99], [18.5, 18.5], color=ORANGE, linewidth=1.1,
            linestyle=(0, (4, 2)), zorder=1)
    ax.text(83.25, 19.5, "perfect-selection ceiling", fontsize=6.8, color="#B45309",
            style="italic", ha="center")
    box(68, 11.5, 28, 6, "Self-verifier", "(prompt)", fc="#D6E9E2", ec=TEAL, fs=7, sfs=5.6, bold=True)
    box(68, 4.3, 28, 6, "NLI grounding", "(entailment)", fc="#C9E3D9", ec=TEAL_D, fs=7, sfs=5.6, bold=True)
    ax.text(83, 2.5, "deployable selectors (no gold info)", fontsize=6.8,
            color="#2f6f5e", style="italic", ha="center")

    # auto-fit: shrink any label that exceeds its allotted box area
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    for tt, bw, bh in fit:
        p0 = ax.transData.transform((0, 0)); p1 = ax.transData.transform((bw, bh))
        max_w = abs(p1[0] - p0[0]) * 0.88; max_h = abs(p1[1] - p0[1]) * 0.92
        ext = tt.get_window_extent(renderer=rend)
        s = min(max_w / ext.width, max_h / ext.height, 1.0)
        if s < 0.999:
            tt.set_fontsize(tt.get_fontsize() * s)
    save("fig_architecture", pad=0.02)


def fig1_detection_selection():
    labels = ["AmbigQA\n7B", "AmbigQA\nV3.2", "ASQA\n7B", "ASQA\nV3.2", "SituatedQA\n7B", "SituatedQA\nV3.2"]
    detection = np.array([4.27, 3.18, 5.20, 4.24, 4.66, 3.85])
    selection = np.array([11.73, 10.31, 12.03, 10.41, 8.55, 7.51])
    # 95% paired-bootstrap CIs (10k, seed 42; recomputed 2026-07-13 from the frozen caches)
    det_lo = detection - np.array([2.70, 1.70, 3.38, 2.41, 3.55, 2.72])
    det_hi = np.array([5.90, 4.72, 7.10, 6.11, 5.81, 5.03]) - detection
    sel_lo = selection - np.array([9.85, 8.62, 11.01, 9.40, 6.99, 6.09])
    sel_hi = np.array([13.73, 12.12, 13.07, 11.42, 10.34, 9.16]) - selection
    x = np.arange(len(labels)); width = 0.36
    fig, ax = plt.subplots(figsize=FIG1_DISPLAY)
    b1 = ax.bar(x - width / 2, detection, width, label="Detection gain", color=BLUE, edgecolor="white",
                linewidth=0.5, yerr=[det_lo, det_hi], capsize=2, ecolor="#444444", error_kw={"elinewidth": 0.8})
    b2 = ax.bar(x + width / 2, selection, width, label="Selection gap", color=CORAL, edgecolor="white",
                linewidth=0.5, yerr=[sel_lo, sel_hi], capsize=2, ecolor="#444444", error_kw={"elinewidth": 0.8})
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel(r"$\Delta$F1"); ax.set_xticks(x); ax.set_xticklabels(labels)
    # Six two-line category labels share a 4.8in axis, so they need a smaller face than
    # the 10pt default or adjacent dataset names touch; the legend goes outside the axes
    # so it cannot land on the tallest bar.
    ax.tick_params(axis="x", labelsize=7); ax.tick_params(axis="y", labelsize=8)
    ax.set_ylim(0, 14.5)
    ax.legend(ncol=2, fontsize=8, loc="lower left", bbox_to_anchor=(0, 1.0, 1, 0.14),
              mode="expand", borderaxespad=0)
    for bars, his in ((b1, det_hi), (b2, sel_hi)):
        for bar, hi in zip(bars, his):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + hi + 0.35,
                    f"{bar.get_height():.1f}", ha="center", va="bottom", fontsize=7, color="#444")
    save("fig1_detection_vs_selection")


def fig2_scale_ladder():
    models = ["Qwen2.5\n7B", "Qwen3\n14B", "Qwen3\n32B", "DeepSeek\nV3.2"]
    direct = np.array([41.45, 41.05, 41.57, 49.02]); perfect = np.array([43.02, 44.65, 45.68, 53.73])
    perf_sel = np.array([56.25, 57.90, 55.70, 65.37]); x = np.arange(len(models))
    fig, ax = plt.subplots(figsize=FIG2_DISPLAY)
    ax.plot(x, direct, marker="o", label="Direct", color=BLUE)
    ax.plot(x, perfect, marker="s", label="Perfect detection", color=TEAL)
    ax.plot(x, perf_sel, marker="^", label="Perfect selection", color=ORANGE)
    ax.set_ylabel("F1"); ax.set_xticks(x); ax.set_xticklabels(models); ax.set_ylim(35, 72)
    ax.tick_params(axis="x", labelsize=7.5); ax.tick_params(axis="y", labelsize=8)
    # outside the axes, as in fig1: the "perfect selection" trace runs through the
    # upper-left corner where an inset legend would sit
    ax.legend(ncol=3, fontsize=7.5, loc="lower left", bbox_to_anchor=(0, 1.0, 1, 0.14),
              mode="expand", borderaxespad=0)
    save("fig2_scale_ladder")


def fig3_two_selectors():
    labels = ["AmbigQA\n7B", "AmbigQA\nV3.2", "ASQA\n7B", "ASQA\nV3.2", "SituatedQA\n7B", "SituatedQA\nV3.2"]
    selfver = np.array([-3.1, -8.9, 37.5, 45.8, 23.7, 21.4])
    grounding = np.array([-3.6, -29.3, 20.0, 12.9, -3.3, -20.9])
    x = np.arange(len(labels)); width = 0.38
    fig, ax = plt.subplots(figsize=(7.3, 3.0))
    b1 = ax.bar(x - width / 2, selfver, width, label="Self-verifier (prompt)", color=TEAL, edgecolor="white", linewidth=0.5)
    b2 = ax.bar(x + width / 2, grounding, width, label="Evidence grounding (NLI)", color=ORANGE, edgecolor="white", linewidth=0.5)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Headroom realized (%)"); ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_ylim(-38, 60)
    # light dataset separators (no universal "both" claim: SituatedQA diverges)
    for xv in (1.5, 3.5):
        ax.axvline(xv, color="#ccc", linewidth=0.8, zorder=0)
    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + (1.6 if h >= 0 else -1.6),
                    f"{h:.0f}", ha="center", va=("bottom" if h >= 0 else "top"), fontsize=6.5, color="#444")
    ax.legend(ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1.01), frameon=False)
    save("fig3_two_selectors")


def fig4_by_level():
    levels = ["1", "2", "3", "4+"]
    direct = np.array([63.13, 36.42, 26.21, 20.02]); allcand = np.array([40.81, 38.28, 35.31, 35.24])
    perf_sel = np.array([75.04, 53.53, 46.03, 40.73]); x = np.arange(len(levels))
    fig, ax = plt.subplots(figsize=FIG_SINGLE)
    ax.plot(x, direct, marker="o", label="Direct", color=BLUE)
    ax.plot(x, allcand, marker="s", label="All candidates", color=CORAL)
    ax.plot(x, perf_sel, marker="^", label="Perfect selection", color=ORANGE)
    ax.set_xlabel("Number of answer groups"); ax.set_ylabel("F1")
    ax.set_xticks(x); ax.set_xticklabels(levels); ax.legend()
    save("fig4_ambigqa_by_level")


if __name__ == "__main__":
    fig_architecture(); fig1_detection_selection(); fig2_scale_ladder()
    fig3_two_selectors(); fig4_by_level()
    print("Saved figures to", OUT)
    for f in sorted(OUT.glob("*.pdf")):
        print("  ", f.name)
