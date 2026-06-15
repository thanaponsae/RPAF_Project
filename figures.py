"""
figures.py
==========
Generate all manuscript figures (Figures 1–5, Table A1).
Requires: results/*.json (run financial_model, rpaf_model, validation_analysis first)
Output:   figures/Fig1_*.png ... figures/Fig5_*.png

Run:  python src/figures.py
"""
import os, sys, json
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

sys.path.insert(0, os.path.dirname(__file__))
from parameters import COUNTRY_ORDER, DIMS, RPAF_WEIGHTS, DISCOUNT_RATE

matplotlib.rcParams.update({
    "font.family": "DejaVu Serif",
    "font.size":   10,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "figure.dpi":        150,
})

BLUE   = "#1a5276"
ORANGE = "#d35400"
GREEN  = "#1e8449"
RED    = "#922b21"
GREY   = "#7f8c8d"
PURPLE = "#8e44ad"
LTBLUE = "#aed6f1"
LTOR   = "#f0b27a"
COLORS = [BLUE, GREEN, ORANGE, RED, GREY, PURPLE]

os.makedirs("figures", exist_ok=True)


def load_results():
    with open("results/financial_results.json")  as f: fin = json.load(f)
    with open("results/rpaf_results.json")        as f: rpf = json.load(f)
    with open("results/validation_results.json")  as f: val = json.load(f)
    return fin, rpf, val


# ── FIGURE 1: Cash Flow + Cumulative NPV ────────────────────────────────────
def fig1_cashflow(fin):
    cfs    = np.array(fin["cashflows"])
    rows   = fin["components"]
    rv     = np.array([r["revenue"]  for r in rows])
    om     = np.array([r["om_cost"]  for r in rows])
    spb    = fin["baseline"]["spb_yr"]
    dpb    = fin["baseline"]["dpb_yr"]
    cum    = np.array(fin["cum_disc_npv"])
    years  = np.arange(0, 26)
    yr25   = np.arange(1, 26)
    w      = 0.35

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), tight_layout=True)

    ax1.bar(yr25 - w/2, rv/1e6, w, color=BLUE,   alpha=0.85,
            label="Revenue (FiT + Carbon)")
    ax1.bar(yr25 + w/2, om/1e6, w, color=ORANGE, alpha=0.85,
            label="O&M Cost")
    ax1.plot(yr25, cfs[1:]/1e6, "k--", lw=1.5, marker="o", ms=3,
             label="Net Cash Flow")
    ax1.axhline(0, color="black", lw=0.7)
    ax1.axvline(spb, color=GREEN, lw=1.5, ls="--",
                label=f"Simple Payback ({spb:.1f} yr)")
    ax1.set_xlabel("Year"); ax1.set_ylabel("Million THB")
    ax1.set_title("(a)  Annual Cash Flow — 500 kWp PV System",
                  fontweight="bold", loc="left")
    ax1.legend(fontsize=8, ncol=2)
    ax1.set_xlim(0.5, 25.5)

    ax2.fill_between(years, cum/1e6, 0,
                     where=cum >= 0, color=LTBLUE,  alpha=0.7,
                     label="Positive region")
    ax2.fill_between(years, cum/1e6, 0,
                     where=cum < 0,  color=LTOR,    alpha=0.7,
                     label="Negative region")
    ax2.plot(years, cum/1e6, color=BLUE, lw=2, marker="o", ms=3,
             label="Cumulative Discounted NPV")
    ax2.axhline(0,   color="black", lw=0.7)
    ax2.axvline(dpb, color=RED, lw=1.5, ls="--",
                label=f"Discounted Payback ({dpb:.1f} yr)")
    final = cum[-1]
    ax2.annotate(f"Final NPV\n{final/1e6:.1f} M THB",
                 xy=(25, final/1e6), xytext=(19, final/1e6 - 4),
                 arrowprops=dict(arrowstyle="->", color=BLUE),
                 fontsize=8, color=BLUE)
    ax2.set_xlabel("Year"); ax2.set_ylabel("Cumulative Disc. NPV (M THB)")
    ax2.set_title("(b)  Cumulative Discounted NPV Progression",
                  fontweight="bold", loc="left")
    ax2.legend(fontsize=8, ncol=2)
    ax2.set_xlim(0, 25)

    fig.savefig("figures/Fig1_CashFlow.png", dpi=180, bbox_inches="tight")
    plt.close()
    print("[Saved] figures/Fig1_CashFlow.png")


# ── FIGURE 2: Tornado ────────────────────────────────────────────────────────
def fig2_tornado(fin):
    tor   = fin["tornado"]
    base  = fin["baseline"]["npv_M"]
    vars_ = [t["variable"]  for t in tor]
    lo_   = [t["npv_low"]   for t in tor]
    hi_   = [t["npv_high"]  for t in tor]

    fig, ax = plt.subplots(figsize=(9, 5), tight_layout=True)
    yp = np.arange(len(vars_))
    for i, (lo, hi) in enumerate(zip(lo_, hi_)):
        ax.barh(i, lo - base, left=base, color=RED,   alpha=0.82, height=0.55)
        ax.barh(i, hi - base, left=base, color=GREEN, alpha=0.82, height=0.55)
        ax.text(lo - 0.12, i, f"{lo:.1f}", va="center", ha="right",  fontsize=8.5)
        ax.text(hi + 0.12, i, f"{hi:.1f}", va="center", ha="left",   fontsize=8.5)
    ax.axvline(base, color="black", lw=2)
    ax.text(base + 0.1, len(vars_) - 0.45,
            f"Baseline\n{base:.2f} M THB", fontsize=8)
    ax.set_yticks(yp); ax.set_yticklabels(vars_, fontsize=9)
    ax.set_xlabel("NPV (Million THB)")
    ax.set_title("Sensitivity Analysis — NPV Tornado Diagram (±20% variation)",
                 fontweight="bold")
    ax.legend(handles=[
        mpatches.Patch(color=RED,   alpha=0.82, label="Negative impact (−20%)"),
        mpatches.Patch(color=GREEN, alpha=0.82, label="Positive impact (+20%)"),
    ], fontsize=9, loc="lower right")

    fig.savefig("figures/Fig2_Tornado.png", dpi=180, bbox_inches="tight")
    plt.close()
    print("[Saved] figures/Fig2_Tornado.png")


# ── FIGURE 3: RPAF Radar ────────────────────────────────────────────────────
def fig3_radar(rpf):
    N      = len(DIMS)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += [angles[0]]
    labels = [
        "Policy\nStability\n(30%)", "Financial\nIncentives\n(25%)",
        "Regulatory\nEnvironment\n(20%)", "Market\nMaturity\n(15%)",
        "Implementation\nBarriers\n(10%)",
    ]
    scores = rpf["scores"]

    fig, ax = plt.subplots(figsize=(8, 8),
                           subplot_kw=dict(polar=True), tight_layout=True)
    for i, (country, col) in enumerate(zip(COUNTRY_ORDER, COLORS)):
        vals = [scores[country][d] for d in DIMS] + [scores[country][DIMS[0]]]
        ov   = scores[country]["overall_score"]
        lw   = 2.5 if country == "Thailand" else 1.5
        ax.plot(angles, vals, color=col, lw=lw, marker="o", ms=5,
                label=f"{country} ({ov:.1f})")
        ax.fill(angles, vals, color=col,
                alpha=0.12 if country == "Thailand" else 0.04)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(["20","40","60","80","100"], fontsize=7)
    ax.set_ylim(0, 100)
    ax.set_title("RPAF Scores by Dimension — Six ASEAN Countries (2023)",
                 fontweight="bold", pad=22)
    ax.legend(loc="lower left", bbox_to_anchor=(1.05, 0.0), fontsize=9)

    fig.savefig("figures/Fig3_Radar.png", dpi=180, bbox_inches="tight")
    plt.close()
    print("[Saved] figures/Fig3_Radar.png")


# ── FIGURE 4: Validation Scatter ─────────────────────────────────────────────
def fig4_scatter(val):
    reg   = val["regression"]
    pr    = val["pearson"]
    sr    = val["spearman"]
    x_raw = np.array(val["raw_x"])
    y_raw = np.array(val["raw_y"])
    ctry  = val["countries"]
    line  = val["scatter_line"]
    xline = np.array(line["x"])
    yline = np.array(line["y"])
    ci_lo = np.array(line["ci_lo"])
    ci_hi = np.array(line["ci_hi"])

    country_color = dict(zip(COUNTRY_ORDER, COLORS))

    fig, ax = plt.subplots(figsize=(9, 6), tight_layout=True)
    for country, col in country_color.items():
        mask = [c == country for c in ctry]
        xs   = x_raw[mask]; ys = y_raw[mask]
        ax.scatter(xs, ys, color=col, s=65, alpha=0.85, zorder=3,
                   label=f"{country} (n={mask.count(True)})")

    ax.plot(xline, yline, color="red", lw=2,
            label=(f"IRR = {reg['slope']:.3f}·RPAF "
                   f"{'+' if reg['intercept']>=0 else ''}{reg['intercept']:.2f}"))
    ax.fill_between(xline, ci_lo, ci_hi, color="red", alpha=0.12,
                    label="95% Confidence Interval")

    stats_txt = (f"n = {reg['n']}\n"
                 f"Pearson r = {pr['r']:.3f}  (p < 0.001)\n"
                 f"Spearman ρ = {sr['r']:.3f}  (p < 0.001)\n"
                 f"R² = {reg['r2']:.3f}   Adj.R² = {reg['adj_r2']:.3f}\n"
                 f"F(1,{reg['df']}) = {reg['F_stat']:.2f}")
    ax.text(0.03, 0.97, stats_txt, transform=ax.transAxes,
            fontsize=8.5, va="top",
            bbox=dict(boxstyle="round,pad=0.4", fc="white",
                      ec=GREY, alpha=0.92))
    ax.axvline(75, color=GREY, lw=1, ls=":", alpha=0.7)
    ax.text(75.3, 5.2, "RPAF = 75\n(high-attractiveness\nthreshold)",
            fontsize=7.5, color=GREY)
    ax.set_xlabel("RPAF Score (0–100)", fontsize=11)
    ax.set_ylabel("Actual Project IRR (%)", fontsize=11)
    ax.set_title(
        "Framework Validation: RPAF Score vs. Actual IRR\n"
        "(45 Institutional PV Projects, ASEAN 2019–2024; "
        "secondary data synthesis from published literature)",
        fontweight="bold")
    ax.legend(fontsize=9, loc="lower right")
    ax.set_xlim(54, 87); ax.set_ylim(4.5, 12.5)

    fig.savefig("figures/Fig4_Validation.png", dpi=180, bbox_inches="tight")
    plt.close()
    print("[Saved] figures/Fig4_Validation.png")


# ── FIGURE 5: RPAF Bar + Ranking ─────────────────────────────────────────────
def fig5_rpaf_bar(rpf):
    scores_mat = np.array([[rpf["scores"][c][d] for d in DIMS]
                            for c in COUNTRY_ORDER])
    overall    = [rpf["scores"][c]["overall_score"] for c in COUNTRY_ORDER]
    dim_short  = ["Policy\nStability","Financial\nIncentives",
                  "Regulatory\nEnv.","Market\nMaturity","Impl.\nBarriers"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), tight_layout=True)

    x  = np.arange(len(DIMS))
    ww = 0.13
    for i, (country, col) in enumerate(zip(COUNTRY_ORDER, COLORS)):
        ax1.bar(x + i*ww, scores_mat[i], ww, color=col, alpha=0.85,
                label=country)
    ax1.set_xticks(x + ww*2.5)
    ax1.set_xticklabels(dim_short, fontsize=9)
    ax1.set_ylabel("Score (0–100)"); ax1.set_ylim(0, 108)
    ax1.set_title("(a)  RPAF Dimensional Scores by Country",
                  fontweight="bold", loc="left")
    ax1.legend(fontsize=8, ncol=2, loc="lower right")
    ax1.axhline(70, color=GREY, lw=1, ls="--", alpha=0.6)

    sid  = np.argsort(overall)[::-1]
    cs   = [COUNTRY_ORDER[i] for i in sid]
    ss   = [overall[i] for i in sid]
    cols = [COLORS[i] for i in sid]
    yp   = np.arange(len(cs))
    bars = ax2.barh(yp, ss, color=cols, alpha=0.85, height=0.6)
    for bar, score in zip(bars, ss):
        ax2.text(bar.get_width() + 0.4, bar.get_y() + bar.get_height()/2,
                 f"{score:.1f}", va="center", ha="left",
                 fontsize=9, fontweight="bold")
    ax2.set_yticks(yp); ax2.set_yticklabels(cs, fontsize=10)
    ax2.set_xlabel("Overall RPAF Score (0–100)")
    ax2.set_title("(b)  Overall RPAF Rankings", fontweight="bold", loc="left")
    ax2.set_xlim(0, 96)
    ax2.axvline(75, color=GREEN,  lw=1.5, ls="--", alpha=0.8)
    ax2.axvline(65, color=ORANGE, lw=1.5, ls="--", alpha=0.8)
    ax2.text(75.3, 5.4, "High (≥75)",     fontsize=7, color=GREEN)
    ax2.text(65.3, 5.4, "Moderate (≥65)", fontsize=7, color=ORANGE)

    fig.savefig("figures/Fig5_RPAF_Bar.png", dpi=180, bbox_inches="tight")
    plt.close()
    print("[Saved] figures/Fig5_RPAF_Bar.png")


def main():
    fin, rpf, val = load_results()
    fig1_cashflow(fin)
    fig2_tornado(fin)
    fig3_radar(rpf)
    fig4_scatter(val)
    fig5_rpaf_bar(rpf)
    print("\nAll figures saved to figures/")


if __name__ == "__main__":
    main()
