"""
================================================================================
Economic Feasibility Analysis of Smart Grid-Connected Rooftop PV Implementation
in a University Campus: Development of a Regional Policy Assessment Framework

Full analysis script
Authors: Thanapon Saengsuwan, Teerawat Kaewpia

Sections:
  1. Financial model (Section 2.4)      -> NPV, IRR, BCR, LCOE, payback
  2. Sensitivity / tornado analysis      -> Section 3.2, Table 4, Figure 3
  3. Cash-flow and NPV-trajectory plot   -> Section 3.1, Figure 2
  4. RPAF weighted scoring model         -> Section 2.5, Table 5, Figure 4
  5. Validation regression (45 projects) -> Section 3.4, Table 6, Figure 5

IMPORTANT — DATA INTEGRITY NOTE
--------------------------------------------------------------------------
Section 5 (validation regression) requires the authors' real project-level
dataset of 45 institutional PV projects (Appendix A, Table A2). Only the
country-level summary is available at the time this script was written, so
a SYNTHETIC placeholder dataset is used below purely so the script runs
end-to-end. This placeholder is clearly flagged (SYNTHETIC_DATA = True) and
MUST be replaced with the real project-level CSV before any of Section 5's
output is used in the manuscript. Do not report numbers derived from the
placeholder as real findings.

Similarly, Section 1's SELF_CONSUMPTION_RATIO (0.70) is a stated assumption,
not a measured value — see Section 4.4 (Limitations) of the manuscript. It
should be replaced with a ratio derived from actual metered campus load
data before the results are treated as final.
================================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm

# ==============================================================================
# SECTION 0 — GLOBAL PARAMETERS (Table 2)
# ==============================================================================

PARAMS = dict(
    capex=24_930_000,           # THB, total installed capital cost
    discount_rate=0.0631,       # Bank of Thailand govt bond yield + risk premium
    inflation=0.025,            # THB inflation, applied to tariff and O&M escalation
    degradation=0.008,          # annual PV output degradation, fraction/yr
    lifetime=25,                # years
    year1_output_kwh=738_776,   # Year-1 energy yield, kWh
    om_rate=0.05,                # THB/kWh, variable O&M
    staff_cost_y1=180_000,      # THB/yr, fixed staff cost
    carbon_price=200,           # THB/tCO2, T-VER
    grid_emission_factor=0.5692,  # tCO2/MWh, EGAT 2023
    tax_rate=0.0,                # effective tax rate (govt university, Rev. Code S3(1))
)

# Self-consumption revenue model — REPLACES the FiT-export model used in R1
# following Reviewer #2's comments (see response table, items 4 & 5).
SELF_CONSUMPTION_RATIO = 0.70   # ASSUMPTION — validate against real load data
RETAIL_TARIFF_Y1 = 4.75          # THB/kWh, midpoint of institutional tariff range 4.5-5.0
EXPORT_CREDIT = 0.0              # THB/kWh for surplus exported (conservative: no
                                  # guaranteed institutional export tariff under
                                  # current Thai regulation)


# ==============================================================================
# SECTION 1 — FINANCIAL MODEL (Section 2.4, Equations 1-5)
# ==============================================================================

def build_cashflow_model(params, sc_ratio, retail_tariff_y1, export_credit,
                          capex_mult=1.0, discount_rate=None, om_mult=1.0):
    """
    Builds the annual technical and financial cash-flow series for the project
    lifetime, following Equations (1)-(5) of Section 2.4.

    Returns a dict of numpy arrays (length = lifetime) plus scalar summary
    metrics: NPV, IRR, BCR, LCOE, simple payback, discounted payback.
    """
    n = params["lifetime"]
    r = discount_rate if discount_rate is not None else params["discount_rate"]
    years = np.arange(1, n + 1)

    # --- Technical output, Eq. basis for E_t in Eq. (4) ---
    output_kwh = params["year1_output_kwh"] * (1 - params["degradation"]) ** (years - 1)

    # --- Revenue: bill-offset (self-consumed) + export credit + carbon credit ---
    tariff = retail_tariff_y1 * (1 + params["inflation"]) ** (years - 1)
    self_consumed = output_kwh * sc_ratio
    exported = output_kwh * (1 - sc_ratio)

    bill_savings = self_consumed * tariff
    export_revenue = exported * export_credit
    carbon_offset_tco2 = (output_kwh / 1000) * params["grid_emission_factor"]
    carbon_revenue = carbon_offset_tco2 * params["carbon_price"]  # not escalated

    revenue = bill_savings + export_revenue + carbon_revenue

    # --- O&M cost (escalated at inflation) ---
    om_cost = (output_kwh * params["om_rate"] * (1 + params["inflation"]) ** (years - 1)
               + params["staff_cost_y1"] * (1 + params["inflation"]) ** (years - 1)) * om_mult

    net_cf = (revenue - om_cost) * (1 - params["tax_rate"])
    capex = params["capex"] * capex_mult
    cashflows = np.insert(net_cf, 0, -capex)

    # --- Eq. (1): NPV ---
    def npv_at(rate, cfs):
        return sum(cf / (1 + rate) ** i for i, cf in enumerate(cfs))

    NPV = npv_at(r, cashflows)

    # --- Eq. (2): IRR via bisection (robust, no external dependency needed) ---
    lo, hi = -0.5, 1.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if npv_at(mid, cashflows) > 0:
            lo = mid
        else:
            hi = mid
    IRR = (lo + hi) / 2

    # --- Eq. (3): BCR ---
    disc_costs = capex + sum(om_cost[i] / (1 + r) ** (i + 1) for i in range(n))
    disc_revenue = sum(revenue[i] / (1 + r) ** (i + 1) for i in range(n))
    BCR = disc_revenue / disc_costs

    # --- Eq. (4): LCOE ---
    disc_energy = sum(output_kwh[i] / (1 + r) ** (i + 1) for i in range(n))
    LCOE = disc_costs / disc_energy

    # --- Eq. (5): Simple and discounted payback ---
    cum_cf = np.cumsum(net_cf)
    simple_payback = float(np.interp(capex, cum_cf, years))

    disc_cf = np.array([net_cf[i] / (1 + r) ** (i + 1) for i in range(n)])
    cum_disc_cf = np.cumsum(disc_cf)
    discounted_payback = float(np.interp(capex, cum_disc_cf, years))

    return dict(
        years=years, output_kwh=output_kwh, revenue=revenue, om_cost=om_cost,
        net_cf=net_cf, cum_cf=cum_cf, cum_disc_cf=cum_disc_cf,
        NPV=NPV, IRR=IRR, BCR=BCR, LCOE=LCOE,
        simple_payback=simple_payback, discounted_payback=discounted_payback,
    )


def print_summary(result, label=""):
    print(f"--- {label} ---")
    print(f"NPV                 = {result['NPV']:>14,.0f} THB")
    print(f"IRR                 = {result['IRR']*100:>14.2f} %")
    print(f"BCR                 = {result['BCR']:>14.2f}")
    print(f"LCOE                = {result['LCOE']:>14.2f} THB/kWh")
    print(f"Simple payback      = {result['simple_payback']:>14.1f} yr")
    print(f"Discounted payback  = {result['discounted_payback']:>14.1f} yr")
    print(f"Year-1 revenue      = {result['revenue'][0]:>14,.0f} THB")
    print(f"Year-1 net cash flow= {result['net_cf'][0]:>14,.0f} THB")
    print()


# ==============================================================================
# SECTION 2 — SENSITIVITY / TORNADO ANALYSIS (Section 3.2, Table 4, Figure 3)
# ==============================================================================

def tornado_sensitivity(params, baseline_kwargs, variation=0.20):
    """
    Runs +/-`variation` sensitivity on five key parameters and returns a
    DataFrame sorted by NPV range (largest first), matching Figure 3.
    """
    base = build_cashflow_model(params, **baseline_kwargs)
    base_npv = base["NPV"]

    scenarios = {
        "Self-consumption ratio": ("sc_ratio", baseline_kwargs["sc_ratio"]),
        "Retail tariff":          ("retail_tariff_y1", baseline_kwargs["retail_tariff_y1"]),
        "Capital cost":           ("capex_mult", 1.0),
        "Discount rate":          ("discount_rate", params["discount_rate"]),
        "O&M cost":               ("om_mult", 1.0),
    }

    rows = []
    for name, (key, base_val) in scenarios.items():
        kwargs_lo = dict(baseline_kwargs)
        kwargs_hi = dict(baseline_kwargs)

        if key == "capex_mult":
            kwargs_lo["capex_mult"] = 1 + variation   # cost UP = capex up -> NPV down
            kwargs_hi["capex_mult"] = 1 - variation
        elif key == "om_mult":
            kwargs_lo["om_mult"] = 1 + variation
            kwargs_hi["om_mult"] = 1 - variation
        elif key == "discount_rate":
            kwargs_lo["discount_rate"] = base_val * (1 + variation)
            kwargs_hi["discount_rate"] = base_val * (1 - variation)
        elif key == "sc_ratio":
            kwargs_lo["sc_ratio"] = max(0.0, base_val * (1 - variation))
            kwargs_hi["sc_ratio"] = min(1.0, base_val * (1 + variation))
        elif key == "retail_tariff_y1":
            kwargs_lo["retail_tariff_y1"] = base_val * (1 - variation)
            kwargs_hi["retail_tariff_y1"] = base_val * (1 + variation)

        npv_lo = build_cashflow_model(params, **kwargs_lo)["NPV"]
        npv_hi = build_cashflow_model(params, **kwargs_hi)["NPV"]

        rows.append(dict(parameter=name, npv_negative=npv_lo, npv_positive=npv_hi,
                          range_thb=abs(npv_hi - npv_lo)))

    df = pd.DataFrame(rows).sort_values("range_thb", ascending=False).reset_index(drop=True)
    df["baseline_npv"] = base_npv
    return df


def plot_tornado(df, outpath):
    fig, ax = plt.subplots(figsize=(8, 5))
    baseline = df["baseline_npv"].iloc[0] / 1e6
    y_pos = np.arange(len(df))[::-1]

    for i, row in df.iterrows():
        lo = row["npv_negative"] / 1e6
        hi = row["npv_positive"] / 1e6
        ax.barh(y_pos[i], lo - baseline, left=baseline, color="firebrick", alpha=0.85)
        ax.barh(y_pos[i], hi - baseline, left=baseline, color="seagreen", alpha=0.85)
        ax.text(lo - 0.3, y_pos[i], f"{lo:.1f}", va="center", ha="right", fontsize=9)
        ax.text(hi + 0.3, y_pos[i], f"{hi:.1f}", va="center", ha="left", fontsize=9)

    ax.axvline(baseline, color="black", linewidth=1)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df["parameter"])
    ax.set_xlabel("Net Present Value (Million THB)")
    ax.set_title("Figure 3. NPV Sensitivity Tornado Diagram (±20% variation)")
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)


# ==============================================================================
# SECTION 3 — CASH FLOW / CUMULATIVE NPV PLOT (Section 3.1, Figure 2)
# ==============================================================================

def plot_cashflow_and_npv(result, outpath):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 9))

    years = result["years"]
    ax1.bar(years, result["revenue"] / 1e6, color="steelblue", label="Revenue (bill savings + carbon credit)")
    ax1.bar(years, result["om_cost"] / 1e6, color="darkorange", label="O&M cost")
    ax1.plot(years, result["net_cf"] / 1e6, "k--o", markersize=3, label="Net cash flow")
    ax1.axvline(result["simple_payback"], color="green", linestyle="--",
                label=f"Simple payback = {result['simple_payback']:.1f} yr")
    ax1.set_xlabel("Year")
    ax1.set_ylabel("Million THB")
    ax1.set_title("(a) Annual cash flow")
    ax1.legend(fontsize=8)

    ax2.fill_between(years, 0, result["cum_disc_cf"] / 1e6, color="lightblue", alpha=0.6)
    ax2.plot(years, result["cum_disc_cf"] / 1e6, "-o", color="steelblue", markersize=3,
             label="Cumulative discounted NPV")
    ax2.axvline(result["discounted_payback"], color="darkred", linestyle="--",
                label=f"Discounted payback = {result['discounted_payback']:.1f} yr")
    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.set_xlabel("Year")
    ax2.set_ylabel("Cumulative Discounted NPV (Million THB)")
    ax2.set_title("(b) Cumulative discounted NPV")
    ax2.legend(fontsize=8)

    fig.suptitle("Figure 2. Annual cash flow and cumulative discounted NPV\n"
                 "(Self-consumption bill-offset model)")
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)


# ==============================================================================
# SECTION 4 — RPAF WEIGHTED SCORING MODEL (Section 2.5, Table 5, Figure 4)
# ==============================================================================

RPAF_WEIGHTS = dict(
    policy_stability=0.30,
    financial_incentives=0.25,
    regulatory_environment=0.20,
    market_maturity=0.15,
    implementation_barriers=0.10,
)

def compute_rpaf_score(row, weights=RPAF_WEIGHTS):
    """row: dict/Series with keys matching RPAF_WEIGHTS, each 0-100."""
    return sum(row[dim] * w for dim, w in weights.items())


def build_rpaf_table():
    """Table 5 — RPAF Scoring Matrix, six ASEAN countries (2023 data)."""
    data = {
        "Thailand":    dict(policy_stability=85, financial_incentives=78, regulatory_environment=72, market_maturity=80, implementation_barriers=75),
        "Singapore":   dict(policy_stability=90, financial_incentives=65, regulatory_environment=85, market_maturity=85, implementation_barriers=90),
        "Malaysia":    dict(policy_stability=75, financial_incentives=70, regulatory_environment=68, market_maturity=75, implementation_barriers=80),
        "Vietnam":     dict(policy_stability=70, financial_incentives=85, regulatory_environment=60, market_maturity=70, implementation_barriers=65),
        "Philippines": dict(policy_stability=65, financial_incentives=75, regulatory_environment=65, market_maturity=65, implementation_barriers=60),
        "Indonesia":   dict(policy_stability=60, financial_incentives=60, regulatory_environment=55, market_maturity=60, implementation_barriers=55),
    }
    df = pd.DataFrame(data).T
    df["overall_score"] = df.apply(compute_rpaf_score, axis=1)
    df = df.sort_values("overall_score", ascending=False)
    return df


def plot_rpaf_radar(df, outpath):
    categories = ["policy_stability", "financial_incentives", "regulatory_environment",
                  "market_maturity", "implementation_barriers"]
    labels = ["Policy Stability\n(30%)", "Financial Incentives\n(25%)",
              "Regulatory Environment\n(20%)", "Market Maturity\n(15%)",
              "Implementation Barriers\n(10%)"]
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    for country, row in df.iterrows():
        values = [row[c] for c in categories]
        values += values[:1]
        ax.plot(angles, values, label=f"{country} ({row['overall_score']:.1f})")
        ax.fill(angles, values, alpha=0.05)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylim(0, 100)
    ax.set_title("Figure 4. RPAF Radar Chart — Six ASEAN Countries (2023)")
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=8)
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)


# ==============================================================================
# SECTION 5 — VALIDATION REGRESSION (Section 3.4, Table 6, Figure 5)
# ==============================================================================

SYNTHETIC_DATA = True   # <-- SET TO FALSE once the real 45-project CSV is available

def load_validation_dataset(csv_path="validation_dataset.csv"):
    """
    Expects a CSV with columns: country, rpaf_score, irr_pct
    (one row per project, 45 rows total; see Appendix A, Table A2).
    """
    if not SYNTHETIC_DATA:
        return pd.read_csv(csv_path)

    # -------------------------------------------------------------------
    # SYNTHETIC PLACEHOLDER — for script-testing purposes ONLY.
    # Generated to sit within the published per-country RPAF score and
    # IRR ranges (Appendix A, Table A2) so the regression pipeline below
    # can be demonstrated. This is NOT the authors' real dataset and
    # MUST NOT be cited as a manuscript result.
    # -------------------------------------------------------------------
    rng = np.random.default_rng(42)
    country_specs = [
        ("Thailand",    8, 78.9, 8.7, 10.1),
        ("Singapore",   6, 82.0, 9.2, 10.8),
        ("Malaysia",    8, 72.8, 7.5, 9.4),
        ("Vietnam",     7, 71.2, 6.8, 11.2),
        ("Philippines", 9, 67.0, 6.8, 8.8),
        ("Indonesia",   7, 58.5, 5.5, 7.8),
    ]
    rows = []
    for country, n, rpaf, irr_lo, irr_hi in country_specs:
        irr_vals = rng.uniform(irr_lo, irr_hi, size=n)
        for irr in irr_vals:
            rows.append(dict(country=country, rpaf_score=rpaf, irr_pct=irr))
    return pd.DataFrame(rows)


def run_validation_regression(df):
    """Table 6 — OLS regression: Actual Project IRR (%) on RPAF Score."""
    x = df["rpaf_score"].values
    y = df["irr_pct"].values

    pearson_r, pearson_p = stats.pearsonr(x, y)
    spearman_rho, spearman_p = stats.spearmanr(x, y)

    X = sm.add_constant(x)
    model = sm.OLS(y, X).fit()

    return dict(model=model, pearson_r=pearson_r, pearson_p=pearson_p,
                spearman_rho=spearman_rho, spearman_p=spearman_p)


def plot_validation_scatter(df, reg_result, outpath):
    fig, ax = plt.subplots(figsize=(8, 6))
    countries = df["country"].unique()
    colors = plt.cm.tab10(np.linspace(0, 1, len(countries)))

    for c, color in zip(countries, colors):
        sub = df[df["country"] == c]
        ax.scatter(sub["rpaf_score"], sub["irr_pct"], label=f"{c} (n={len(sub)})",
                   color=color, s=40)

    x_line = np.linspace(df["rpaf_score"].min() - 2, df["rpaf_score"].max() + 2, 100)
    b0, b1 = reg_result["model"].params
    ax.plot(x_line, b0 + b1 * x_line, "darkred",
            label=f"OLS: IRR = {b1:.3f}·RPAF + {b0:.2f}")

    pred = reg_result["model"].get_prediction(sm.add_constant(x_line)).summary_frame(alpha=0.05)
    ax.fill_between(x_line, pred["mean_ci_lower"], pred["mean_ci_upper"],
                     color="lightpink", alpha=0.3, label="95% CI")

    stats_text = (f"n = {len(df)}\n"
                   f"Pearson r = {reg_result['pearson_r']:.3f} (p<0.001)\n"
                   f"Spearman ρ = {reg_result['spearman_rho']:.3f}\n"
                   f"R² = {reg_result['model'].rsquared:.3f}")
    ax.text(0.03, 0.97, stats_text, transform=ax.transAxes, va="top", fontsize=9,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    ax.set_xlabel("RPAF Score (0-100)")
    ax.set_ylabel("Actual Project IRR (%)")
    ax.set_title("Figure 5. Framework Validation Scatter Plot"
                 + (" [SYNTHETIC DATA - DO NOT CITE]" if SYNTHETIC_DATA else ""))
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)


# ==============================================================================
# MAIN — reproduces all manuscript tables/figures in sequence
# ==============================================================================

if __name__ == "__main__":

    print("=" * 78)
    print("SECTION 1-3: FINANCIAL MODEL, SENSITIVITY, AND CASH-FLOW PLOT")
    print("=" * 78)

    baseline_kwargs = dict(
        sc_ratio=SELF_CONSUMPTION_RATIO,
        retail_tariff_y1=RETAIL_TARIFF_Y1,
        export_credit=EXPORT_CREDIT,
    )

    baseline_result = build_cashflow_model(PARAMS, **baseline_kwargs)
    print_summary(baseline_result, label="Baseline (Table 3)")

    # Table 4 alternative scenarios
    upper_bound = build_cashflow_model(PARAMS, sc_ratio=1.00, retail_tariff_y1=4.75, export_credit=0.0)
    print_summary(upper_bound, label="Upper bound: 100% self-consumption")

    conservative = build_cashflow_model(PARAMS, sc_ratio=0.50, retail_tariff_y1=4.50, export_credit=0.0)
    print_summary(conservative, label="Conservative: 50% self-consumption, lower tariff")

    tornado_df = tornado_sensitivity(PARAMS, baseline_kwargs)
    print("Table 4 / Figure 3 — Tornado sensitivity ranking:")
    print(tornado_df.to_string(index=False))
    print()

    plot_cashflow_and_npv(baseline_result, "/mnt/user-data/outputs/figure2_cashflow_npv.png")
    plot_tornado(tornado_df, "/mnt/user-data/outputs/figure3_tornado.png")

    print("=" * 78)
    print("SECTION 4: RPAF SCORING MODEL")
    print("=" * 78)

    rpaf_df = build_rpaf_table()
    print("Table 5 — RPAF Scoring Matrix:")
    print(rpaf_df.round(1).to_string())
    print()
    plot_rpaf_radar(rpaf_df, "/mnt/user-data/outputs/figure4_rpaf_radar.png")

    print("=" * 78)
    print("SECTION 5: VALIDATION REGRESSION")
    print("=" * 78)
    if SYNTHETIC_DATA:
        print(">>> WARNING: SYNTHETIC_DATA = True. Results below are for script")
        print(">>> testing only and MUST NOT be cited as manuscript findings.")
        print(">>> Replace load_validation_dataset() with the real 45-project CSV.\n")

    validation_df = load_validation_dataset()
    reg_result = run_validation_regression(validation_df)

    print(reg_result["model"].summary())
    print(f"\nPearson r  = {reg_result['pearson_r']:.3f} (p = {reg_result['pearson_p']:.4f})")
    print(f"Spearman ρ = {reg_result['spearman_rho']:.3f} (p = {reg_result['spearman_p']:.4f})")

    plot_validation_scatter(validation_df, reg_result, "/mnt/user-data/outputs/figure5_validation_scatter.png")

    print("\nAll figures saved to /mnt/user-data/outputs/")
