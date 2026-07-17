"""
================================================================================
RECHECK SCRIPT — Independent recomputation of all statistical/financial
results reported in the manuscript, for cross-validation purposes.

This script does NOT reuse any previously saved numbers. Every value is
recalculated from the stated parameters (Section 2.4/Table 2) using the
same formulas, and then compared against what is currently written in the
manuscript. Any mismatch is flagged explicitly (not silently rounded away).
================================================================================
"""

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm

TOL_THB = 5_000        # THB tolerance for NPV-type comparisons (rounding)
TOL_PCT = 0.02          # percentage-point tolerance for IRR/BCR/LCOE etc.
TOL_YR = 0.05           # year tolerance for payback periods

# ==============================================================================
# 1. RECOMPUTE THE BASELINE FINANCIAL MODEL FROM SCRATCH
# ==============================================================================

capex = 24_930_000
discount_rate = 0.0631
inflation = 0.025
degradation = 0.008
lifetime = 25
year1_output_kwh = 738_776
om_rate = 0.05
staff_cost_y1 = 180_000
carbon_price = 200
grid_ef = 0.5692
self_consumption_ratio = 0.70
retail_tariff_y1 = 4.75
export_credit = 0.0

years = np.arange(1, lifetime + 1)
output_kwh = year1_output_kwh * (1 - degradation) ** (years - 1)
tariff = retail_tariff_y1 * (1 + inflation) ** (years - 1)

self_consumed = output_kwh * self_consumption_ratio
exported = output_kwh * (1 - self_consumption_ratio)
bill_savings = self_consumed * tariff
export_revenue = exported * export_credit
carbon_offset_tco2 = (output_kwh / 1000) * grid_ef
carbon_revenue = carbon_offset_tco2 * carbon_price

revenue = bill_savings + export_revenue + carbon_revenue
om_cost = (output_kwh * om_rate * (1 + inflation) ** (years - 1)
           + staff_cost_y1 * (1 + inflation) ** (years - 1))
net_cf = revenue - om_cost
cashflows = np.insert(net_cf, 0, -capex)


def npv_at(rate, cfs):
    return sum(cf / (1 + rate) ** i for i, cf in enumerate(cfs))


NPV = npv_at(discount_rate, cashflows)

lo, hi = -0.5, 1.0
for _ in range(300):
    mid = (lo + hi) / 2
    if npv_at(mid, cashflows) > 0:
        lo = mid
    else:
        hi = mid
IRR = (lo + hi) / 2

disc_costs = capex + sum(om_cost[i] / (1 + discount_rate) ** (i + 1) for i in range(lifetime))
disc_revenue = sum(revenue[i] / (1 + discount_rate) ** (i + 1) for i in range(lifetime))
BCR = disc_revenue / disc_costs

disc_energy = sum(output_kwh[i] / (1 + discount_rate) ** (i + 1) for i in range(lifetime))
LCOE = disc_costs / disc_energy

cum_cf = np.cumsum(net_cf)
simple_payback = float(np.interp(capex, cum_cf, years))

disc_cf = np.array([net_cf[i] / (1 + discount_rate) ** (i + 1) for i in range(lifetime)])
cum_disc_cf = np.cumsum(disc_cf)
discounted_payback = float(np.interp(capex, cum_disc_cf, years))
final_cum_npv = cum_disc_cf[-1]

recomputed = dict(
    NPV=NPV, IRR=IRR * 100, BCR=BCR, LCOE=LCOE,
    simple_payback=simple_payback, discounted_payback=discounted_payback,
    year1_revenue=revenue[0], year1_net_cf=net_cf[0], final_cum_npv=final_cum_npv,
)

# Values currently stated in the manuscript (Table 3 / Section 3.1)
manuscript_stated = dict(
    NPV=8_310_000, IRR=9.42, BCR=1.29, LCOE=3.31,
    simple_payback=10.0, discounted_payback=15.6,
    year1_revenue=2_540_000, year1_net_cf=2_320_000, final_cum_npv=None,  # not stated precisely
)

print("=" * 78)
print("1. BASELINE FINANCIAL MODEL — RECOMPUTED vs MANUSCRIPT")
print("=" * 78)
print(f"{'Metric':<20}{'Recomputed':>16}{'Manuscript':>16}{'Match?':>10}")
for key in ["NPV", "IRR", "BCR", "LCOE", "simple_payback", "discounted_payback",
            "year1_revenue", "year1_net_cf"]:
    rv = recomputed[key]
    mv = manuscript_stated[key]
    if key in ("IRR", "BCR", "LCOE"):
        ok = abs(rv - mv) <= TOL_PCT
    elif key in ("simple_payback", "discounted_payback"):
        ok = abs(rv - mv) <= TOL_YR
    else:
        ok = abs(rv - mv) <= TOL_THB
    flag = "OK" if ok else "!! MISMATCH"
    print(f"{key:<20}{rv:>16,.3f}{mv:>16,.3f}{flag:>12}")

print(f"\nFinal cumulative discounted NPV (yr 25, recomputed) = {final_cum_npv:,.0f} THB")
print("(Manuscript text states 'approximately 33.2 million THB' in the Fig. 2 caption -- "
      f"recomputed value differs by {abs(final_cum_npv-33_200_000):,.0f} THB; treat the "
      "manuscript's caption figure as approximate/needs updating if this recheck disagrees.)")


# ==============================================================================
# 2. RECOMPUTE ALTERNATIVE SCENARIOS (Table 4)
# ==============================================================================

def run_scenario(sc_ratio, tariff_y1, export_cr):
    t = tariff_y1 * (1 + inflation) ** (years - 1)
    sc = output_kwh * sc_ratio
    ex = output_kwh * (1 - sc_ratio)
    bill = sc * t
    exp_rev = ex * export_cr
    carbon = (output_kwh / 1000 * grid_ef) * carbon_price
    om = (output_kwh * om_rate * (1 + inflation) ** (years - 1)
          + staff_cost_y1 * (1 + inflation) ** (years - 1))
    rev = bill + exp_rev + carbon
    ncf = rev - om
    cfs = np.insert(ncf, 0, -capex)

    npv = npv_at(discount_rate, cfs)
    lo, hi = -0.5, 1.0
    for _ in range(300):
        mid = (lo + hi) / 2
        irr_val = npv_at(mid, cfs)
        if irr_val > 0:
            lo = mid
        else:
            hi = mid
    irr = (lo + hi) / 2

    dc = capex + sum(om[i] / (1 + discount_rate) ** (i + 1) for i in range(lifetime))
    dr = sum(rev[i] / (1 + discount_rate) ** (i + 1) for i in range(lifetime))
    bcr = dr / dc
    return npv, irr * 100, bcr


print("\n" + "=" * 78)
print("2. TABLE 4 SCENARIOS — RECOMPUTED vs MANUSCRIPT")
print("=" * 78)

scenarios = [
    ("Baseline (SC=70%, tariff=4.75)", 0.70, 4.75, 0.0, (8_310_000, 9.42, 1.29)),
    ("Upper bound (SC=100%)",          1.00, 4.75, 0.0, (23_570_000, 14.43, 1.83)),
    ("Conservative (SC=50%, tariff=4.50)", 0.50, 4.50, 0.0, (-3_210_000, 4.98, 0.89)),
]

for label, sc, tf, ec, stated in scenarios:
    npv, irr, bcr = run_scenario(sc, tf, ec)
    s_npv, s_irr, s_bcr = stated
    ok_npv = abs(npv - s_npv) <= TOL_THB * 4  # wider tolerance, these are rounded to 10k in ms
    ok_irr = abs(irr - s_irr) <= TOL_PCT
    ok_bcr = abs(bcr - s_bcr) <= TOL_PCT
    print(f"\n{label}")
    print(f"  NPV: recomputed={npv:>14,.0f}  manuscript={s_npv:>14,.0f}  {'OK' if ok_npv else '!! MISMATCH'}")
    print(f"  IRR: recomputed={irr:>14.2f}%  manuscript={s_irr:>14.2f}%  {'OK' if ok_irr else '!! MISMATCH'}")
    print(f"  BCR: recomputed={bcr:>14.2f}   manuscript={s_bcr:>14.2f}   {'OK' if ok_bcr else '!! MISMATCH'}")


# ==============================================================================
# 3. RECOMPUTE RPAF OVERALL SCORES (Table 5)
# ==============================================================================

print("\n" + "=" * 78)
print("3. RPAF OVERALL SCORES — RECOMPUTED vs MANUSCRIPT")
print("=" * 78)

weights = dict(policy_stability=0.30, financial_incentives=0.25,
               regulatory_environment=0.20, market_maturity=0.15,
               implementation_barriers=0.10)

rpaf_raw = {
    "Thailand":    dict(policy_stability=85, financial_incentives=78, regulatory_environment=72, market_maturity=80, implementation_barriers=75, stated=78.9),
    "Singapore":   dict(policy_stability=90, financial_incentives=65, regulatory_environment=85, market_maturity=85, implementation_barriers=90, stated=82.0),
    "Malaysia":    dict(policy_stability=75, financial_incentives=70, regulatory_environment=68, market_maturity=75, implementation_barriers=80, stated=72.8),
    "Vietnam":     dict(policy_stability=70, financial_incentives=85, regulatory_environment=60, market_maturity=70, implementation_barriers=65, stated=71.2),
    "Philippines": dict(policy_stability=65, financial_incentives=75, regulatory_environment=65, market_maturity=65, implementation_barriers=60, stated=67.0),
    "Indonesia":   dict(policy_stability=60, financial_incentives=60, regulatory_environment=55, market_maturity=60, implementation_barriers=55, stated=58.5),
}

print(f"{'Country':<14}{'Recomputed':>12}{'Manuscript':>12}{'Match?':>10}")
recomputed_scores = {}
for country, d in rpaf_raw.items():
    score = sum(d[k] * w for k, w in weights.items())
    recomputed_scores[country] = score
    ok = abs(score - d["stated"]) <= 0.05
    print(f"{country:<14}{score:>12.2f}{d['stated']:>12.2f}{'OK' if ok else '!! MISMATCH':>12}")

# Regional average check
avg_stated = dict(policy_stability=74.2, financial_incentives=72.2, regulatory_environment=67.5,
                   market_maturity=72.5, implementation_barriers=70.8, overall_stated=71.2)
recomputed_avgs = {k: np.mean([rpaf_raw[c][k] for c in rpaf_raw]) for k in weights}
overall_avg = np.mean(list(recomputed_scores.values()))
print(f"\nRegional average dimension scores (recomputed vs manuscript):")
for k in weights:
    ok = abs(recomputed_avgs[k] - avg_stated[k]) <= 0.05
    print(f"  {k:<26}{recomputed_avgs[k]:>8.2f} vs {avg_stated[k]:>8.2f}  {'OK' if ok else '!! MISMATCH'}")
ok = abs(overall_avg - avg_stated["overall_stated"]) <= 0.05
print(f"  {'overall_score':<26}{overall_avg:>8.2f} vs {avg_stated['overall_stated']:>8.2f}  {'OK' if ok else '!! MISMATCH'}")


# ==============================================================================
# 4. VALIDATION REGRESSION — STRUCTURAL RECHECK ONLY
# ==============================================================================

print("\n" + "=" * 78)
print("4. VALIDATION REGRESSION (Table 6) — STRUCTURAL RECHECK")
print("=" * 78)
print("NOTE: The manuscript reports Pearson r=0.757, Spearman rho=0.739, R2=0.574,")
print("      n=45, based on the AUTHORS' real 45-project dataset, which this script")
print("      does not have access to (only the country-level summary in Table A2).")
print("      This script therefore CANNOT independently verify those specific")
print("      numbers -- doing so would require fabricating 45 data points, which")
print("      is exactly what was avoided in the previous script. What CAN be")
print("      checked without the raw data is internal consistency:\n")

# Check: does the regression equation reported (IRR = 0.145*RPAF - 1.855) predict
# IRR values inside the stated country-level IRR ranges (Table A2), for the RPAF
# score of each country? This is a real, non-fabricated consistency check.
beta0, beta1 = -1.855, 0.145
table_a2 = {
    "Thailand":    (78.9, 8.7, 10.1),
    "Singapore":   (82.0, 9.2, 10.8),
    "Malaysia":    (72.8, 7.5, 9.4),
    "Vietnam":     (71.2, 6.8, 11.2),
    "Philippines": (67.0, 6.8, 8.8),
    "Indonesia":   (58.5, 5.5, 7.8),
}
print(f"{'Country':<14}{'RPAF':>8}{'Predicted IRR':>16}{'Stated range':>18}{'Inside range?':>16}")
for country, (rpaf, lo_irr, hi_irr) in table_a2.items():
    pred = beta0 + beta1 * rpaf
    inside = lo_irr <= pred <= hi_irr
    print(f"{country:<14}{rpaf:>8.1f}{pred:>16.2f}{f'{lo_irr}-{hi_irr}':>18}{'YES' if inside else 'NO':>16}")

print("\n>>> ACTION REQUIRED: to fully recheck Table 6 and Figure 5, supply the real")
print(">>> 45-project CSV (columns: country, rpaf_score, irr_pct) and re-run")
print(">>> Section 5 of rpaf_analysis.py with SYNTHETIC_DATA = False.")

print("\n" + "=" * 78)
print("RECHECK COMPLETE")
print("=" * 78)
