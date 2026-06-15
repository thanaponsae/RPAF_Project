"""
financial_model.py
==================
Core financial functions: NPV, IRR, BCR, LCOE, SPB, DPB
Sensitivity analysis, Tornado analysis
Output: results/financial_results.json

Run:  python src/financial_model.py
"""
import os, sys, json
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from parameters import (
    CAPACITY_KWP, IRRADIATION, PERFORMANCE_RATIO, AVAILABILITY,
    DEGRADATION, LIFETIME, FIT_RATE, DISCOUNT_RATE, CARBON_PRICE,
    EMISSION_FACTOR, OM_RATE, STAFF_COST, TAX_RATE, CAPEX
)

# ── CORE FUNCTIONS ─────────────────────────────────────────────────────────

def annual_energy(year: int, pr_mult: float = 1.0) -> float:
    """Annual energy output kWh for year y (1-indexed)."""
    degrade = (1 - DEGRADATION) ** (year - 1)
    return (CAPACITY_KWP * IRRADIATION * PERFORMANCE_RATIO
            * pr_mult * AVAILABILITY * 365 * degrade)


def annual_cf(year: int, capex_mult=1.0, rev_mult=1.0,
              pr_mult=1.0, om_mult=1.0, carbon_mult=1.0) -> dict:
    """Return dict of cash flow components for one year."""
    en         = annual_energy(year, pr_mult)
    rev_fit    = en * FIT_RATE * rev_mult
    rev_carbon = en / 1000 * EMISSION_FACTOR * CARBON_PRICE * carbon_mult * rev_mult
    revenue    = rev_fit + rev_carbon
    om_cost    = (en * OM_RATE + STAFF_COST) * om_mult
    ebit       = revenue - om_cost
    tax        = max(ebit, 0) * TAX_RATE
    net_cf     = ebit - tax
    return {"year": year, "energy_kwh": round(en, 2),
            "rev_fit": round(rev_fit, 2), "rev_carbon": round(rev_carbon, 2),
            "revenue": round(revenue, 2), "om_cost": round(om_cost, 2),
            "ebit": round(ebit, 2), "net_cf": round(net_cf, 2)}


def build_cashflows(capex_mult=1.0, rev_mult=1.0, pr_mult=1.0,
                    om_mult=1.0, carbon_mult=1.0):
    """Return (np.array of cashflows, list of component dicts)."""
    rows = [annual_cf(y, capex_mult, rev_mult, pr_mult, om_mult, carbon_mult)
            for y in range(1, LIFETIME + 1)]
    cfs  = np.array([-CAPEX * capex_mult] + [r["net_cf"] for r in rows])
    return cfs, rows


def npv(cfs: np.ndarray, dr: float = DISCOUNT_RATE) -> float:
    return float(sum(cf / (1 + dr) ** t for t, cf in enumerate(cfs)))


def irr(cfs: np.ndarray, tol: float = 1e-10) -> float:
    lo, hi = -0.9999, 10.0
    for _ in range(15_000):
        mid = (lo + hi) / 2
        val = sum(cf / (1 + mid) ** t for t, cf in enumerate(cfs))
        if abs(val) < tol:
            break
        lo, hi = (mid, hi) if val > 0 else (lo, mid)
    return float(mid)


def payback(cfs: np.ndarray, discounted: bool = False,
            dr: float = DISCOUNT_RATE) -> float:
    cap = abs(cfs[0])
    cum = 0.0
    for y in range(1, len(cfs)):
        cf   = cfs[y] / (1 + dr) ** y if discounted else cfs[y]
        cum += cf
        if cum >= cap:
            return float(y - 1 + (cap - (cum - cf)) / cf)
    return float("inf")


def lcoe(cfs: np.ndarray, rows: list,
         dr: float = DISCOUNT_RATE) -> float:
    cap = abs(cfs[0])
    num = cap + sum(r["om_cost"] / (1 + dr) ** r["year"] for r in rows)
    den = sum(r["energy_kwh"] / (1 + dr) ** r["year"] for r in rows)
    return float(num / den)


def bcr(cfs: np.ndarray, dr: float = DISCOUNT_RATE) -> float:
    cap = abs(cfs[0])
    return float(sum(max(cf, 0) / (1 + dr) ** t
                     for t, cf in enumerate(cfs)) / cap)


# ── MAIN ───────────────────────────────────────────────────────────────────

def main():
    # Baseline
    cfs_b, rows_b = build_cashflows()
    res = {
        "npv_M":   round(npv(cfs_b) / 1e6, 4),
        "irr_pct": round(irr(cfs_b) * 100, 4),
        "bcr":     round(bcr(cfs_b), 4),
        "spb_yr":  round(payback(cfs_b), 2),
        "dpb_yr":  round(payback(cfs_b, discounted=True), 2),
        "lcoe_THBkWh": round(lcoe(cfs_b, rows_b), 4),
    }
    print("=" * 50)
    print("BASELINE RESULTS")
    print("=" * 50)
    for k, v in res.items():
        print(f"  {k:<18}: {v}")

    # Cumulative discounted NPV (for Figure 1)
    cum_disc = [0.0]
    run = 0.0
    for y in range(1, LIFETIME + 1):
        run += cfs_b[y] / (1 + DISCOUNT_RATE) ** y
        cum_disc.append(round(run, 2))

    # Sensitivity
    SCENARIOS = {
        "Baseline":         {},
        "Cost +20%":        {"capex_mult": 1.20},
        "Revenue -20%":     {"rev_mult":   0.80},
        "Combined adverse": {"capex_mult": 1.20, "rev_mult": 0.80},
    }
    print("\nSENSITIVITY:")
    sensitivity = []
    for name, kw in SCENARIOS.items():
        c, r = build_cashflows(**kw)
        n_   = npv(c); i_ = irr(c); b_ = bcr(c)
        s_   = payback(c); l_ = lcoe(c, r)
        via  = "High" if n_>5e6 else ("Moderate" if n_>0 else "Unviable")
        print(f"  {name:<22} NPV={n_/1e6:>6.2f}M  IRR={i_*100:>5.2f}%  via={via}")
        sensitivity.append({
            "scenario": name,
            "npv_M_THB": round(n_/1e6, 3),
            "irr_pct":   round(i_*100, 3),
            "bcr":       round(b_, 3),
            "spb_yr":    round(s_, 2) if s_ != float("inf") else None,
            "lcoe":      round(l_, 3),
            "viability": via,
        })

    # Tornado ±20%
    TVARS = [
        ("Feed-in Tariff rate",  {"rev_mult":0.80},   {"rev_mult":1.20}),
        ("Initial capital cost", {"capex_mult":1.20},  {"capex_mult":0.80}),
        ("Performance ratio",    {"pr_mult":0.80},    {"pr_mult":1.20}),
        ("O&M costs",            {"om_mult":1.20},    {"om_mult":0.80}),
        ("Carbon credit price",  {"carbon_mult":0.80},{"carbon_mult":1.20}),
    ]
    tornado = []
    for label, klo, khi in TVARS:
        clo, _ = build_cashflows(**klo)
        chi, _ = build_cashflows(**khi)
        tornado.append({"variable": label,
                         "npv_low":  round(npv(clo)/1e6, 3),
                         "npv_high": round(npv(chi)/1e6, 3)})
    # Discount rate
    tornado.append({"variable": "Discount rate",
                     "npv_low":  round(npv(cfs_b, dr=DISCOUNT_RATE*1.2)/1e6, 3),
                     "npv_high": round(npv(cfs_b, dr=DISCOUNT_RATE*0.8)/1e6, 3)})
    tornado.sort(key=lambda x: abs(x["npv_high"] - x["npv_low"]))

    # Save
    os.makedirs("results", exist_ok=True)
    out = {
        "baseline":      res,
        "cum_disc_npv":  cum_disc,
        "cashflows":     cfs_b.tolist(),
        "components":    rows_b,
        "sensitivity":   sensitivity,
        "tornado":       tornado,
    }
    with open("results/financial_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\n[Saved] results/financial_results.json")
    return out


if __name__ == "__main__":
    main()
