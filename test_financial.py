"""
test_financial.py — Unit tests for financial model
Run:  pytest tests/  OR  python tests/test_financial.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from financial_model import (
    build_cashflows, npv, irr, payback, lcoe, bcr, annual_energy
)
from parameters import (
    CAPEX, DISCOUNT_RATE, CAPACITY_KWP, IRRADIATION,
    PERFORMANCE_RATIO, AVAILABILITY, LIFETIME
)

# ── helpers ────────────────────────────────────────────────────────────────
TOL = 0.05   # 5% tolerance for published values

def approx(val, target, tol=TOL):
    return abs(val - target) / abs(target) < tol

# ── tests ──────────────────────────────────────────────────────────────────
def test_year1_energy():
    """Year-1 energy ≈ 738,776 kWh (Table note in manuscript)."""
    en = annual_energy(1)
    assert approx(en, 738_776, tol=0.01), f"Energy={en:.0f}"
    print(f"  PASS  Year-1 energy = {en:,.0f} kWh")

def test_baseline_npv():
    """NPV ≈ 8.61 M THB (Table 3)."""
    cfs, _ = build_cashflows()
    val = npv(cfs)
    assert approx(val/1e6, 8.61, tol=0.05), f"NPV={val/1e6:.2f}M"
    print(f"  PASS  NPV = {val/1e6:.2f} M THB  (target: 8.61)")

def test_baseline_irr():
    """IRR ≈ 9.94% (Table 3)."""
    cfs, _ = build_cashflows()
    val = irr(cfs) * 100
    assert approx(val, 9.94, tol=0.05), f"IRR={val:.2f}%"
    print(f"  PASS  IRR = {val:.2f}%  (target: 9.94%)")

def test_baseline_bcr():
    """BCR ≈ 1.35 (Table 3)."""
    cfs, _ = build_cashflows()
    val = bcr(cfs)
    assert approx(val, 1.35, tol=0.05), f"BCR={val:.2f}"
    print(f"  PASS  BCR = {val:.2f}  (target: 1.35)")

def test_simple_payback():
    """SPB ≈ 8.9 years (Table 3)."""
    cfs, _ = build_cashflows()
    val = payback(cfs)
    assert approx(val, 8.9, tol=0.05), f"SPB={val:.1f}"
    print(f"  PASS  SPB = {val:.1f} yr  (target: 8.9)")

def test_discounted_payback():
    """DPB ≈ 13.6 years (Table 3)."""
    cfs, _ = build_cashflows()
    val = payback(cfs, discounted=True)
    assert approx(val, 13.6, tol=0.05), f"DPB={val:.1f}"
    print(f"  PASS  DPB = {val:.1f} yr  (target: 13.6)")

def test_lcoe():
    """LCOE ≈ 3.23 THB/kWh (Table 3)."""
    cfs, rows = build_cashflows()
    val = lcoe(cfs, rows)
    assert approx(val, 3.23, tol=0.03), f"LCOE={val:.3f}"
    print(f"  PASS  LCOE = {val:.3f} THB/kWh  (target: 3.23)")

def test_tax_exempt():
    """Tax rate = 0 for government university (Revenue Code §3(1))."""
    from parameters import TAX_RATE
    assert TAX_RATE == 0.0, f"TAX_RATE={TAX_RATE}"
    print("  PASS  TAX_RATE = 0.0 (government university)")

def test_sensitivity_combined_adverse():
    """Combined adverse (cost+20%, rev-20%) NPV < 0 (Table 5)."""
    cfs, _ = build_cashflows(capex_mult=1.20, rev_mult=0.80)
    val = npv(cfs)
    assert val < 0, f"Combined adverse NPV={val/1e6:.2f}M should be < 0"
    print(f"  PASS  Combined adverse NPV = {val/1e6:.2f} M THB (< 0)")

def test_cost_stress_remains_viable():
    """Cost +20% alone: NPV > 0, IRR > DR (Table 5)."""
    from parameters import DISCOUNT_RATE
    cfs, _ = build_cashflows(capex_mult=1.20)
    assert npv(cfs) > 0,                      "Cost+20% NPV should be > 0"
    assert irr(cfs) > DISCOUNT_RATE,          "Cost+20% IRR should exceed DR"
    print(f"  PASS  Cost+20% NPV={npv(cfs)/1e6:.2f}M > 0, "
          f"IRR={irr(cfs)*100:.2f}% > {DISCOUNT_RATE*100:.2f}%")

def test_npv_monotone_in_fit():
    """Higher FiT → higher NPV (sanity check)."""
    cfs_lo, _ = build_cashflows(rev_mult=0.80)
    cfs_hi, _ = build_cashflows(rev_mult=1.20)
    assert npv(cfs_hi) > npv(cfs_lo), "Higher revenue should yield higher NPV"
    print(f"  PASS  NPV monotone in FiT: "
          f"{npv(cfs_lo)/1e6:.2f}M < {npv(cfs_hi)/1e6:.2f}M")


if __name__ == "__main__":
    tests = [
        test_year1_energy, test_baseline_npv, test_baseline_irr,
        test_baseline_bcr, test_simple_payback, test_discounted_payback,
        test_lcoe, test_tax_exempt, test_sensitivity_combined_adverse,
        test_cost_stress_remains_viable, test_npv_monotone_in_fit,
    ]
    print("=" * 55)
    print("FINANCIAL MODEL — UNIT TESTS")
    print("=" * 55)
    passed = failed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            print(f"  FAIL  {t.__name__}: {e}")
            failed += 1
    print(f"\n{'='*55}")
    print(f"  {passed} passed  |  {failed} failed  |  {len(tests)} total")
    print(f"{'='*55}")
    sys.exit(failed)
