"""
test_validation.py — Unit tests for validation/regression analysis
Run:  pytest tests/  OR  python tests/test_validation.py
"""
import sys, os
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from validation_analysis import load_data, ols_regression
from scipy import stats

def test_dataset_size():
    """Dataset must have exactly 45 projects."""
    df = load_data()
    n  = len(df)
    assert n == 45, f"Expected 45 rows, got {n}"
    print(f"  PASS  Dataset size = {n}")

def test_all_six_countries():
    """Dataset must cover all six ASEAN countries."""
    df       = load_data()
    expected = {"Thailand","Singapore","Malaysia",
                "Vietnam","Philippines","Indonesia"}
    found    = set(df["Country"].unique())
    assert found == expected, f"Countries found: {found}"
    print(f"  PASS  All 6 countries present")

def test_country_counts():
    """Country n: TH=8, SG=6, MY=8, VN=7, PH=9, ID=7."""
    df       = load_data()
    expected = {"Thailand":8,"Singapore":6,"Malaysia":8,
                "Vietnam":7,"Philippines":9,"Indonesia":7}
    counts   = df.groupby("Country").size().to_dict()
    for country, n in expected.items():
        assert counts.get(country) == n, \
            f"{country}: expected {n}, got {counts.get(country)}"
    print(f"  PASS  Country counts correct: {counts}")

def test_irr_range_sensible():
    """All IRR values between 3% and 15%."""
    df = load_data()
    assert df["IRR_pct"].min() >= 3.0,  f"Min IRR = {df['IRR_pct'].min()}"
    assert df["IRR_pct"].max() <= 15.0, f"Max IRR = {df['IRR_pct'].max()}"
    print(f"  PASS  IRR range: {df['IRR_pct'].min():.1f}–"
          f"{df['IRR_pct'].max():.1f}%")

def test_rpaf_range_sensible():
    """All RPAF scores between 50 and 90."""
    df = load_data()
    assert df["RPAF_Score"].min() >= 50, f"Min RPAF = {df['RPAF_Score'].min()}"
    assert df["RPAF_Score"].max() <= 90, f"Max RPAF = {df['RPAF_Score'].max()}"
    print(f"  PASS  RPAF range: {df['RPAF_Score'].min()}–"
          f"{df['RPAF_Score'].max()}")

def test_capacity_range():
    """All capacities between 100 and 1000 kWp."""
    df = load_data()
    assert df["Capacity_kWp"].min() >= 100, \
        f"Min cap = {df['Capacity_kWp'].min()}"
    assert df["Capacity_kWp"].max() <= 1000, \
        f"Max cap = {df['Capacity_kWp'].max()}"
    print(f"  PASS  Capacity range: {df['Capacity_kWp'].min()}–"
          f"{df['Capacity_kWp'].max()} kWp")

def test_regression_slope_positive():
    """OLS slope must be positive (higher RPAF → higher IRR)."""
    df  = load_data()
    x   = df["RPAF_Score"].values.astype(float)
    y   = df["IRR_pct"].values.astype(float)
    reg = ols_regression(x, y)
    assert reg["slope"] > 0, f"Slope = {reg['slope']}"
    print(f"  PASS  Slope = {reg['slope']:.3f} > 0")

def test_regression_slope_significant():
    """OLS slope p-value < 0.001."""
    df  = load_data()
    x   = df["RPAF_Score"].values.astype(float)
    y   = df["IRR_pct"].values.astype(float)
    reg = ols_regression(x, y)
    assert reg["p_slope"] < 0.001, f"p = {reg['p_slope']}"
    print(f"  PASS  p-value = {reg['p_slope']:.6f} < 0.001")

def test_r2_range():
    """R² must be between 0.40 and 0.80 (plausible range)."""
    df  = load_data()
    x   = df["RPAF_Score"].values.astype(float)
    y   = df["IRR_pct"].values.astype(float)
    reg = ols_regression(x, y)
    assert 0.40 < reg["r2"] < 0.80, f"R² = {reg['r2']}"
    print(f"  PASS  R² = {reg['r2']:.3f} (in [0.40, 0.80])")

def test_pearson_spearman_agree():
    """Pearson r and Spearman ρ must be within 0.10 of each other."""
    df = load_data()
    x  = df["RPAF_Score"].values.astype(float)
    y  = df["IRR_pct"].values.astype(float)
    pr, _ = stats.pearsonr(x, y)
    sr, _ = stats.spearmanr(x, y)
    assert abs(pr - sr) < 0.10, \
        f"Pearson={pr:.3f}, Spearman={sr:.3f}, diff={abs(pr-sr):.3f}"
    print(f"  PASS  Pearson r={pr:.3f}, Spearman ρ={sr:.3f} "
          f"(diff={abs(pr-sr):.3f} < 0.10)")

def test_singapore_irr_exceeds_indonesia():
    """Singapore mean IRR > Indonesia mean IRR (highest vs lowest RPAF)."""
    df  = load_data()
    sg  = df[df.Country=="Singapore"]["IRR_pct"].mean()
    idn = df[df.Country=="Indonesia"]["IRR_pct"].mean()
    assert sg > idn, f"SG IRR={sg:.2f}%, ID IRR={idn:.2f}%"
    print(f"  PASS  Singapore mean IRR ({sg:.2f}%) > Indonesia ({idn:.2f}%)")

def test_regression_n_equals_45():
    """Regression n must be 45."""
    df  = load_data()
    x   = df["RPAF_Score"].values.astype(float)
    y   = df["IRR_pct"].values.astype(float)
    reg = ols_regression(x, y)
    assert reg["n"] == 45, f"n = {reg['n']}"
    print(f"  PASS  n = {reg['n']}")

def test_ci_slope_excludes_zero():
    """95% CI for slope must not include zero (statistically significant)."""
    df  = load_data()
    x   = df["RPAF_Score"].values.astype(float)
    y   = df["IRR_pct"].values.astype(float)
    reg = ols_regression(x, y)
    assert reg["ci_slope"][0] > 0, \
        f"CI lower bound = {reg['ci_slope'][0]} (should be > 0)"
    print(f"  PASS  95% CI slope = [{reg['ci_slope'][0]:.3f}, "
          f"{reg['ci_slope'][1]:.3f}] (excludes zero)")


if __name__ == "__main__":
    tests = [
        test_dataset_size, test_all_six_countries, test_country_counts,
        test_irr_range_sensible, test_rpaf_range_sensible,
        test_capacity_range, test_regression_slope_positive,
        test_regression_slope_significant, test_r2_range,
        test_pearson_spearman_agree, test_singapore_irr_exceeds_indonesia,
        test_regression_n_equals_45, test_ci_slope_excludes_zero,
    ]
    print("=" * 55)
    print("VALIDATION ANALYSIS — UNIT TESTS")
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
