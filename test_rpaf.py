"""
test_rpaf.py — Unit tests for RPAF model
Run:  pytest tests/  OR  python tests/test_rpaf.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from rpaf_model import compute_overall, risk_rating
from parameters import RPAF_WEIGHTS, COUNTRY_SCORES, COUNTRY_ORDER

def test_weights_sum_to_one():
    """Dimension weights must sum to 1.0."""
    total = sum(RPAF_WEIGHTS.values())
    assert abs(total - 1.0) < 1e-9, f"Weights sum = {total}"
    print(f"  PASS  Weights sum = {total:.4f}")

def test_thailand_overall():
    """Thailand overall RPAF ≈ 78.9."""
    ov = compute_overall("Thailand")
    assert abs(ov - 78.9) < 0.15, f"Thailand = {ov:.1f}"
    print(f"  PASS  Thailand RPAF = {ov:.1f} (target: 78.9)")

def test_singapore_overall():
    """Singapore overall RPAF ≈ 82.0."""
    ov = compute_overall("Singapore")
    assert abs(ov - 82.0) < 0.15, f"Singapore = {ov:.1f}"
    print(f"  PASS  Singapore RPAF = {ov:.1f} (target: 82.0)")

def test_indonesia_overall():
    """Indonesia overall RPAF ≈ 58.5."""
    ov = compute_overall("Indonesia")
    assert abs(ov - 58.5) < 0.15, f"Indonesia = {ov:.1f}"
    print(f"  PASS  Indonesia RPAF = {ov:.1f} (target: 58.5)")

def test_ranking_order():
    """Singapore > Thailand > Malaysia > Vietnam > Philippines > Indonesia."""
    ovs = {c: compute_overall(c) for c in COUNTRY_ORDER}
    ranked = sorted(ovs, key=ovs.get, reverse=True)
    expected = ["Singapore","Thailand","Malaysia","Vietnam","Philippines","Indonesia"]
    assert ranked == expected, f"Ranking: {ranked}"
    print(f"  PASS  Ranking: {' > '.join(ranked)}")

def test_risk_ratings():
    """Spot-check risk ratings."""
    assert risk_rating(82.0) == "Low",        "82 should be Low"
    assert risk_rating(78.9) == "Low-Medium", "78.9 should be Low-Medium"
    assert risk_rating(72.8) == "Medium",     "72.8 should be Medium"
    assert risk_rating(58.5) == "High",       "58.5 should be High"
    print("  PASS  Risk ratings correct")

def test_all_countries_present():
    """All six ASEAN countries must be in COUNTRY_SCORES."""
    expected = {"Thailand","Singapore","Malaysia",
                "Vietnam","Philippines","Indonesia"}
    assert set(COUNTRY_SCORES.keys()) == expected
    print(f"  PASS  All 6 countries present")

def test_scores_in_range():
    """All dimension scores must be in [0, 100]."""
    for country, dims in COUNTRY_SCORES.items():
        for dim, score in dims.items():
            assert 0 <= score <= 100, f"{country}/{dim} = {score}"
    print("  PASS  All scores in [0, 100]")

def test_singapore_regulation_exceeds_thailand():
    """Singapore regulatory environment > Thailand (key finding)."""
    sg = COUNTRY_SCORES["Singapore"]["Regulatory Environment"]
    th = COUNTRY_SCORES["Thailand"]["Regulatory Environment"]
    assert sg > th, f"SG={sg}, TH={th}"
    print(f"  PASS  Singapore regulatory ({sg}) > Thailand ({th})")

def test_vietnam_financial_incentives_highest():
    """Vietnam has highest financial incentives score (FiT 1/2 legacy)."""
    fi = {c: COUNTRY_SCORES[c]["Financial Incentives"] for c in COUNTRY_ORDER}
    top = max(fi, key=fi.get)
    assert top == "Vietnam", f"Top FI = {top}"
    print(f"  PASS  Vietnam has highest FI score ({fi['Vietnam']})")


if __name__ == "__main__":
    tests = [
        test_weights_sum_to_one, test_thailand_overall,
        test_singapore_overall, test_indonesia_overall,
        test_ranking_order, test_risk_ratings,
        test_all_countries_present, test_scores_in_range,
        test_singapore_regulation_exceeds_thailand,
        test_vietnam_financial_incentives_highest,
    ]
    print("=" * 55)
    print("RPAF MODEL — UNIT TESTS")
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
