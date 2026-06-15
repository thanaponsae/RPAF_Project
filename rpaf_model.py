"""
rpaf_model.py
=============
Computes RPAF scores for all countries.
Output: results/rpaf_results.json

Run:  python src/rpaf_model.py
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(__file__))
from parameters import RPAF_WEIGHTS, COUNTRY_SCORES, COUNTRY_ORDER, DIMS


def compute_overall(country: str) -> float:
    scores = COUNTRY_SCORES[country]
    return sum(scores[d] * RPAF_WEIGHTS[d] for d in DIMS)


def risk_rating(score: float) -> str:
    if score >= 80:   return "Low"
    if score >= 75:   return "Low-Medium"
    if score >= 65:   return "Medium"
    if score >= 60:   return "Medium-High"
    return "High"


def main():
    results = {}
    print("=" * 65)
    print("RPAF SCORES — Six ASEAN Countries (2023)")
    print("=" * 65)
    hdr = f"{'Country':<14}" + "".join(f"{d[:8]:>10}" for d in DIMS) + f"{'Overall':>10}{'Risk':>14}"
    print(hdr)
    print("-" * 65)

    for country in COUNTRY_ORDER:
        scores  = COUNTRY_SCORES[country]
        overall = round(compute_overall(country), 1)
        risk    = risk_rating(overall)
        row     = {d: scores[d] for d in DIMS}
        row["overall_score"] = overall
        row["risk_rating"]   = risk
        results[country]     = row
        line = f"{country:<14}" + "".join(f"{scores[d]:>10}" for d in DIMS)
        print(f"{line}{overall:>10.1f}{risk:>14}")

    reg_avg = {d: round(sum(COUNTRY_SCORES[c][d] for c in COUNTRY_ORDER)/6, 1)
               for d in DIMS}
    reg_overall = round(sum(compute_overall(c) for c in COUNTRY_ORDER)/6, 1)
    results["Regional Average"] = {**reg_avg, "overall_score": reg_overall,
                                    "risk_rating": risk_rating(reg_overall)}
    line = f"{'Regional Avg':<14}" + "".join(f"{reg_avg[d]:>10.1f}" for d in DIMS)
    print(f"{line}{reg_overall:>10.1f}{'Medium':>14}")

    os.makedirs("results", exist_ok=True)
    with open("results/rpaf_results.json", "w") as f:
        json.dump({"weights": RPAF_WEIGHTS, "scores": results,
                   "dims": DIMS, "country_order": COUNTRY_ORDER}, f, indent=2)
    print("\n[Saved] results/rpaf_results.json")
    return results


if __name__ == "__main__":
    main()
