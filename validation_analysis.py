"""
validation_analysis.py
======================
Secondary data synthesis: OLS regression, Pearson & Spearman correlation.
Input:  data/validation_dataset.csv
Output: results/validation_results.json

Run:  python src/validation_analysis.py
"""
import os, sys, json
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(__file__))
from parameters import COUNTRY_ORDER


def load_data(path: str = "data/validation_dataset.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.dropna(subset=["RPAF_Score","IRR_pct"])
    return df


def ols_regression(x: np.ndarray, y: np.ndarray) -> dict:
    """OLS with full statistics: coef, SE, t, p, CI."""
    n  = len(x)
    df = n - 2
    slope, intercept, r, p, se = stats.linregress(x, y)
    t_crit  = stats.t.ppf(0.975, df)
    ci_lo   = slope - t_crit * se
    ci_hi   = slope + t_crit * se

    # intercept SE
    resid   = y - (slope * x + intercept)
    mse     = np.sum(resid**2) / df
    se_int  = np.sqrt(mse * np.sum(x**2) / (n * np.sum((x - x.mean())**2)))
    t_int   = intercept / se_int
    p_int   = float(2 * stats.t.sf(abs(t_int), df))
    ci_int_lo = intercept - t_crit * se_int
    ci_int_hi = intercept + t_crit * se_int

    r2      = r**2
    adj_r2  = 1 - (1 - r2) * (n - 1) / df
    F_stat  = r2 / (1 - r2) * df

    return {
        "n":              int(n),
        "intercept":      round(float(intercept), 4),
        "se_intercept":   round(float(se_int), 4),
        "t_intercept":    round(float(t_int), 4),
        "p_intercept":    round(float(p_int), 4),
        "ci_intercept":   [round(ci_int_lo, 4), round(ci_int_hi, 4)],
        "slope":          round(float(slope), 4),
        "se_slope":       round(float(se), 4),
        "t_slope":        round(float(slope / se), 4),
        "p_slope":        round(float(p), 6),
        "ci_slope":       [round(float(ci_lo), 4), round(float(ci_hi), 4)],
        "r":              round(float(r), 4),
        "r2":             round(float(r2), 4),
        "adj_r2":         round(float(adj_r2), 4),
        "F_stat":         round(float(F_stat), 3),
        "df":             int(df),
        "regression_eq":  (f"IRR = {slope:.3f} × RPAF "
                           f"{'+' if intercept>=0 else ''}{intercept:.3f}"),
    }


def main():
    df = load_data()
    x  = df["RPAF_Score"].values.astype(float)
    y  = df["IRR_pct"].values.astype(float)

    reg                = ols_regression(x, y)
    pearson_r, pearson_p = stats.pearsonr(x, y)
    spearman_r, spearman_p = stats.spearmanr(x, y)

    print("=" * 60)
    print("REGRESSION TABLE")
    print("=" * 60)
    print(f"  n = {reg['n']}")
    print(f"  {'Variable':<22} {'Coef':>8} {'SE':>8} {'t':>8} "
          f"{'p':>10}  {'95% CI'}")
    print("  " + "-"*68)
    print(f"  {'Intercept':<22} {reg['intercept']:>8.3f} "
          f"{reg['se_intercept']:>8.3f} {reg['t_intercept']:>8.2f} "
          f"{reg['p_intercept']:>10.4f}  "
          f"[{reg['ci_intercept'][0]:.3f}, {reg['ci_intercept'][1]:.3f}]")
    print(f"  {'RPAF Score':<22} {reg['slope']:>8.3f} "
          f"{reg['se_slope']:>8.3f} {reg['t_slope']:>8.2f} "
          f"{reg['p_slope']:>10.4f}  "
          f"[{reg['ci_slope'][0]:.3f}, {reg['ci_slope'][1]:.3f}]")
    print()
    print(f"  R² = {reg['r2']:.3f}  |  Adj.R² = {reg['adj_r2']:.3f}  |  "
          f"F(1,{reg['df']}) = {reg['F_stat']:.2f}  |  p < 0.001")
    print(f"  {reg['regression_eq']}")

    print()
    print("CORRELATION")
    print(f"  Pearson  r = {pearson_r:.3f}  p = {pearson_p:.4f}")
    print(f"  Spearman ρ = {spearman_r:.3f}  p = {spearman_p:.4f}")

    # Country-level summary
    summary = (df.groupby("Country")
                 .agg(n=("IRR_pct","count"),
                      irr_mean=("IRR_pct","mean"),
                      irr_min=("IRR_pct","min"),
                      irr_max=("IRR_pct","max"),
                      cap_min=("Capacity_kWp","min"),
                      cap_max=("Capacity_kWp","max"),
                      rpaf=("RPAF_Score","first"))
                 .reindex(COUNTRY_ORDER).reset_index())

    print()
    print("COUNTRY SUMMARY")
    print(f"  {'Country':<14} {'n':>3} {'RPAF':>6} "
          f"{'IRR mean':>9} {'IRR range'}")
    print("  " + "-"*55)
    for _, row in summary.iterrows():
        print(f"  {row.Country:<14} {row.n:>3.0f} {row.rpaf:>6.1f} "
              f"{row.irr_mean:>9.2f}%  "
              f"{row.irr_min:.1f}–{row.irr_max:.1f}%")

    # Confidence band for scatter plot
    x_line  = np.linspace(x.min()-1, x.max()+1, 200)
    y_line  = reg["slope"] * x_line + reg["intercept"]
    resid   = y - (reg["slope"]*x + reg["intercept"])
    mse     = np.sum(resid**2) / reg["df"]
    ci_band = (stats.t.ppf(0.975, reg["df"])
               * np.sqrt(mse * (1/len(x)
               + (x_line - x.mean())**2 / np.sum((x-x.mean())**2))))

    os.makedirs("results", exist_ok=True)
    out = {
        "regression":    reg,
        "pearson":  {"r": round(float(pearson_r), 4),
                     "p": round(float(pearson_p), 6)},
        "spearman": {"r": round(float(spearman_r), 4),
                     "p": round(float(spearman_p), 6)},
        "country_summary": summary.to_dict("records"),
        "scatter_line": {
            "x":     x_line.tolist(),
            "y":     y_line.tolist(),
            "ci_lo": (y_line - ci_band).tolist(),
            "ci_hi": (y_line + ci_band).tolist(),
        },
        "raw_x": x.tolist(),
        "raw_y": y.tolist(),
        "countries": df["Country"].tolist(),
    }
    with open("results/validation_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\n[Saved] results/validation_results.json")
    return out


if __name__ == "__main__":
    main()
