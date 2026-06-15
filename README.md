# RPAF Analysis — Reproducible Code & Data

[![Tests](https://github.com/USERNAME/REPO/actions/workflows/tests.yml/badge.svg)](https://github.com/USERNAME/REPO/actions/workflows/tests.yml)

**Paper:** Saengsuwan T, Kaewpia T (2025). Economic Feasibility Analysis of PV Rooftop Smart Microgrid Implementation in University Campus: Development of a Regional Policy Assessment Framework. *Utilities Policy*. JUIP-D-25-02437.

---

## Reproducing All Results in One Command

```bash
git clone https://github.com/USERNAME/REPO.git
cd REPO
pip install -r requirements.txt
cd src && python run_all.py
```

All results appear in `results/` (JSON) and `figures/` (PNG).

---

## Project Structure

```
├── src/
│   ├── parameters.py          # Central parameter store (Table 1)
│   ├── financial_model.py     # NPV, IRR, BCR, LCOE, SPB, DPB, Sensitivity, Tornado
│   ├── rpaf_model.py          # RPAF scores & country rankings
│   ├── validation_analysis.py # OLS regression, Pearson r, Spearman ρ
│   ├── figures.py             # All manuscript figures (Fig 1–5)
│   └── run_all.py             # Master script — runs everything
├── data/
│   └── validation_dataset.csv # 45 institutional PV projects (Table A1)
├── results/                   # Generated JSON outputs
├── figures/                   # Generated PNG figures
├── tests/
│   ├── test_financial.py      # 11 unit tests — financial model
│   ├── test_rpaf.py           # 10 unit tests — RPAF model
│   └── test_validation.py     # 13 unit tests — regression analysis
├── requirements.txt
└── README.md
```

---

## Scripts Overview

| Script | Purpose | Output |
|--------|---------|--------|
| `parameters.py` | All parameters (Table 1) | — |
| `financial_model.py` | NPV/IRR/BCR/LCOE/Sensitivity/Tornado | `results/financial_results.json` |
| `rpaf_model.py` | RPAF scores, rankings | `results/rpaf_results.json` |
| `validation_analysis.py` | OLS regression, correlations | `results/validation_results.json` |
| `figures.py` | Figures 1–5 | `figures/Fig*.png` |
| `run_all.py` | Run everything | all of the above |

---

## Key Results (Table 3)

| Metric | Value |
|--------|-------|
| NPV | 8.61 million THB |
| IRR | 9.94% |
| BCR | 1.35 |
| Simple Payback | 8.9 years |
| Discounted Payback | 13.6 years |
| LCOE | 3.23 THB/kWh |

**Note:** Effective tax rate = 0% (Rajabhat University is a government entity
exempt from corporate income tax under Thailand Revenue Code §3(1)).

---

## Validation Results (Table 6)

OLS Regression: **IRR = 0.145 × RPAF − 1.855**

| Statistic | Value |
|-----------|-------|
| Pearson r | 0.757 (p < 0.001) |
| Spearman ρ | 0.739 (p < 0.001) |
| R² | 0.574 |
| Adj. R² | 0.564 |
| F(1,43) | 57.86 |
| n | 45 projects |

---

## Validation Dataset

`data/validation_dataset.csv` contains 45 institutional PV projects (100–1,000 kWp)
across 6 ASEAN countries (2019–2024), compiled from published feasibility studies
via secondary data synthesis. Every project cites its source paper (Source_ref column).

---

## RPAF Dimension Weights

Derived from systematic review of 18 empirical studies (Section 2.5):

| Dimension | Weight | Key Source |
|-----------|--------|-----------|
| Policy Stability | 30% | North (1990); Schmidt & Sewerin (2019) |
| Financial Incentives | 25% | Polzin et al. (2019) |
| Regulatory Environment | 20% | Sadeghi et al. (2025) |
| Market Maturity | 15% | Kostakis (2024) |
| Implementation Barriers | 10% | Mamat et al. (2025) |

---

## Running Tests

```bash
python tests/test_financial.py    # 11 tests
python tests/test_rpaf.py         # 10 tests
python tests/test_validation.py   # 13 tests
```

All 34 tests pass on Python 3.10, 3.11, 3.12.

---

## Citation

```bibtex
@article{saengsuwan2025rpaf,
  author  = {Saengsuwan, Thanapon and Kaewpia, Teerawat},
  title   = {Economic Feasibility Analysis of {PV} Rooftop Smart Microgrid
             Implementation in University Campus: Development of a Regional
             Policy Assessment Framework},
  journal = {Utilities Policy},
  year    = {2025},
  note    = {JUIP-D-25-02437}
}
```

---

## License

MIT License — see `LICENSE` for details.
