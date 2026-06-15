"""
run_all.py
==========
Master script — runs every analysis step in sequence.
Produces all results JSON files and all figures.

Usage:
    python src/run_all.py
    python src/run_all.py --skip-figures   (CI / headless environments)
"""
import sys, os, time, argparse

sys.path.insert(0, os.path.dirname(__file__))

def step(name, fn):
    print(f"\n{'='*55}")
    print(f"  STEP: {name}")
    print(f"{'='*55}")
    t = time.time()
    fn()
    print(f"  Done in {time.time()-t:.1f}s")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-figures", action="store_true",
                        help="Skip figure generation (headless mode)")
    args = parser.parse_args()

    import financial_model    as fm
    import rpaf_model         as rm
    import validation_analysis as va

    step("Financial Model (NPV, IRR, BCR, LCOE, Sensitivity, Tornado)",
         fm.main)
    step("RPAF Model (Country Scores & Rankings)",
         rm.main)
    step("Validation Analysis (OLS Regression, Pearson, Spearman)",
         va.main)

    if not args.skip_figures:
        import figures as fg
        step("Figure Generation (Fig 1–5)", fg.main)
    else:
        print("\n[Skipped] Figure generation (--skip-figures)")

    print(f"\n{'='*55}")
    print("  ALL STEPS COMPLETE")
    print(f"  results/  — JSON outputs")
    print(f"  figures/  — PNG figures")
    print(f"{'='*55}\n")

if __name__ == "__main__":
    main()
