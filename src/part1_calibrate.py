"""
Part 1 -- Calibrate the B-Bicycle digital twin against 30 years of field data.

Usage
-----
    python src/part1_calibrate.py                 # writes p1_artifacts/
    python src/part1_calibrate.py --out-dir tmp/  # write somewhere else

Pipeline
--------
1. Load 360 monthly field observations (Jan-1996 .. Dec-2025).
2. Split chronologically 80/20 -> 288 train months, 72 test months.
3. Run the raw DES on every month to get the uncalibrated baseline.
4. Phase 1: fit the stacking ensemble on the training months.
5. Phase 2: recursive Bayesian bias/scale correction across the test months.
6. Write 12 artifacts for Part 2 to consume, plus figures.

Reproducibility
---------------
Every DES call is seeded from its row index (TRAIN_SEED_BASE + i,
TEST_SEED_BASE + i), so this script is bit-identical across runs and the
result for any given month does not depend on evaluation order. See
docs/seeding.md. sklearn estimators are pinned with random_state=42.

WARNING: running this OVERWRITES p1_artifacts/. The committed artifacts are
the originals from the submitted coursework, which were produced before the
seeding fix; regenerating them shifts Part 2's optimum. See the README section
"Which artifacts are committed, and why".
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import joblib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from simulator import BBicycleSimulator, DEFAULT_THETA, save_theta
from calibrator import EnsembleCalibrator, evaluate

RANDOM_STATE = 42
TRAIN_SEED_BASE = 1000      # DES seeds for training months: 1000 .. 1287
TEST_SEED_BASE = 500_000    # DES seeds for test months:    500000 .. 500071
SIM_REPS = 3                # replications averaged per month
TRAIN_FRAC = 0.8

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DEFAULT_DATA = os.path.join(ROOT, 'data', 'FieldData.txt')
DEFAULT_OUT = os.path.join(ROOT, 'p1_artifacts')
DEFAULT_FIGS = os.path.join(ROOT, 'results')


def load_field_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep='\t')
    df.columns = df.columns.str.strip()
    return df


def make_figures(ote, sim_te, p1_te, p2_te, cal, metrics, fig_dir):
    os.makedirs(fig_dir, exist_ok=True)

    # Observed vs predicted scatter, three panels
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    panels = [("Baseline Simulation", sim_te, metrics['baseline']),
              ("Phase 1 Calibrated", p1_te, metrics['phase1']),
              ("Phase 2 Online", p2_te, metrics['phase2'])]
    for ax, (title, pred, m) in zip(axes, panels):
        ax.scatter(ote, pred, alpha=0.6, s=20)
        lo = min(ote.min(), pred.min())
        hi = max(ote.max(), pred.max())
        ax.plot([lo, hi], [lo, hi], 'r--', lw=1.2)
        ax.set_title(f"{title}\nRMSE=${m['rmse']:,.0f}, R2={m['r2']:.3f}",
                     fontsize=10, fontweight='bold')
        ax.set_xlabel('Observed NetRevenue')
        ax.set_ylabel('Predicted NetRevenue')
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'fig_p1_01_scatter.png'),
                dpi=200, bbox_inches='tight')
    plt.close()

    # Time series over the test horizon
    plt.figure(figsize=(12, 5))
    plt.plot(ote / 1e6, 'r-o', ms=3, lw=1.5, label='Observed')
    plt.plot(sim_te / 1e6, 'g--', lw=1.2, label='Baseline Simulation')
    plt.plot(p1_te / 1e6, 'b-', lw=1.5, label='Phase 1 Calibrated')
    plt.plot(p2_te / 1e6, 'k-', lw=1.8, label='Phase 2 Online')
    plt.axhline(0, color='gray', ls=':', lw=1)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.xlabel('Test Month Index')
    plt.ylabel('Net Revenue ($M)')
    plt.title('Observed vs Predicted Net Revenue (Test Period)',
              fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'fig_p1_02_timeseries.png'),
                dpi=200, bbox_inches='tight')
    plt.close()

    # Online parameter convergence
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6))
    ax1.plot(cal.online_bias_history)
    ax1.axhline(0, ls=':', lw=1)
    ax1.set_title('Online Bias History', fontweight='bold')
    ax1.set_ylabel('Bias ($)')
    ax1.grid(True, alpha=0.3)
    ax2.plot(cal.online_scale_history)
    ax2.axhline(1.0, ls=':', lw=1)
    ax2.set_title('Online Scale History', fontweight='bold')
    ax2.set_xlabel('Test Month Index')
    ax2.set_ylabel('Scale')
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'fig_p1_03_online_params.png'),
                dpi=200, bbox_inches='tight')
    plt.close()

    # Cumulative absolute error
    plt.figure(figsize=(12, 5))
    plt.plot(np.cumsum(np.abs(sim_te - ote)) / 1e6, '--', lw=1.8, label='Baseline')
    plt.plot(np.cumsum(np.abs(p1_te - ote)) / 1e6, lw=1.8, label='Phase 1')
    plt.plot(np.cumsum(np.abs(p2_te - ote)) / 1e6, lw=2.2, label='Phase 2 Online')
    plt.title('Cumulative Absolute Error Over Test Period', fontweight='bold')
    plt.xlabel('Test Month Index')
    plt.ylabel('Cumulative |Error| ($M)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'fig_p1_04_cumulative_error.png'),
                dpi=200, bbox_inches='tight')
    plt.close()

    print(f"  Figures -> {fig_dir}/")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data', default=DEFAULT_DATA)
    ap.add_argument('--out-dir', default=DEFAULT_OUT)
    ap.add_argument('--fig-dir', default=DEFAULT_FIGS)
    ap.add_argument('--sim-reps', type=int, default=SIM_REPS)
    ap.add_argument('--no-figures', action='store_true')
    args = ap.parse_args()

    t0 = time.time()
    np.random.seed(RANDOM_STATE)

    print("=" * 72)
    print("  PART 1 -- B-BICYCLE DIGITAL TWIN CALIBRATION")
    print("=" * 72)

    df = load_field_data(args.data)
    n_train = int(len(df) * TRAIN_FRAC)
    df_train = df.iloc[:n_train].copy().reset_index(drop=True)
    df_test = df.iloc[n_train:].copy().reset_index(drop=True)
    otr = df_train['NetRevenue'].values
    ote = df_test['NetRevenue'].values

    print(f"  Field data : {len(df)} months ({df['Time'].iloc[0]} .. {df['Time'].iloc[-1]})")
    print(f"  Train/test : {len(df_train)} / {len(df_test)} months")
    print(f"  DES reps   : {args.sim_reps} per month")
    print(f"  DES seeds  : train {TRAIN_SEED_BASE}+i, test {TEST_SEED_BASE}+i")
    print("=" * 72)

    sim = BBicycleSimulator(DEFAULT_THETA)
    print("\n  Running baseline DES (uncalibrated) ...")
    sim_tr = sim.predict_batch(df_train, n_reps=args.sim_reps,
                               base_seed=TRAIN_SEED_BASE)
    sim_te = sim.predict_batch(df_test, n_reps=args.sim_reps,
                               base_seed=TEST_SEED_BASE)

    cal = EnsembleCalibrator()
    p1_info = cal.calibrate_phase1(df_train, sim_tr, verbose=True)

    p1_te, p1_te_std = cal.predict_uncertainty(df_test, sim_te)
    p1_stream, p2_te = cal.online_update(df_test, sim_te, verbose=True)

    metrics = {
        'baseline': evaluate(ote, sim_te),
        'phase1': evaluate(ote, p1_te),
        'phase2': evaluate(ote, p2_te),
    }

    print("\n" + "=" * 72)
    print("  TEST-SET RESULTS (72 held-out months, Jan-2020 .. Dec-2025)")
    print("=" * 72)
    print(f"  {'Method':<26} {'RMSE($)':>12} {'MAE($)':>12} {'MAPE':>8} "
          f"{'sMAPE':>8} {'R2':>9}")
    print("  " + "-" * 78)
    for label, key in [("Baseline Simulation", 'baseline'),
                       ("Phase 1: Ensemble", 'phase1'),
                       ("Phase 2: Online Bayesian", 'phase2')]:
        m = metrics[key]
        print(f"  {label:<26} {m['rmse']:>12,.0f} {m['mae']:>12,.0f} "
              f"{m['mape']:>7.1f}% {m['smape']:>7.1f}% {m['r2']:>9.4f}")
    base = metrics['baseline']['rmse']
    for label, key in [("Phase 1", 'phase1'), ("Phase 2", 'phase2')]:
        print(f"\n  -> {label}: {(1 - metrics[key]['rmse'] / base) * 100:.1f}% "
              "RMSE reduction vs baseline")

    # -- write artifacts ---------------------------------------------------
    out = args.out_dir
    os.makedirs(out, exist_ok=True)

    joblib.dump(cal, os.path.join(out, 'calibrator.joblib'))
    save_theta(sim.theta, os.path.join(out, 'simulator_theta.json'))
    df_train.to_csv(os.path.join(out, 'df_train.csv'), index=False)
    df_test.to_csv(os.path.join(out, 'df_test.csv'), index=False)

    arrays = {
        'train_observed': otr,
        'train_sim_preds': sim_tr,
        'test_observed': ote,
        'test_sim_baseline': sim_te,
        'test_p1_preds': p1_te,
        'test_p1_std': p1_te_std,
        'test_p2_preds': p2_te,
    }
    for name, arr in arrays.items():
        np.save(os.path.join(out, f'{name}.npy'), np.asarray(arr))

    summary = {
        # Schema matches the original submitted p1_metrics.json so regenerated
        # artifacts are drop-in compatible with anything that read the originals.
        'train_months': len(df_train),
        'test_months': len(df_test),
        'data_start': str(df['Time'].iloc[0]),
        'train_end': str(df_train['Time'].iloc[-1]),
        'test_start': str(df_test['Time'].iloc[0]),
        'data_end': str(df['Time'].iloc[-1]),

        'baseline_rmse': metrics['baseline']['rmse'],
        'baseline_mae': metrics['baseline']['mae'],
        'baseline_r2': metrics['baseline']['r2'],
        'baseline_mape': metrics['baseline']['mape'],
        'baseline_smape': metrics['baseline']['smape'],

        'p1_test_rmse': metrics['phase1']['rmse'],
        'p1_test_mae': metrics['phase1']['mae'],
        'p1_test_r2': metrics['phase1']['r2'],
        'p1_test_mape': metrics['phase1']['mape'],
        'p1_test_smape': metrics['phase1']['smape'],

        'p2_test_rmse': metrics['phase2']['rmse'],
        'p2_test_mae': metrics['phase2']['mae'],
        'p2_test_r2': metrics['phase2']['r2'],
        'p2_test_mape': metrics['phase2']['mape'],
        'p2_test_smape': metrics['phase2']['smape'],

        'p2_final_bias': float(cal.online_bias_history[-1]),
        'p2_final_scale': float(cal.online_scale_history[-1]),
        'p2_bias_history': [float(v) for v in cal.online_bias_history],
        'p2_scale_history': [float(v) for v in cal.online_scale_history],

        'feature_importances': p1_info['feature_importance'],

        'field_revenue_mean': float(df['NetRevenue'].mean()),
        'field_revenue_std': float(df['NetRevenue'].std()),
        'field_revenue_max': float(df['NetRevenue'].max()),
        'field_revenue_min': float(df['NetRevenue'].min()),

        # Added by this repo; absent from the original artifacts.
        'p1_train_rmse_in_sample': p1_info['train_rmse'],
        'p1_train_r2_in_sample': p1_info['train_r2'],
        'p1_cv5_rmse': p1_info['cv_rmse'],
        'sim_reps': args.sim_reps,
        'train_seed_base': TRAIN_SEED_BASE,
        'test_seed_base': TEST_SEED_BASE,
        'random_state': RANDOM_STATE,
    }
    with open(os.path.join(out, 'p1_metrics.json'), 'w') as fh:
        json.dump(summary, fh, indent=2)

    print(f"\n  Artifacts -> {out}/  ({len(os.listdir(out))} files)")
    print(f"  Phase 2 converged: bias=${summary['p2_final_bias']:,.0f}  "
          f"scale={summary['p2_final_scale']:.4f}")
    print(f"  In-sample train RMSE ${p1_info['train_rmse']:,.0f} vs "
          f"5-fold CV RMSE ${p1_info['cv_rmse']:,.0f}  <- gap is the overfit signal")

    if not args.no_figures:
        make_figures(ote, sim_te, p1_te, p2_te, cal, metrics, args.fig_dir)

    print(f"\n  Done in {time.time() - t0:.1f}s")


if __name__ == '__main__':
    main()
