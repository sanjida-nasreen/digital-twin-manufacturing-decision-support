"""
Part 2 -- Simulation-based optimization over the calibrated B-Bicycle twin.

Usage
-----
    python src/part2_optimize.py                  # ~12-15 min
    python src/part2_optimize.py --n-calls 40     # quick smoke run

Loads the frozen Part 1 artifacts and searches six decision variables
(C, R, Q, W, M1, M2) with Gaussian-Process Bayesian Optimization using
Expected Improvement. Nothing is retrained: the calibrator, simulator theta,
and the Phase 2 bias/scale scalars are read from p1_artifacts/ and held fixed.

Then: OAT sensitivity analysis, validation against the raw DES as a physical
system proxy (the reality gap), and a value-of-synchronization comparison
against a grid search on the uncalibrated DES.


SEEDING -- three levels, deliberately different from Part 1
-----------------------------------------------------------
Level 1  np.random.seed(SEED) once at import. The DES draws from the global
         stream and is NEVER re-seeded, so the seed is consumed
         deterministically from top to bottom. A fresh run of this script
         reproduces itself exactly, because main() executes a fixed sequence
         of DES calls.

Level 2  RNG = np.random.default_rng(SEED), a separate Generator used only for
         BO candidate sampling, so the candidate pool does not shift when the
         number of DES draws changes.

Level 3  random_state=SEED on the GaussianProcessRegressor.

Part 1 uses per-row seeding instead (see simulator.py). The difference is
intentional and is why simulate() takes seed=None here: this file is what
reproduced the submitted report, and re-seeding it would change every number.
The cost is that Level 1 is order-fragile -- inserting a single extra DES call
anywhere shifts everything downstream. Treat this file as frozen.
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
import warnings

from scipy import stats
from scipy.stats import norm as sp_norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from simulator import BBicycleSimulator, load_theta
from features import create_features
from calibrator import EnsembleCalibrator

# The committed calibrator.joblib was pickled from a Jupyter notebook, so the
# pickle records the class path as __main__.EnsembleCalibrator. Registering the
# alias lets joblib resolve it when loading from a script. Artifacts produced by
# part1_calibrate.py record calibrator.EnsembleCalibrator and do not need this,
# but the shim must stay for the committed originals to remain loadable.
sys.modules['__main__'].EnsembleCalibrator = EnsembleCalibrator

warnings.filterwarnings('ignore')

SEED = 42
np.random.seed(SEED)                 # Level 1: DES stochasticity
RNG = np.random.default_rng(SEED)    # Level 2: BO candidate sampling

MACHINE_BUDGET = 300_000             # $300K/month  ->  M1 + M2 <= 6
BO_PENALTY = -5_000_000.0            # revenue assigned to infeasible candidates

VAR_BOUNDS = np.array([[0, 1], [10, 200], [10, 200], [1, 50], [1, 5], [1, 5]],
                      dtype=float)
VAR_NAMES = ['C', 'R', 'Q', 'W', 'M1', 'M2']

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DEFAULT_P1 = os.path.join(ROOT, 'p1_artifacts')
DEFAULT_OUT = os.path.join(ROOT, 'results')


# --------------------------------------------------------------------------
# Section 1 -- load Part 1 artifacts
# --------------------------------------------------------------------------

def load_p1_artifacts(p1_dir):
    print("=" * 62)
    print("  LOADING PROJECT 1 ARTIFACTS")
    print("=" * 62)

    cal = joblib.load(os.path.join(p1_dir, 'calibrator.joblib'))

    with open(os.path.join(p1_dir, 'p1_metrics.json')) as fh:
        m = json.load(fh)

    # Read the converged Phase 2 parameters from JSON rather than from a
    # possibly-absent attribute on the unpickled object.
    cal.p2_bias = m['p2_final_bias']
    cal.p2_scale = m['p2_final_scale']
    print(f"  calibrator.joblib     Phase2 bias=${cal.p2_bias:,.2f}  "
          f"scale={cal.p2_scale:.6f}")

    theta = load_theta(os.path.join(p1_dir, 'simulator_theta.json'))
    print(f"  simulator_theta.json  ({len(theta)} params)")

    arrs = {k: np.load(os.path.join(p1_dir, f'{k}.npy'))
            for k in ['test_observed', 'test_sim_baseline', 'test_p1_preds',
                      'test_p1_std', 'test_p2_preds', 'train_observed',
                      'train_sim_preds']}
    print(f"  prediction arrays     (test={len(arrs['test_observed'])} months)")

    df_train = pd.read_csv(os.path.join(p1_dir, 'df_train.csv'))
    df_test = pd.read_csv(os.path.join(p1_dir, 'df_test.csv'))
    print(f"  df_train({len(df_train)})  df_test({len(df_test)})")
    print(f"  p1_metrics.json       P1 R2={m['p1_test_r2']:.4f}  "
          f"RMSE=${m['p1_test_rmse']:,.0f}")
    print("=" * 62)

    return cal, theta, arrs, df_train, df_test, m


# --------------------------------------------------------------------------
# Section 2 -- digital twin evaluator
# --------------------------------------------------------------------------

class DigitalTwinEvaluator:
    """DES -> ensemble calibration -> frozen Phase 2 correction."""

    def __init__(self, cal, sim: BBicycleSimulator, sim_n_reps: int = 3):
        self.cal = cal
        self.sim = sim
        self.sim_reps = sim_n_reps
        self.n_calls = 0
        self.call_log = []

    def _twin_from_sim(self, C, R, Q, W, M1, M2, y_sim):
        row = pd.DataFrame([{'C': C, 'R': R, 'Q': Q, 'W': W, 'M1': M1, 'M2': M2,
                             'NetRevenue': 0.0, 'Standard Deviation': 20000.0}])
        feat = create_features(row)
        feat['sim_pred'] = y_sim
        X = self.cal.scaler.transform(feat.values)

        rf_p = self.cal.rf.predict(X)
        gbr_p = self.cal.gbr.predict(X)
        stack = np.column_stack([rf_p, gbr_p, np.array([y_sim])])
        mean_p1, std_br = self.cal.bayesian_ridge.predict(stack, return_std=True)

        tree_p = np.array([t.predict(X) for t in self.cal.rf.estimators_])
        rf_std = float(np.std(tree_p, axis=0)[0])
        total_std = float(np.sqrt(float(std_br[0]) ** 2 + rf_std ** 2))

        revenue = float(self.cal.p2_scale * float(mean_p1[0]) + self.cal.p2_bias)
        return revenue, total_std

    def evaluate(self, C, R, Q, W, M1, M2):
        # seed=None -> global np.random stream (Level 1). See module docstring.
        y_sim = self.sim.simulate(C, R, Q, W, M1, M2,
                                  n_reps=self.sim_reps, seed=None)
        revenue, total_std = self._twin_from_sim(C, R, Q, W, M1, M2, y_sim)
        self.n_calls += 1
        self.call_log.append((C, R, Q, W, M1, M2, revenue, total_std))
        return revenue, total_std

    def evaluate_scalar(self, C, R, Q, W, M1, M2):
        return self.evaluate(C, R, Q, W, M1, M2)[0]


# --------------------------------------------------------------------------
# Section 3 -- Bayesian optimization
# --------------------------------------------------------------------------

def _normalise(X):
    return (X - VAR_BOUNDS[:, 0]) / (VAR_BOUNDS[:, 1] - VAR_BOUNDS[:, 0])


def _random_candidates(n):
    return np.array([[RNG.integers(int(lo), int(hi) + 1) for lo, hi in VAR_BOUNDS]
                     for _ in range(n)], dtype=float)


def _feasible(C, R, Q, W, M1, M2):
    return 50_000 * (int(M1) + int(M2)) <= MACHINE_BUDGET


def _expected_improvement(gp, X_cand, best_y, xi=0.01):
    mu, sig = gp.predict(_normalise(X_cand), return_std=True)
    sig = np.maximum(sig, 1e-9)
    Z = (mu - best_y - xi) / sig
    ei = (mu - best_y - xi) * sp_norm.cdf(Z) + sig * sp_norm.pdf(Z)
    ei[sig < 1e-9] = 0.0
    return ei


def run_bayesian_optimisation(evaluator, n_calls=120, n_initial=20,
                              n_pool=2000, verbose=True):
    """GP-EI loop. Returns (x_star, best_history, df_log)."""
    kernel = Matern(nu=2.5) + WhiteKernel(noise_level=1e-3,
                                          noise_level_bounds=(1e-6, 1e-1))
    X_obs, y_obs = [], []
    best_y, best_x = BO_PENALTY, None
    best_history, log = [], []

    print("\n" + "=" * 62)
    print("  BAYESIAN OPTIMISATION -- B-BICYCLE DIGITAL TWIN")
    print("=" * 62)
    print(f"  Budget      : {n_calls} evals "
          f"({n_initial} random + {n_calls - n_initial} GP-guided)")
    print(f"  Constraint  : M1+M2 <= 6  (budget $300K/month)")
    print(f"  Acquisition : Expected Improvement, xi=0.01")
    print(f"  GP kernel   : Matern-5/2 + WhiteKernel")
    print(f"  random_state: {SEED}")
    print()

    for i in range(n_calls):
        if i < n_initial:
            for _ in range(500):
                cand = _random_candidates(1)[0]
                if _feasible(*cand):
                    break
        else:
            gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True,
                                          n_restarts_optimizer=3,
                                          random_state=SEED)
            Xa = np.array(X_obs)
            ya = np.array(y_obs)
            ys = max(np.std(ya), 1.0)
            gp.fit(_normalise(Xa), ya / ys)

            pool = _random_candidates(n_pool)
            ei = _expected_improvement(gp, pool, best_y / ys)
            for j, row in enumerate(pool):
                if not _feasible(*row):
                    ei[j] = 0.0
            cand = pool[np.argmax(ei)]

        C, R, Q, W, M1, M2 = (int(cand[0]), int(cand[1]), int(cand[2]),
                              int(cand[3]), int(cand[4]), int(cand[5]))
        feas = _feasible(C, R, Q, W, M1, M2)
        rev = evaluator.evaluate_scalar(C, R, Q, W, M1, M2) if feas else BO_PENALTY

        X_obs.append(cand.copy())
        y_obs.append(rev)
        if feas and rev > best_y:
            best_y, best_x = rev, cand.copy()
        best_history.append(best_y)
        log.append(dict(iter=i + 1, C=C, R=R, Q=Q, W=W, M1=M1, M2=M2,
                        revenue=rev if feas else np.nan,
                        feasible=feas, best_so_far=best_y))

        if verbose and (i + 1) % 10 == 0:
            print(f"  Iter {i + 1:>3d} | C={C} R={R:>3d} Q={Q:>3d} W={W:>2d} "
                  f"M1={M1} M2={M2} | Rev=${rev:>12,.0f} | Best=${best_y:>12,.0f}")

    x_star = {n: int(v) for n, v in zip(VAR_NAMES, best_x)}
    return x_star, best_history, pd.DataFrame(log)


# --------------------------------------------------------------------------
# Section 4 -- sensitivity, validation, value of synchronization
# --------------------------------------------------------------------------

def sensitivity_analysis(evaluator, x_star):
    """One-at-a-time sweep of each variable with the others held at x*."""
    print("\n  OAT sensitivity analysis ...")
    sweep = {
        'C': ([0, 1], "Contract Type"),
        'R': (list(range(10, 201, 10)), "Reorder Point"),
        'Q': (list(range(10, 201, 10)), "Order Quantity"),
        'W': (list(range(1, 51, 2)), "Max WIP"),
        'M1': ([1, 2, 3, 4, 5], "Assembly Machines"),
        'M2': ([1, 2, 3, 4, 5], "Rework Machines"),
    }
    base_rev = evaluator.evaluate_scalar(**x_star)
    results = {}
    for var, (levels, label) in sweep.items():
        revs = []
        for val in levels:
            cfg = {**x_star, var: val}
            revs.append(evaluator.evaluate_scalar(**cfg)
                        if _feasible(**cfg) else np.nan)
        results[var] = {'label': label, 'levels': levels, 'revenues': revs}
        print(f"    {var:3s}  {label:<22s}  {len(levels)} levels")
    return base_rev, results


def validate_against_physical(sim, evaluator, x_star, n_reps=30):
    """Run x* on the raw DES as a physical proxy; compare to the twin."""
    print("\n" + "=" * 62)
    print("  VALIDATION -- REALITY GAP")
    print("=" * 62)
    C, R, Q, W, M1, M2 = (x_star['C'], x_star['R'], x_star['Q'],
                          x_star['W'], x_star['M1'], x_star['M2'])
    print(f"  x* = {x_star}")
    print(f"  Running {n_reps} DES replications ...")

    raw = sim.replicate(C, R, Q, W, M1, M2, n_days=30, n_reps=n_reps, seed=None)
    mu = float(np.mean(raw))
    sd = float(np.std(raw))
    ci = stats.t.interval(0.95, df=n_reps - 1, loc=mu, scale=stats.sem(raw))

    twin, twin_std = evaluator._twin_from_sim(C, R, Q, W, M1, M2, mu)

    gap_abs = twin - mu
    gap_pct = abs(gap_abs) / (abs(mu) + 1e-9) * 100
    inside = bool(ci[0] <= twin <= ci[1])

    print(f"\n  {'Metric':<40} {'Value':>18}")
    print("  " + "-" * 58)
    print(f"  {'Physical mean (DES)':<40} ${mu:>17,.0f}")
    print(f"  {'Physical std':<40} ${sd:>17,.0f}")
    print(f"  {'Physical 95% CI':<40} [${ci[0]:,.0f} -- ${ci[1]:,.0f}]")
    print(f"  {'Digital twin prediction':<40} ${twin:>17,.0f}")
    print(f"  {'Twin uncertainty (1 sigma)':<40} ${twin_std:>17,.0f}")
    print(f"  {'Absolute gap (twin - physical)':<40} ${gap_abs:>17,.0f}")
    print(f"  {'Relative gap':<40} {gap_pct:>17.2f}%")
    print(f"  {'Twin inside physical 95% CI':<40} {'YES' if inside else 'NO':>18}")

    return dict(raw=raw, mu=mu, sd=sd, ci=ci, twin=twin, twin_std=twin_std,
                gap_abs=gap_abs, gap_pct=gap_pct, inside=inside)


def value_of_synchronisation(evaluator, sim, x_star, best_revenue):
    """Grid search on the raw DES, then score that optimum on the twin."""
    print("\n  Computing value of synchronisation ...")
    C_vals = [0, 1]
    # (2,2) appears twice. It is a duplicate in the original coursework code.
    # Removing it would change how many DES calls are drawn from the global
    # stream and therefore shift every downstream number, so it stays.
    M_pairs = [(1, 1), (2, 1), (1, 2), (2, 2), (3, 1), (1, 3), (3, 2), (2, 3),
               (3, 3), (4, 1), (1, 4), (4, 2), (2, 4), (2, 2), (5, 1), (1, 5)]
    R_vals = [20, 50, 80, 100, 130, 160, 200]
    Q_vals = [20, 50, 80, 100, 130, 160, 200]
    W_vals = [5, 10, 20, 30, 40, 50]

    best_raw_rev, best_raw_cfg = -np.inf, None
    for C in C_vals:
        for M1, M2 in M_pairs:
            if not _feasible(C, 10, 10, 1, M1, M2):
                continue
            for R in R_vals:
                for Q in Q_vals:
                    for W in W_vals:
                        rev = sim.simulate(C, R, Q, W, M1, M2, n_reps=3, seed=None)
                        if rev > best_raw_rev:
                            best_raw_rev = rev
                            best_raw_cfg = {'C': C, 'R': R, 'Q': Q,
                                            'W': W, 'M1': M1, 'M2': M2}

    twin_of_raw = evaluator.evaluate_scalar(**best_raw_cfg)
    gain = best_revenue - twin_of_raw

    print(f"  Uncalibrated DES optimum  : {best_raw_cfg}")
    print(f"  DES score of raw optimum  : ${best_raw_rev:,.0f}")
    print(f"  Twin score of raw optimum : ${twin_of_raw:,.0f}")
    print(f"  Calibrated twin optimum   : ${best_revenue:,.0f}")
    print(f"  Gain from synchronisation : ${gain:,.0f}")

    return dict(raw_cfg=best_raw_cfg, raw_rev=best_raw_rev,
                twin_of_raw=twin_of_raw, gain=gain)


def training_coverage(df_train, x_star):
    """Where x* sits inside the training distribution -- the reality-gap diagnosis."""
    feat = create_features(df_train)
    rows = []
    for var in VAR_NAMES + ['Q_x_W', 'W_sq']:
        if var in VAR_NAMES:
            val = float(x_star[var])
        elif var == 'Q_x_W':
            val = float(x_star['Q'] * x_star['W'])
        else:
            val = float(x_star['W'] ** 2)
        col = feat[var].values
        # Inclusive percentile rank, matching Table 7 of the Part 2 report.
        # A strict '<' would place C=0 at the 0th percentile even though half
        # the training months run C=0; ties must count.
        pct = float((col <= val).mean() * 100)
        flag = ''
        if val < col.min() or val > col.max():
            flag = 'out-of-distribution'
        elif pct <= 2 or pct >= 98:
            flag = 'extreme'
        elif pct <= 10 or pct >= 90:
            flag = 'tail'
        rows.append({'variable': var, 'x_star': val,
                     'train_mean': float(col.mean()),
                     'train_min': float(col.min()),
                     'train_max': float(col.max()),
                     'percentile': pct, 'flag': flag})
    return rows


# --------------------------------------------------------------------------
# Section 5 -- figures
# --------------------------------------------------------------------------

def _fmt_m(x, _):
    return f'${x / 1e6:.2f}M'


def _save(fig_dir, fname):
    path = os.path.join(fig_dir, fname)
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def plot_p1_recap(arrs, m, fig_dir):
    ote = arrs['test_observed']
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    sets = [("Baseline Simulation", arrs['test_sim_baseline'], '#999999',
             m['baseline_r2'], m['baseline_rmse']),
            ("Phase 1 Calibrated", arrs['test_p1_preds'], '#2E5496',
             m['p1_test_r2'], m['p1_test_rmse']),
            ("Phase 2 Online Bayesian", arrs['test_p2_preds'], '#1A5276',
             m['p2_test_r2'], m['p2_test_rmse'])]
    for ax, (title, pred, col, r2, rmse) in zip(axes, sets):
        lo = min(ote.min(), pred.min())
        hi = max(ote.max(), pred.max())
        ax.scatter(ote, pred, alpha=0.5, s=18, color=col)
        ax.plot([lo, hi], [lo, hi], 'r--', lw=1.2)
        ax.set_title(f'{title}\nRMSE=${rmse:,.0f}  R2={r2:.3f}',
                     fontsize=10, fontweight='bold')
        ax.set_xlabel('Observed ($)')
        ax.set_ylabel('Predicted ($)')
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(_fmt_m))
        ax.yaxis.set_major_formatter(plt.FuncFormatter(_fmt_m))
    plt.suptitle('Project 1 Digital Twin -- Test-Set Performance (loaded from artifacts)',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    _save(fig_dir, 'fig_p2_00_p1_recap.png')


def plot_convergence(best_history, df_log, best_revenue, n_initial, fig_dir):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax = axes[0]
    ax.plot(best_history, color='#2E5496', lw=2)
    ax.axvline(n_initial, color='gray', ls=':', lw=1.2, label='BO phase starts')
    ax.axhline(best_revenue, color='#C00000', ls='--', lw=1.5,
               label=f'Optimum = ${best_revenue:,.0f}')
    ax.fill_between(range(len(best_history)), best_history, alpha=0.12,
                    color='#2E5496')
    ax.set_xlabel('BO Iteration')
    ax.set_ylabel('Best Revenue Found')
    ax.set_title('Convergence: Best Revenue vs. Iteration', fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(_fmt_m))

    ax2 = axes[1]
    feas = df_log[df_log['feasible']]
    infeas = df_log[~df_log['feasible']]
    ax2.scatter(feas['iter'], feas['revenue'], alpha=0.5, s=18,
                color='#2E5496', label='Feasible')
    if len(infeas):
        ax2.scatter(infeas['iter'], [0] * len(infeas), alpha=0.3, s=12,
                    color='gray', marker='x', label='Infeasible')
    ax2.axhline(best_revenue, color='#C00000', ls='--', lw=1.5,
                label=f'Optimum = ${best_revenue:,.0f}')
    ax2.set_xlabel('BO Iteration')
    ax2.set_ylabel('Evaluated Revenue')
    ax2.set_title('All Candidate Evaluations', fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(_fmt_m))
    plt.tight_layout()
    _save(fig_dir, 'fig_p2_01_convergence.png')


def plot_sensitivity(base_rev, sens, x_star, fig_dir):
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    for ax, var in zip(axes.flatten(), sens):
        info = sens[var]
        levels = info['levels']
        revs = [0 if np.isnan(r) else r for r in info['revenues']]
        cols = ['#AAAAAA' if np.isnan(info['revenues'][j]) else '#2E5496'
                for j in range(len(levels))]
        ax.bar(range(len(levels)), revs, color=cols, alpha=0.8)
        ax.axhline(base_rev, color='#C00000', ls='--', lw=1.5, label='x* optimum')
        if x_star[var] in levels:
            oi = levels.index(x_star[var])
            ax.bar(oi, revs[oi], color='#C00000', alpha=1.0,
                   label=f'x*={x_star[var]}')
        ax.set_xticks(range(len(levels)))
        ax.set_xticklabels([str(l) for l in levels], fontsize=8, rotation=45)
        ax.set_title(f"{info['label']} ({var})", fontweight='bold')
        ax.set_ylabel('Revenue ($)')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')
        ax.yaxis.set_major_formatter(plt.FuncFormatter(_fmt_m))
    plt.suptitle('Sensitivity Analysis (OAT): Revenue vs. Each Decision Variable',
                 fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()
    _save(fig_dir, 'fig_p2_02_sensitivity.png')


def plot_validation(val, fig_dir):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax = axes[0]
    ax.hist(val['raw'], bins=12, color='#2E5496', alpha=0.75, edgecolor='white')
    ax.axvline(val['mu'], color='navy', lw=2,
               label=f"Physical mean = ${val['mu']:,.0f}")
    ax.axvline(val['twin'], color='#C00000', lw=2, ls='--',
               label=f"Twin pred = ${val['twin']:,.0f}")
    ax.axvline(val['ci'][0], color='gray', lw=1, ls=':')
    ax.axvline(val['ci'][1], color='gray', lw=1, ls=':', label='Physical 95% CI')
    ax.set_xlabel('Net Revenue ($)')
    ax.set_ylabel('Count')
    ax.set_title('Physical System Distribution at x*', fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(_fmt_m))

    ax2 = axes[1]
    vals_ = [val['mu'], val['twin']]
    bars = ax2.bar(['Physical\n(DES mean)', 'Digital Twin\n(Phase 1+2)'],
                   vals_, color=['#2E5496', '#C00000'], alpha=0.85, width=0.4)
    ci_h = (val['ci'][1] - val['ci'][0]) / 2
    ax2.errorbar([0], [val['mu']], yerr=ci_h, fmt='none', color='black',
                 capsize=8, lw=2)
    ax2.errorbar([1], [val['twin']], yerr=1.96 * val['twin_std'], fmt='none',
                 color='black', capsize=8, lw=2)
    for bar, v in zip(bars, vals_):
        ax2.text(bar.get_x() + bar.get_width() / 2, v + max(vals_) * 0.005,
                 f'${v:,.0f}', ha='center', va='bottom', fontsize=10,
                 fontweight='bold')
    ax2.set_ylabel('Net Revenue ($)')
    ax2.set_title(f"Reality Gap: {val['gap_pct']:.2f}%  (${val['gap_abs']:,.0f})",
                  fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(_fmt_m))
    plt.tight_layout()
    _save(fig_dir, 'fig_p2_03_validation.png')


def plot_value_of_sync(sync, best_revenue, fig_dir):
    fig, ax = plt.subplots(figsize=(10, 5))
    labels = ['Uncalibrated DES\nOptimum\n(DES score)',
              'Uncalibrated DES\nOptimum\n(Twin score)',
              'Calibrated Twin\nOptimum\n(Twin score)']
    vals = [sync['raw_rev'], sync['twin_of_raw'], best_revenue]
    bars = ax.bar(labels, vals, color=['#AAAAAA', '#7090C0', '#2E5496'],
                  alpha=0.88, width=0.5)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2,
                v + (max(vals) - min(vals)) * 0.01, f'${v:,.0f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax.annotate('', xy=(2, best_revenue), xytext=(1, sync['twin_of_raw']),
                arrowprops=dict(arrowstyle='<->', color='#C00000', lw=2))
    ax.text(1.5, (best_revenue + sync['twin_of_raw']) / 2,
            f" Gain = ${sync['gain']:,.0f}", ha='left', va='center',
            color='#C00000', fontweight='bold')
    ax.set_ylabel('Net Revenue ($)')
    ax.set_title('Value of Digital Twin Synchronisation\n'
                 'Calibrated vs Uncalibrated Optimisation', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.yaxis.set_major_formatter(plt.FuncFormatter(_fmt_m))
    plt.tight_layout()
    _save(fig_dir, 'fig_p2_04_value_of_sync.png')


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--p1-dir', default=DEFAULT_P1)
    ap.add_argument('--out-dir', default=DEFAULT_OUT)
    ap.add_argument('--n-calls', type=int, default=120)
    ap.add_argument('--n-initial', type=int, default=20)
    ap.add_argument('--val-reps', type=int, default=30)
    ap.add_argument('--skip-sync', action='store_true',
                    help='skip the value-of-synchronization grid (the slow part)')
    args = ap.parse_args()

    t0 = time.time()
    os.makedirs(args.out_dir, exist_ok=True)

    print("\n" + "=" * 62)
    print("  PART 2 -- SIMULATION-BASED OPTIMISATION")
    print("=" * 62)

    cal, theta, arrs, df_train, df_test, m = load_p1_artifacts(args.p1_dir)
    sim = BBicycleSimulator(theta)
    evaluator = DigitalTwinEvaluator(cal, sim, sim_n_reps=3)

    print("\n  Generating P1 recap figure ...")
    plot_p1_recap(arrs, m, args.out_dir)

    x_star, best_history, df_log = run_bayesian_optimisation(
        evaluator, n_calls=args.n_calls, n_initial=args.n_initial, verbose=True)
    best_revenue, best_std = evaluator.evaluate(**x_star)

    print("\n" + "=" * 62)
    print("  OPTIMAL SOLUTION x*")
    print("=" * 62)
    for k in VAR_NAMES:
        extra = ''
        if k == 'C':
            extra = f"  ({'Standard' if x_star['C'] == 0 else 'Expedited'})"
        print(f"  {k:<3}= {x_star[k]}{extra}")
    print(f"  Machine investment = ${50_000 * (x_star['M1'] + x_star['M2']):,}/month")
    print(f"  Twin revenue       = ${best_revenue:,.0f}  "
          f"(95% CI ${best_revenue - 1.96 * best_std:,.0f} -- "
          f"${best_revenue + 1.96 * best_std:,.0f})")
    print(f"  Historical mean    = ${m['field_revenue_mean']:,.0f}")
    print(f"  Improvement        = ${best_revenue - m['field_revenue_mean']:,.0f}  "
          f"(+{(best_revenue / m['field_revenue_mean'] - 1) * 100:.1f}%)")

    base_rev, sens = sensitivity_analysis(evaluator, x_star)
    val = validate_against_physical(sim, evaluator, x_star, n_reps=args.val_reps)

    if args.skip_sync:
        sync = dict(raw_cfg=None, raw_rev=float('nan'),
                    twin_of_raw=float('nan'), gain=float('nan'))
    else:
        sync = value_of_synchronisation(evaluator, sim, x_star, best_revenue)

    coverage = training_coverage(df_train, x_star)
    print("\n  Training-data coverage at x* (reality-gap diagnosis):")
    print(f"    {'var':<8}{'x*':>10}{'train mean':>13}{'percentile':>12}   flag")
    for r in coverage:
        print(f"    {r['variable']:<8}{r['x_star']:>10,.0f}"
              f"{r['train_mean']:>13,.1f}{r['percentile']:>11.1f}%   {r['flag']}")

    print("\n  Generating figures ...")
    plot_convergence(best_history, df_log, best_revenue, args.n_initial, args.out_dir)
    plot_sensitivity(base_rev, sens, x_star, args.out_dir)
    plot_validation(val, args.out_dir)
    if not args.skip_sync:
        plot_value_of_sync(sync, best_revenue, args.out_dir)

    df_log.to_csv(os.path.join(args.out_dir, 'p2_bo_log.csv'), index=False)

    summary = {
        'x_star': x_star,
        'best_revenue_twin': float(best_revenue),
        'best_revenue_std': float(best_std),
        'best_revenue_ci95_lo': float(best_revenue - 1.96 * best_std),
        'best_revenue_ci95_hi': float(best_revenue + 1.96 * best_std),
        'machine_cost_per_month': 50_000 * (x_star['M1'] + x_star['M2']),
        'hist_mean': m['field_revenue_mean'],
        'hist_max': m['field_revenue_max'],
        'improvement_abs': float(best_revenue - m['field_revenue_mean']),
        'improvement_pct': float((best_revenue / m['field_revenue_mean'] - 1) * 100),
        'physical_mean': val['mu'],
        'physical_std': val['sd'],
        'physical_ci95_lo': float(val['ci'][0]),
        'physical_ci95_hi': float(val['ci'][1]),
        'gap_abs': val['gap_abs'],
        'gap_pct': val['gap_pct'],
        'twin_inside_physical_ci': bool(val['inside']),
        'uncalib_cfg': sync['raw_cfg'],
        'uncalib_raw_rev': float(sync['raw_rev']),
        'uncalib_twin_rev': float(sync['twin_of_raw']),
        'sync_gain': float(sync['gain']),
        'bo_total_evals': int(df_log.shape[0]),
        'bo_feasible_evals': int(df_log['feasible'].sum()),
        'dt_evaluations': int(evaluator.n_calls),
        'convergence_last10': [float(v) for v in best_history[-10:]],
        'training_coverage': coverage,
        'p1_test_r2': m['p1_test_r2'],
        'p1_test_rmse': m['p1_test_rmse'],
        'p2_final_bias': m['p2_final_bias'],
        'p2_final_scale': m['p2_final_scale'],
        'runtime_seconds': round(time.time() - t0, 1),
    }
    with open(os.path.join(args.out_dir, 'p2_results.json'), 'w') as fh:
        json.dump(summary, fh, indent=2)
    print(f"  Saved: {os.path.join(args.out_dir, 'p2_results.json')}")

    print("\n" + "=" * 62)
    print("  FINAL SUMMARY")
    print("=" * 62)
    print(f"  DT evaluations   : {evaluator.n_calls}")
    print(f"  x*               : {x_star}")
    print(f"  Twin revenue     : ${best_revenue:,.0f}")
    print(f"  Physical revenue : ${val['mu']:,.0f}")
    print(f"  Reality gap      : {val['gap_pct']:.2f}%")
    print(f"  Sync gain        : ${sync['gain']:,.0f}")
    print(f"  Runtime          : {summary['runtime_seconds']}s")


if __name__ == '__main__':
    main()
