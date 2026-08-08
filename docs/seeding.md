# Seeding and reproducibility

The original Part 1 notebook was not reproducible. This document records what
was wrong, how to see it, and what the fix changed. Run
`python tools/demo_seeding_bug.py` to reproduce the evidence below.

## The defect

`digital_twin_1.ipynb` called `np.random.seed(42)` once, in the imports cell.
Every random draw inside `BBicycleSimulator._run` then came from the
session-wide `np.random` stream:

```python
np.random.seed(42)          # imports cell, executed once
...
for _ in range(np.random.poisson(...)):        # inside _run
    ... np.random.normal(...) ...
    while np.random.random() < fail_rate: ...
```

A single seed at the top of a program is enough only when the program is a
straight line executed once. A notebook is neither. Because the stream advances
with every draw, each simulated month depended on how many draws had already
happened, which meant it depended on which cells had been run and in what order.

The notebook also contains the `pipeline.run(...)` cell **twice**. The second
execution therefore trained on a different set of simulated inputs than the
first, and whichever one happened to run last is what got exported.

## Evidence

Same configuration `C=0, R=134, Q=21, W=1, M1=4, M2=2`, five consecutive calls
after one `np.random.seed(42)`:

| call | simulated net revenue |
|---|---|
| 1 | $3,528,596.25 |
| 2 | $3,449,879.41 |
| 3 | $3,471,240.91 |
| 4 | $3,428,019.36 |
| 5 | $3,488,877.01 |

Spread $100,577 across identical inputs.

Worse, a row's value depended on its position in the batch:

| row | evaluated 1st | evaluated 2nd |
|---|---|---|
| `C=0, R=134, Q=21, W=1, M1=4, M2=2` | $3,528,596.25 | $3,449,879.41 |
| `C=1, R=80, Q=50, W=20, M1=2, M2=3` | $1,448,802.79 | $1,412,747.69 |

Reversing the order of a two-row batch changed both rows' values. The training
matrix fed to the ensemble was therefore a function of row ordering, not just of
the decision variables.

## The fix

`simulator.py` takes an optional `seed`:

```python
sim.simulate(C, R, Q, W, M1, M2, n_reps=3, seed=1042)
```

- `seed=None` draws from the global `np.random` stream (the original behaviour).
- `seed=<int>` draws from a private `np.random.RandomState(seed)`.

`RandomState` is used rather than `default_rng` deliberately: it exposes the
same method names (`poisson`, `normal`, `random`) as the `np.random` module, so
both modes run through one code path and the two branches cannot silently
diverge.

`predict_batch(df, base_seed=B)` gives row *i* the seed `B + i`. Part 1 uses
`TRAIN_SEED_BASE = 1000` and `TEST_SEED_BASE = 500000`.

### Verified properties

Running `src/part1_calibrate.py` twice into different directories gives
**all 12 artifacts bit-identical**, including the pickled `calibrator.joblib`:

```
MATCH   calibrator.joblib          MATCH   test_p1_std.npy
MATCH   df_test.csv                MATCH   test_p2_preds.npy
MATCH   df_train.csv               MATCH   test_sim_baseline.npy
MATCH   p1_metrics.json            MATCH   train_observed.npy
MATCH   simulator_theta.json       MATCH   train_sim_preds.npy
MATCH   test_observed.npy          MATCH   test_p1_preds.npy
```

Beyond run-to-run stability, each month's value is now independent of
evaluation order and of any unrelated simulation:

| row | forward order | reverse order | after burning the global stream |
|---|---|---|---|
| 0 | 1,356,003.54 | 1,356,003.54 | 1,356,003.54 |
| 1 | 3,271,714.27 | 3,271,714.27 | 3,271,714.27 |
| 2 | 3,452,231.60 | 3,452,231.60 | 3,452,231.60 |

Results are also identical under scikit-learn 1.5.1 and 1.8.0.

## Why Part 2 was deliberately left alone

`part2_optimize.py` still uses the global-stream approach: `np.random.seed(42)`
once at import, never re-seeded, so the stream is consumed deterministically
from top to bottom.

This is sound *for this file* because `main()` is a single linear sequence — a
fresh process reproduces itself exactly. It is what produced the submitted Part 2
report, and re-seeding it would change every number in that report.

The cost is fragility. Inserting, removing, or reordering a single DES call
anywhere in `main()` shifts the stream and therefore every downstream result.
Two places where this bites:

- The `M_pairs` list in `value_of_synchronisation` contains `(2, 2)` twice. It
  is a genuine duplicate in the original coursework code, but removing it would
  change the draw count, so it is kept with a comment.
- `--n-calls` and `--skip-sync` change the number of draws, so smoke runs will
  not match the full run's numbers. That is expected.

Treat `part2_optimize.py` as frozen. If it ever needs restructuring, the right
move is to give the evaluator a per-configuration seed — which would make the
twin a deterministic function of `x` and remove the order dependence entirely,
at the cost of no longer matching the submitted report.

## Effect on the reported numbers

The seeding fix changes the simulated inputs, so retraining produces slightly
different metrics. Both sit within simulation noise and neither changes any
conclusion. See the README section "Which artifacts are committed, and why" for
the side-by-side comparison.
