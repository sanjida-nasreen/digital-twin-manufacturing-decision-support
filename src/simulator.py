"""
Discrete-event simulator for the B-Bicycle production and fulfillment line.

Flow:  Supplier -> Receiving -> Buffer -> Assembly -> Cleaning -> Inspection
                                                         |
                                          pass -> Warehouse / Shipping
                                          fail -> Rework -> re-Inspection
                                                  (max 3 attempts, then scrapped)

Revenue over a 30-day horizon:
    + UnitPrice        1000 - 5*f  (C=0, Standard)   or   500 - 2*f  (C=1, Expedited)
                       where f is flow time in minutes
    - OrderingCost     100 per unit ordered, triggered when inventory <= R
    - ReworkPenalty    25 per rework cycle
    - MachineCost      50,000 * (M1 + M2) per month
    - HoldingCost      1.50 per flow-hour per unit


SEEDING
-------
The original notebook drew every random number from the session-wide
``np.random`` stream after a single ``np.random.seed(42)`` at import. That made
each DES result depend on how many draws had happened before it, so results
changed when a cell was re-run and even depended on the ORDER of rows in a
batch. See ``docs/seeding.md`` for a demonstration.

This module keeps both behaviours, selected by the ``seed`` argument:

    seed=None  -> draw from the global ``np.random`` stream (legacy behaviour).
                  Part 2 uses this so it still reproduces the submitted report.

    seed=<int> -> draw from a private ``np.random.RandomState(seed)``.
                  The call becomes a deterministic, order-independent function
                  of its arguments. Part 1 uses this.

``RandomState`` is used rather than ``default_rng`` because it exposes the same
method names as the ``np.random`` module, so a single code path serves both
modes and the two branches stay provably identical in structure.
"""

from __future__ import annotations

import json
import numpy as np


# Synchronized DES parameters. Processing-time means and standard deviations
# were fitted from the Tecnomatix logs in data/ (see data/README.md); fail_rate
# and arrival_rate come from the course model specification.
DEFAULT_THETA = {
    'assembly_mu': 540,
    'assembly_sigma': 25,
    'cleaning_mu': 180,
    'cleaning_sigma': 20,
    'inspection_mu': 60,
    'inspection_sigma': 14,
    'rework_mu': 300,
    'rework_sigma': 40,
    'fail_rate': 0.12,
    'arrival_rate': 6.25,
}

MAX_REWORK_ATTEMPTS = 3
REWORK_PENALTY = 25.0
MACHINE_COST = 50_000.0
ORDER_COST_PER_UNIT = 100.0
HOLDING_COST_PER_HOUR = 1.50
INITIAL_INVENTORY = 50


def load_theta(path: str) -> dict:
    """Load a theta dict from JSON, filling any missing key from DEFAULT_THETA."""
    with open(path) as fh:
        theta = json.load(fh)
    return {**DEFAULT_THETA, **theta}


def save_theta(theta: dict, path: str) -> None:
    with open(path, 'w') as fh:
        json.dump(theta, fh, indent=2)


class BBicycleSimulator:
    """DES of the B-Bicycle line. One ``simulate`` call = one monthly figure."""

    def __init__(self, theta: dict | None = None):
        self.theta = dict(DEFAULT_THETA) if theta is None else dict(theta)

    # -- internals ---------------------------------------------------------

    @staticmethod
    def _resolve_rng(seed):
        """None -> global np.random stream; int -> private RandomState."""
        if seed is None:
            return np.random
        if isinstance(seed, np.random.RandomState):
            return seed
        return np.random.RandomState(int(seed))

    def _run(self, C, R, Q, W, M1, M2, n_days=30, rng=None):
        """One replication. ``rng`` is anything exposing poisson/normal/random."""
        if rng is None:
            rng = np.random
        t = self.theta
        nr = -MACHINE_COST * (M1 + M2)
        inv = INITIAL_INVENTORY

        for _ in range(n_days):
            for _ in range(rng.poisson(t['arrival_rate'] * 24)):
                if inv <= R:
                    inv += Q
                    nr -= ORDER_COST_PER_UNIT * Q
                if inv <= 0:
                    continue
                inv -= 1

                ft = (max(10, rng.normal(t['assembly_mu'], t['assembly_sigma']))
                      + max(5, rng.normal(t['cleaning_mu'], t['cleaning_sigma'])) / max(M1, 1)
                      + max(3, rng.normal(t['inspection_mu'], t['inspection_sigma'])) / max(M2, 1))

                rn = 0
                while rng.random() < t['fail_rate'] and rn < MAX_REWORK_ATTEMPTS:
                    ft += max(10, rng.normal(t['rework_mu'], t['rework_sigma']))
                    ft += max(3, rng.normal(t['inspection_mu'], t['inspection_sigma'])) / max(M2, 1)
                    rn += 1
                    nr -= REWORK_PENALTY

                fm = ft / 60.0
                nr += (1000 - 5 * fm) if C == 0 else (500 - 2 * fm)
                nr -= HOLDING_COST_PER_HOUR * (ft / 3600.0)

        return nr

    # -- public API --------------------------------------------------------

    def simulate(self, C, R, Q, W, M1, M2, n_days=30, n_reps=5, seed=None):
        """Mean net revenue over ``n_reps`` replications of ``n_days`` days.

        With an integer ``seed`` this is a pure function of its arguments: the
        replications differ from one another, but the returned mean is
        identical on every call and independent of any other simulation.
        """
        rng = self._resolve_rng(seed)
        return float(np.mean([
            self._run(int(C), int(R), int(Q), int(W), int(M1), int(M2), n_days, rng)
            for _ in range(n_reps)
        ]))

    def replicate(self, C, R, Q, W, M1, M2, n_days=30, n_reps=30, seed=None):
        """Return the individual replication values instead of their mean.

        Used by the Part 2 validation step, which needs the spread of the
        'physical system' distribution rather than a point estimate.
        """
        rng = self._resolve_rng(seed)
        return [
            self._run(int(C), int(R), int(Q), int(W), int(M1), int(M2), n_days, rng)
            for _ in range(n_reps)
        ]

    def predict_batch(self, df, n_reps=3, base_seed=None):
        """Simulate one monthly figure per row of ``df``.

        With ``base_seed`` set, row *i* uses ``base_seed + i``, so each row's
        value depends only on its own decision variables and its position --
        never on how many draws preceded it. Without it, the whole batch is
        drawn sequentially from the global stream (legacy behaviour).
        """
        out = []
        for i, (_, r) in enumerate(df.iterrows()):
            seed = None if base_seed is None else base_seed + i
            out.append(self.simulate(
                int(r['C']), int(r['R']), int(r['Q']),
                int(r['W']), int(r['M1']), int(r['M2']),
                n_reps=n_reps, seed=seed,
            ))
        return np.array(out)
