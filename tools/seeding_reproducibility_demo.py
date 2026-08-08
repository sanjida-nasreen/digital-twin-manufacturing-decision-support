"""
Reproduce the Part 1 seeding defect, exactly as notebooks/digital_twin_1.ipynb
had it: one np.random.seed(42) at import, every DES draw taken from the global
stream thereafter.

    python tools/demo_seeding_bug.py

Three tests: repeated calls with identical inputs disagree; re-running the
pipeline in one session gives different answers; and a row's value changes with
its position in the batch. See docs/seeding.md for the fix.
"""
import numpy as np
import pandas as pd

THETA = {'assembly_mu':540,'assembly_sigma':25,'cleaning_mu':180,
         'cleaning_sigma':20,'inspection_mu':60,'inspection_sigma':14,
         'rework_mu':300,'rework_sigma':40,'fail_rate':0.12,'arrival_rate':6.25}


def _run(C, R, Q, W, M1, M2, n_days=30):
    """Verbatim from digital_twin_1.ipynb — draws from the global np.random stream."""
    nr = -50000.0*(M1+M2); inv = 50
    for day in range(n_days):
        for _ in range(np.random.poisson(THETA['arrival_rate']*24)):
            if inv <= R: inv += Q; nr -= 100.0*Q
            if inv <= 0: continue
            inv -= 1
            ft = (max(10, np.random.normal(THETA['assembly_mu'], THETA['assembly_sigma']))
                + max(5,  np.random.normal(THETA['cleaning_mu'], THETA['cleaning_sigma']))/max(M1,1)
                + max(3,  np.random.normal(THETA['inspection_mu'], THETA['inspection_sigma']))/max(M2,1))
            rn = 0
            while np.random.random() < THETA['fail_rate'] and rn < 3:
                ft += max(10, np.random.normal(THETA['rework_mu'], THETA['rework_sigma']))
                ft += max(3,  np.random.normal(THETA['inspection_mu'], THETA['inspection_sigma']))/max(M2,1)
                rn += 1; nr -= 25.0
            fm = ft/60.0
            nr += (1000-5*fm) if C == 0 else (500-2*fm)
            nr -= 1.5*(ft/3600.0)
    return nr


def simulate(C, R, Q, W, M1, M2, n_reps=3):
    return np.mean([_run(C, R, Q, W, M1, M2) for _ in range(n_reps)])


x = dict(C=0, R=134, Q=21, W=1, M1=4, M2=2)

print("TEST 1 — same config called repeatedly after ONE seed (notebook behaviour)")
np.random.seed(42)
vals = [simulate(**x) for _ in range(5)]
for i, v in enumerate(vals):
    print(f"    call {i+1}: ${v:,.2f}")
print(f"    spread: ${max(vals)-min(vals):,.2f}\n")

print("TEST 2 — re-running the whole pipeline twice in one session")
np.random.seed(42)
first = simulate(**x)
second_pass_first_call = simulate(**x)
print(f"    1st pipeline run, 1st config: ${first:,.2f}")
print(f"    2nd pipeline run, 1st config: ${second_pass_first_call:,.2f}")
print(f"    identical? {first == second_pass_first_call}\n")

print("TEST 3 — does row ORDER change a row's value?")
df = pd.DataFrame([
    dict(C=0, R=134, Q=21, W=1,  M1=4, M2=2),
    dict(C=1, R=80,  Q=50, W=20, M1=2, M2=3),
])
np.random.seed(42)
fwd = [simulate(**r) for r in df.to_dict('records')]
np.random.seed(42)
rev = [simulate(**r) for r in df.iloc[::-1].to_dict('records')][::-1]
print(f"    row 0 evaluated 1st: ${fwd[0]:,.2f}   evaluated 2nd: ${rev[0]:,.2f}")
print(f"    row 1 evaluated 2nd: ${fwd[1]:,.2f}   evaluated 1st: ${rev[1]:,.2f}")
print(f"    order-independent? {fwd == rev}")
