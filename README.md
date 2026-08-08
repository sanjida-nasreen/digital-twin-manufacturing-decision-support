# Data-Driven Digital Twin for Manufacturing Decision Support

A manufacturing digital twin that combines **discrete-event simulation, machine learning calibration, online Bayesian synchronization, and simulation-based optimization** to support production and inventory decisions.

The project models a bicycle manufacturing system and calibrates the simulation against 30 years of monthly field data. The calibrated twin reduces prediction error by about **93%** compared with the original simulation and is then used to search for improved operating policies.

---

## Project Overview

The original discrete-event simulation represented the production structure but did not match observed system performance well.

On the 72-month held-out test period, the uncalibrated simulator produced:

| Metric | Baseline |
|---|---:|
| RMSE | $2.28M |
| R² | -7.07 |
| sMAPE | 129.0% |

A negative R² means the simulation performed worse than simply predicting the historical mean revenue.

To address this simulation-to-reality gap, the project uses a two-phase synchronization framework:

1. **Offline ensemble calibration**
2. **Online recursive Bayesian updating**

The calibrated twin is then used inside a Bayesian optimization workflow for manufacturing decision support.

---

## Manufacturing System

The B-Bicycle system follows a sequential production flow:

```text
Supplier
   ↓
Receiving
   ↓
Buffer
   ↓
Assembly
   ↓
Cleaning
   ↓
Inspection ─── Pass ───→ Warehouse / Shipping
   ↓
  Fail
   ↓
Rework
   ↓
Re-inspection
(maximum 3 rework attempts)
