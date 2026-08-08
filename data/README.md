# Data

This folder contains the datasets and simulation inputs used in the B-Bicycle Digital Twin project.

Only `FieldData.txt` is read directly by the Python calibration pipeline. The processing-time logs and SimTalk file document the Siemens Tecnomatix Plant Simulation side of the project.

| File | Rows | Used by Python? | Purpose |
|---|---:|---|---|
| `FieldData.txt` | 360 | Yes | Monthly observations used for calibration |
| `AssemblyLog.txt` | 2,000 | No | Assembly processing-time observations |
| `CleaningLog.txt` | 2,000 | No | Cleaning processing-time observations |
| `InspectionLog.txt` | 2,000 | No | Inspection processing-time observations |
| `Init_SimTalk.txt` | — | No | Tecnomatix initialization logic |

---

## FieldData.txt

`FieldData.txt` is a tab-separated dataset containing 360 monthly observations from January 1996 through December 2025.

Each row records the operating policy for that month, the resulting net revenue, and the reported revenue variability.

| Column | Type | Range | Description |
|---|---|---|---|
| `Time` | string | Jan-96 to Dec-25 | Month |
| `C` | binary | 0 or 1 | Contract type |
| `R` | integer | 10–200 | Reorder point |
| `Q` | integer | 10–200 | Order quantity |
| `W` | integer | 1–50 | Maximum work-in-process |
| `M1` | integer | 1–5 | Number of assembly machines |
| `M2` | integer | 1–5 | Number of inspection/rework machines |
| `NetRevenue` | float | −$1.36M to $2.76M | Monthly net revenue |
| `Standard Deviation` | float | — | Revenue variability for that observation |

The `Standard Deviation` column is used during Phase 2 Bayesian updating to weight observations according to their uncertainty.

### Summary

- Mean revenue: approximately **$422K**
- Revenue range: **−$1.36M to $2.76M**
- Training set: first **288 months** (Jan-1996 to Dec-2019)
- Test set: final **72 months** (Jan-2020 to Dec-2025)

The split is chronological so that the model is evaluated on future observations rather than on randomly mixed historical data.

---

## Processing-Time Logs

The assembly, cleaning, and inspection logs contain 2,000 processing-time observations each, measured in seconds.

| Process | Observations | Mean | Std. Dev. | Simulator Parameters |
|---|---:|---:|---:|---|
| Assembly | 2,000 | 540.77 s | 23.40 s | μ = 540, σ = 25 |
| Cleaning | 2,000 | 179.81 s | 18.85 s | μ = 180, σ = 20 |
| Inspection | 2,000 | 60.17 s | 13.51 s | μ = 60, σ = 14 |

These values support the processing-time parameters used in the Python simulator.

The rework parameters are not derived from a separate processing-time log. The model uses:

```text
Rework mean = 300 seconds
Rework standard deviation = 40 seconds
Failure probability = 12%
Arrival rate = 6.25 per hour
