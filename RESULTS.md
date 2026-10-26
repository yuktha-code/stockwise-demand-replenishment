# Executed results - 16 October 2026

50 series; 1,941 observed days each; three 28-day pre-test windows and a final
28-day test (25 April through 22 May 2016). See the manifest for all chosen IDs,
source archive hash and software versions. Models and hyperparameters were
fixed before evaluation; there was no tuning on the final test.

| Method | Mean validation WAPE | Test WAPE | Test MAE | Test signed bias |
|---|---:|---:|---:|---:|
| Seasonal naive (7-day) | 0.8266 | 0.7748 | 1.5871 | -0.0363 |
| Trailing 28-day mean | **0.6721** | 0.6802 | 1.3934 | -0.0192 |
| Poisson histogram boosting | 0.6854 | **0.6573** | **1.3466** | -0.0167 |

Validation selected the trailing-mean method. Boosting performed better on the
later test; do not retroactively call it the validation-selected method.
Its test WAPE was about 15.2% lower than seasonal naive and about 3.4% lower
than the trailing mean on this sample. These are model-error comparisons,
not financial savings or performance on the complete M5 benchmark.

## Inventory scenario comparison

This is a separate rolling-daily replenishment simulation over the same 28-day
test period, with forecasts updated using prior-day observations. Holding cost
is 1/unit/day and lost-sales penalty is 10/unit, both assumed unitless costs.

| Lead time | Method | Unit fill rate | Lost units | Mean inventory per SKU/day | Assumed cost |
|---|---|---:|---:|---:|---:|
| 3 days | Seasonal naive | 94.49% | 158 | 7.488 | 12,063 |
| 3 days | Trailing mean | 94.21% | 166 | 7.196 | 11,735 |
| 3 days | Poisson boosting | 95.54% | 128 | 7.286 | 11,481 |
| 7 days | Seasonal naive | 95.68% | 124 | 12.379 | 18,570 |
| 7 days | Trailing mean | 93.48% | 187 | 11.376 | 17,797 |
| 7 days | Poisson boosting | 95.82% | 120 | 11.346 | 17,085 |

The relative ordering depends on the specified lead time, buffer, initial state
and cost assumptions. These scenarios do not demonstrate a universal policy
improvement, calibrated service guarantee, pharma suitability or deployed ROI.

Three automated tests passed: recursive forecast feedback, future-demand
isolation of previous orders, and correct lead-time receipts/stock conservation.
SQLite unique keys and runtime data/quantity assertions passed in the full run.
