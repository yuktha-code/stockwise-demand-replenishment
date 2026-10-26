# Data and methodological references

## Data

Original benchmark: M5 Forecasting Competition, M Competitions / Makridakis Open
Forecasting Center; retail sales made available by Walmart.

- Official competition resources: https://github.com/Mcompetitions/M5-methods
- Nixtla dataset loader documentation:
  https://nixtlaverse.nixtla.io/datasetsforecast/m5.html
- Download mirror used by this run:
  https://raw.githubusercontent.com/Nixtla/m5-forecasts/main/datasets/m5.zip

Files read: `sales_train_evaluation.csv` and `calendar.csv`. This mirror omits
the redundant original ID and calendar d_N columns; this implementation builds
the ID as item_id + '_' + store_id and uses the ordered calendar dates.
The source archive's SHA-256 is recorded in the run manifest.

The data remains US retail data. Selected records, dates and origin must not be
relabelled as Indian markets, pharmaceutical products or employer/client data.
Use data in accordance with the original competition/data-provider conditions.
This package includes experiment outputs, not the original dataset archive.

## Method references

- Nixtla MLForecast examples and time-series cross-validation concepts:
  https://github.com/Nixtla/mlforecast
  https://nixtlaverse.nixtla.io/mlforecast/docs/how-to-guides/cross_validation.html
- Stockpyl inventory-policy and simulation references:
  https://stockpyl.readthedocs.io/en/latest/tutorial/tutorial_sim.html

These are credited references for public problem formulations and concepts.
The implementation uses scikit-learn directly, not the Nixtla or Stockpyl APIs.
Those projects are credited as methodological references; their implementations
are not bundled in this repository.
