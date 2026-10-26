CREATE UNIQUE INDEX forecast_key ON forecasts(split, fold, method, series_id, date);
CREATE UNIQUE INDEX inventory_key ON inventory_daily(method, lead_time, series_id, date);

CREATE VIEW inventory_summary AS
SELECT method, lead_time,
       SUM(fulfilled) / NULLIF(SUM(sales_proxy), 0) AS unit_fill_rate,
       SUM(lost_sales) AS lost_sales,
       AVG(on_hand) AS mean_daily_inventory_per_series,
       SUM(assumed_cost) AS assumed_total_cost,
       SUM(ordered) AS ordered_units
FROM inventory_daily GROUP BY method, lead_time;

CREATE VIEW sku_risk AS
SELECT method, lead_time, series_id,
       SUM(lost_sales) AS lost_sales,
       SUM(fulfilled) / NULLIF(SUM(sales_proxy), 0) AS unit_fill_rate,
       AVG(on_hand) AS mean_inventory
FROM inventory_daily GROUP BY method, lead_time, series_id;

CREATE VIEW forecast_error_by_series AS
SELECT method, series_id, AVG(ABS(actual - forecast)) AS test_mae,
       SUM(forecast - actual) / NULLIF(SUM(actual), 0) AS test_bias
FROM forecasts WHERE split = 'test' GROUP BY method, series_id;
