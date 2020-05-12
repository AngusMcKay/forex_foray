CREATE DATABASE forex;
USE forex;

SELECT * FROM gbpusd_spread_data LIMIT 100;

ALTER TABLE gbpusd_spread_data DROP row_id;
