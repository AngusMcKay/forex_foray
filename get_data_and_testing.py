

from historic_analysis_helpers import GenerateHistoricFeatures
import api_helpers as ah
import pandas as pd
import numpy as np
import sqlalchemy
import datetime
import config_local

'''
get prices data
'''

# credentials
db_pw = config_local.db_pw
acc_id = config_local.acc_id
acc_pw = config_local.acc_pw
api_key = config_local.api_key

# get security details
cst, x_security_token = ah.get_creds(acc_id, acc_pw, api_key)

market_deets = ah.get_market_deets(api_key, cst, x_security_token,
                                epic="CS.D.GBPUSD.TODAY.IP")

# check latest date
connect_string = 'mysql+pymysql://root:'+db_pw+'@localhost/forex'
sql_engine = sqlalchemy.create_engine(connect_string)
latest_data = pd.read_sql("SELECT MAX(snapshotTime) FROM gbpusd_spread_data", con=sql_engine).iloc[0,0]
latest_data_plus_1m = latest_data + pd.Timedelta(minutes=1.5)  # add 1.5 as rounds to nearest 2-minutes
latest_data_plus_1m_str = str(latest_data_plus_1m)[:10] + 'T' + str(latest_data_plus_1m)[11:]

future_time = pd.to_datetime(datetime.datetime.today()) + pd.Timedelta(days=1)  # some future time
future_time_str = str(future_time)[:10] + 'T' + str(future_time)[11:]

prices = ah.get_prices(api_key, cst, x_security_token,
                       epic="CS.D.GBPUSD.TODAY.IP", date_from=latest_data_plus_1m_str, date_to=future_time_str,
                       res='MINUTE_2')

prices_df = ah.convert_prices(prices)

# send to db
prices_df.to_sql(name='gbpusd_spread_data', con=sql_engine, schema='forex', if_exists='append', index=False)



'''
create and check features
'''
connect_string = 'mysql+pymysql://root:'+db_pw+'@localhost/forex'
sql_engine = sqlalchemy.create_engine(connect_string)

prices_df = pd.read_sql("SELECT DISTINCT * FROM gbpusd_spread_data ORDER BY snapshotTime ASC", con=sql_engine)
#prices_df = pd.read_sql("SELECT DISTINCT * FROM gbpusd_spread_data_2hr ORDER BY snapshotTime ASC", con=sql_engine)

pricesPastFeatures = GenerateHistoricFeatures(prices_df)

pricesPastFeatures.add_previous_price_diff('closePrice_mid', 5)
pricesPastFeatures.series[['closePrice_mid', 'previous_price_5', 'previous_price_5_diff']].tail(10)

pricesPastFeatures.add_lowest_price_in_last_x_diff('closePrice_mid', 7)
pricesPastFeatures.series[['closePrice_mid', 'lowest_last_7', 'lowest_last_7_diff', 'lowest_last_7_timedelta']].tail(20)

pricesPastFeatures.add_highest_price_in_last_x_diff('closePrice_mid', 7)
pricesPastFeatures.series[['closePrice_mid', 'highest_last_7', 'highest_last_7_diff', 'highest_last_7_timedelta']].tail(10)

pricesPastFeatures.add_average_price_in_last_x_diff('closePrice_mid', 4)
pricesPastFeatures.series[['closePrice_mid', 'average_last_4', 'average_last_4_diff']].tail(10)

pricesPastFeatures.add_func_price_in_last_x_diff('closePrice_mid', 4, np.median, 'median', include_current_price=True)
pricesPastFeatures.series[['closePrice_mid', 'median_last_4', 'median_last_4_diff']].tail(10)

pricesPastFeatures.regression_slope_last_x('closePrice_mid', 7)
pricesPastFeatures.series[['closePrice_mid', 'reg_slope_last_7']].tail(10)

pricesPastFeatures.regression_slope_since_min_last_x('closePrice_mid', 7)
pricesPastFeatures.series[['closePrice_mid', 'lowest_last_7_timedelta', 'reg_slope_since_min_last_7']].tail(10)

pricesPastFeatures.regression_slope_since_max_last_x('closePrice_mid', 12)
pricesPastFeatures.series[['closePrice_mid', 'highest_last_12_timedelta', 'reg_slope_since_max_last_12']].tail(10)

pricesPastFeatures.number_times_passed_price_last_x_diff('closePrice_mid', 20)
pricesPastFeatures.series[['closePrice_mid', 'closePrice_mid_passed_last_20']].tail(20)

pricesPastFeatures.consecutive_falls('closePrice_mid')
pricesPastFeatures.consecutive_rises('closePrice_mid')
pricesPastFeatures.series[['closePrice_mid', 'consecutive_rises', 'consecutive_falls']].head(10)

pricesPastFeatures.support_resistance_last_x('closePrice_mid', 6, 0.2, '20pc')
pricesPastFeatures.series[['closePrice_mid', 'min_max_range_last_6', 'resistance_rebounds_20pc_last_6']].tail(30)

pricesPastFeatures.moving_avg_short_vs_long('closePrice_mid', 5, 10)
# check by reviewing dataframe

pricesPastFeatures.add_highest_price_in_next_x_diff('closePrice_mid', 15)
pricesPastFeatures.series[['closePrice_mid', 'highest_next_15', 'highest_next_15_diff',
                           #'highest_next_15_timedelta'
                           ]].tail(20)

pricesPastFeatures.add_lowest_price_in_next_x_diff('closePrice_mid', 15)
pricesPastFeatures.series[['closePrice_mid', 'lowest_next_15', 'lowest_next_15_diff',
                           #'highest_next_15_timedelta'
                           ]].tail(20)

pricesPastFeatures.add_future_price_diff('closePrice_mid', 15)
pricesPastFeatures.series[['closePrice_mid', 'future_price_15', 'future_price_15_diff',
                           #'highest_next_15_timedelta'
                           ]].tail(20)

pricesPastFeatures.add_price_after_breach_price_change('closePrice_mid', 15, 1)
pricesPastFeatures.add_price_after_breach_price_change('closePrice_mid', 15, -1)
prices_df[['closePrice_mid', 'price_after_breach_-1_next_15_timedelta']].head(20)


'''
plot some lines
'''
prices_df[['closePrice_mid', 'lowest_last_7', 'highest_last_7', 'average_last_5', 'average_last_10']].iloc[-50:, :].plot()

prices_df[['closePrice_mid', 'lowest_next_15', 'highest_next_15']].iloc[-100:, :].plot()


