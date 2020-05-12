
"""
Trading process:
- Every 5 seconds:
    - check open positions dictionary, if any have passed expiry date then close positions
    - check for new 2-minute price (allowed up to 30/60 non-trading requests per minute, so could go finer)
    - when new price(s) received:
        - do prediction and decision making (including checking number of open positions)
        - if decision is to trade:
            - send trade request
            - confirm trade request
            - if confirmed then update data and details locally and in sql database
        - if decision is not to trade:
            - update data and details locally and in sql database
"""

# import importlib
from historic_analysis_helpers import GenerateHistoricFeatures
# import api_helpers
# importlib.reload(api_helpers)
import api_helpers as ah
# import trading_helpers
# importlib.reload(trading_helpers)
from trading_helpers import GenerateFeatures
import pandas as pd
import numpy as np
import sqlalchemy
import datetime
import pickle
import time
import sys
import signal
import config_local

'''
Initialise
'''

# credentials
db_pw = config_local.db_pw
acc_id = config_local.acc_id
acc_pw = config_local.acc_pw
api_key = config_local.api_key


# function to end while loop gracefully with keyboard interrupt
def signal_handler(signal, frame):
    print("\nprogram exiting gracefully")
    sys.exit(0)
signal.signal(signal.SIGINT, signal_handler)


# get recent data from database
print('getting latest data from db')
connect_string = 'mysql+pymysql://root:'+db_pw+'@localhost/forex'
sql_engine = sqlalchemy.create_engine(connect_string)
latest_data = pd.read_sql("""
                            SELECT *
                            FROM (SELECT DISTINCT * FROM gbpusd_spread_data ORDER BY snapshotTime DESC LIMIT 100) t
                            ORDER BY snapshotTime ASC
                            """,
                          con=sql_engine)


# generate dataframe with features (so that structure is in place for generating new data's features)
print('generating past features')
getPastFeatures = GenerateHistoricFeatures(latest_data)
getPastFeatures.add_spread('closePrice_bid', 'closePrice_ask')
getPastFeatures.add_previous_price_diff('closePrice_mid', 12)
getPastFeatures.add_previous_price_diff('closePrice_mid', 15)
getPastFeatures.add_previous_price_diff('closePrice_mid', 30)
getPastFeatures.add_lowest_price_in_last_x_diff('closePrice_mid', 30)
getPastFeatures.add_highest_price_in_last_x_diff('closePrice_mid', 30)
getPastFeatures.add_average_price_in_last_x_diff('closePrice_mid', 60)
getPastFeatures.add_func_price_in_last_x_diff(
    'closePrice_mid', 60, lambda x: np.quantile(x, 0.25), 'q25', include_current_price=True)
getPastFeatures.add_func_price_in_last_x_diff(
    'closePrice_mid', 60, lambda x: np.quantile(x, 0.75), 'q75', include_current_price=True)
getPastFeatures.add_func_price_in_last_x_diff(
        'closePrice_mid', 10, np.std, 'stdev', include_current_price=True)
getPastFeatures.add_func_price_in_last_x_diff(
        'closePrice_mid', 60, np.std, 'stdev', include_current_price=True)
getPastFeatures.regression_slope_last_x('closePrice_mid', 10)
getPastFeatures.regression_slope_last_x('closePrice_mid', 60)


# load models
print('loading models')
with open('models/voyager_1_l_lin_reg_20200501', 'rb') as f:
    lin_reg_long_dict = pickle.load(f)
lin_reg_long = lin_reg_long_dict['model']
with open('models/voyager_1_l_log_reg_20200501', 'rb') as f:
    log_reg_long_dict = pickle.load(f)
log_reg_long = log_reg_long_dict['model']
with open('models/voyager_1_l_trade_mod_20200501', 'rb') as f:
    trade_mod_long_dict = pickle.load(f)
trade_mod_long = trade_mod_long_dict['model']
with open('models/voyager_1_s_lin_reg_20200501', 'rb') as f:
    lin_reg_short_dict = pickle.load(f)
lin_reg_short = lin_reg_short_dict['model']
with open('models/voyager_1_s_log_reg_20200501', 'rb') as f:
    log_reg_short_dict = pickle.load(f)
log_reg_short = log_reg_short_dict['model']
with open('models/voyager_1_s_trade_mod_20200501', 'rb') as f:
    trade_mod_short_dict = pickle.load(f)
trade_mod_short = trade_mod_short_dict['model']


# get ig security details
cst, x_security_token = ah.get_creds(acc_id, acc_pw, api_key)


# initialise positions dict
trades_dict_long = {}
trades_dict_short = {}
max_open_long = 3
max_open_short = 3
# Note: counts trades as open until their time period expires (i.e. not earlier if they meet stop or limit before)


"""
Loop every 5 seconds to:
 - 1. close expired trades
 - 2. get new data when available
 - 3. check if latest position meets criteria for placing trade
 - 4. place trades
 - 5. send data to database
"""
print('starting trade loop')
while True:
    '''
    1. check and close current trades
    '''
    current_time = pd.to_datetime(datetime.datetime.today())

    for trade in list(trades_dict_long.keys()):
        if trades_dict_long[trade]['close_time'] <= current_time:
            print('attempting to close trade '+trade)
            # check if trade still open
            open_position_response = ah.open_position_details(api_key, cst, x_security_token,
                                                              trades_dict_long[trade]['dealId'])
            open_position_response_json = open_position_response.json()
            open_position_response_json_position = open_position_response_json.get('position', {})
            open_position_dealId = open_position_response_json_position.get('dealId', 'NOT_FOUND')

            if open_position_dealId == 'NOT_FOUND':
                print('trade '+trade+' closed before end datetime, no need to close manually, removing from trades dict')
                del trades_dict_long[trade]

            else:
                # close trade
                close_response = ah.close_trade(api_key, cst, x_security_token,
                                                trades_dict_long[trade]['dealId'], direction='SELL',
                                                size=trades_dict_long[trade]['size'])
                close_response_json = close_response.json()
                close_deal_reference = close_response_json.get('dealReference', 'NOT_FOUND')

                # if closed response returns deal reference or not
                if close_deal_reference == 'NOT_FOUND':
                    print('WARNING: problems closing deal '+trade)
                else:
                    # confirm closed
                    confirm_response = ah.confirm_trade(api_key, cst, x_security_token, dealReference=trade)
                    confirm_response_json = confirm_response.json()

                    # if deal confirmed then remove details from trades_dict
                    deal_status = confirm_response_json.get('dealStatus', 'NOT_FOUND')
                    if deal_status == 'NOT FOUND':
                        print('WARNING: close deal confirmation status for ' + close_deal_reference +
                              ' not found, not removing from trades dict')
                    if deal_status in ['ACCEPTED', 'REJECTED']:
                        deal_close_time = trades_dict_long[trade]['close_time']
                        edited_response_json = confirm_response_json.copy()
                        edited_response_json['close_time'] = deal_close_time

                        if deal_status == 'ACCEPTED':
                            del trades_dict_long[trade]
                            print('trade closed: '+trade)
                        else:
                            print('WARNING: trade '+trade+' close rejected')

                        # send details to db
                        print('sending details to db')
                        edited_response_df = ah.generate_deal_details(edited_response_json)
                        try:
                            edited_response_df.to_sql(
                                name='voyager_1_trade_confirmations', con=sql_engine, schema='forex', if_exists='append', index=False)
                        except:
                            connect_string = 'mysql+pymysql://root:'+db_pw+'@localhost/forex'
                            sql_engine = sqlalchemy.create_engine(connect_string)
                            edited_response_df.to_sql(
                                name='voyager_1_trade_confirmations', con=sql_engine, schema='forex', if_exists='append', index=False)

    number_open_trades_long = len(list(trades_dict_long.keys()))

    for trade in list(trades_dict_short.keys()):
        if trades_dict_short[trade]['close_time'] <= current_time:
            print('attempting to close trade ' + trade)
            # check if trade still open
            open_position_response = ah.open_position_details(api_key, cst, x_security_token,
                                                              trades_dict_short[trade]['dealId'])
            open_position_response_json = open_position_response.json()
            open_position_response_json_position = open_position_response_json.get('position', {})
            open_position_dealId = open_position_response_json_position.get('dealId', 'NOT_FOUND')

            if open_position_dealId == 'NOT_FOUND':
                print('trade '+trade+' closed before end datetime, no need to close manually, removing from trades dict')
                del trades_dict_long[trade]

            else:
                # close trade
                close_response = ah.close_trade(api_key, cst, x_security_token,
                                                trades_dict_short[trade]['dealId'], direction='BUY',
                                                size=trades_dict_short[trade]['size'])
                close_response_json = close_response.json()
                close_deal_reference = close_response_json.get('dealReference', 'NOT_FOUND')

                # if closed response returns deal reference or not
                if close_deal_reference == 'NOT_FOUND':
                    print('WARNING: problems closing deal ' + trade)
                else:
                    # confirm closed
                    confirm_response = ah.confirm_trade(api_key, cst, x_security_token, dealReference=close_deal_reference)
                    confirm_response_json = confirm_response.json()

                    # if deal confirmed then remove details from trades_dict
                    deal_status = confirm_response_json.get('dealStatus', 'NOT_FOUND')
                    if deal_status == 'NOT FOUND':
                        print('WARNING: close deal confirmation status for ' + close_deal_reference +
                              ' not found, not removing from trades dict')
                    if deal_status in ['ACCEPTED', 'REJECTED']:
                        deal_close_time = trades_dict_short[trade]['close_time']
                        edited_response_json = confirm_response_json.copy()
                        edited_response_json['close_time'] = deal_close_time

                        if deal_status == 'ACCEPTED':
                            del trades_dict_short[trade]
                            print('trade closed: ' + trade)
                        else:
                            print('WARNING: trade ' + trade + ' close rejected')

                        # send details to db
                        print('sending details to db')
                        edited_response_df = ah.generate_deal_details(edited_response_json)
                        try:
                            edited_response_df.to_sql(
                                name='voyager_1_trade_confirmations', con=sql_engine, schema='forex', if_exists='append', index=False)
                        except:
                            connect_string = 'mysql+pymysql://root:'+db_pw+'@localhost/forex'
                            sql_engine = sqlalchemy.create_engine(connect_string)
                            edited_response_df.to_sql(
                                name='voyager_1_trade_confirmations', con=sql_engine, schema='forex', if_exists='append', index=False)

    number_open_trades_short = len(list(trades_dict_short.keys()))


    '''
    2. get current data
    '''
    # get current data
    latest_datetime = latest_data['snapshotTime'].iloc[-1]
    latest_datetime_plus_1m = latest_datetime + pd.Timedelta(minutes=1.5)  # add 1.5 as rounds to nearest 2-minutes
    latest_datetime_plus_1m_str = str(latest_datetime_plus_1m)[:10] + 'T' + str(latest_datetime_plus_1m)[11:]

    future_time = pd.to_datetime(datetime.datetime.today()) + pd.Timedelta(days=1)  # some arbitrary future end time
    future_time_str = str(future_time)[:10] + 'T' + str(future_time)[11:]

    try:
        prices = ah.get_prices(api_key, cst, x_security_token,
                               epic="CS.D.GBPUSD.TODAY.IP", date_from=latest_datetime_plus_1m_str, date_to=future_time_str,
                               res='MINUTE_2')
    except:
        cst, x_security_token = ah.get_creds(acc_id, acc_pw, api_key)
        prices = ah.get_prices(api_key, cst, x_security_token,
                               epic="CS.D.GBPUSD.TODAY.IP", date_from=latest_datetime_plus_1m_str, date_to=future_time_str,
                               res='MINUTE_2')


    '''
    3/4/5 check if current data meets criteria to place trade, place trade and send details to db
    '''
    if len(prices) > 0:
        print('new price received, generating prediction data')
        current_prices_df = ah.convert_prices(prices)
        latest_data = pd.concat([latest_data, current_prices_df], sort=False, ignore_index=True).sort_values(
            'snapshotTime', ascending=True)

        # add current features
        getCurrentFeatures = GenerateFeatures(latest_data)
        getCurrentFeatures.add_spread('closePrice_bid', 'closePrice_ask')
        getCurrentFeatures.add_previous_price_diff('closePrice_mid', 12)
        getCurrentFeatures.add_previous_price_diff('closePrice_mid', 15)
        getCurrentFeatures.add_previous_price_diff('closePrice_mid', 30)
        getCurrentFeatures.add_lowest_price_in_last_x_diff('closePrice_mid', 30)
        getCurrentFeatures.add_highest_price_in_last_x_diff('closePrice_mid', 30)
        getCurrentFeatures.add_average_price_in_last_x_diff('closePrice_mid', 60)
        getCurrentFeatures.add_func_price_in_last_x_diff(
            'closePrice_mid', 60, lambda x: np.quantile(x, 0.25), 'q25', include_current_price=True)
        getCurrentFeatures.add_func_price_in_last_x_diff(
            'closePrice_mid', 60, lambda x: np.quantile(x, 0.75), 'q75', include_current_price=True)
        getCurrentFeatures.add_func_price_in_last_x_diff(
            'closePrice_mid', 10, np.std, 'stdev', include_current_price=True)
        getCurrentFeatures.add_func_price_in_last_x_diff(
            'closePrice_mid', 60, np.std, 'stdev', include_current_price=True)
        getCurrentFeatures.regression_slope_last_x('closePrice_mid', 10)
        getCurrentFeatures.regression_slope_last_x('closePrice_mid', 60)

        # predictions
        print('making predictions')
        prediction_X = np.array([1] + list(latest_data[lin_reg_long_dict['features']].iloc[-1, :]))
        lin_reg_long_pred = lin_reg_long.predict(prediction_X)[0]
        log_reg_long_pred = log_reg_long.predict(prediction_X)[0]
        trade_mod_long_pred = np.matmul(prediction_X, trade_mod_long)
        lin_reg_short_pred = lin_reg_short.predict(prediction_X)[0]
        log_reg_short_pred = log_reg_short.predict(prediction_X)[0]
        trade_mod_short_pred = np.matmul(prediction_X, trade_mod_short)

        # long decision
        print('making decision long')
        print('lin reg long: '+str(lin_reg_long_pred))
        print('log reg long: ' + str(log_reg_long_pred))
        print('trade mod long: ' + str(trade_mod_long_pred))
        if number_open_trades_long < max_open_long:
            if (lin_reg_long_pred > 0) and (log_reg_long_pred > 0.5) and (trade_mod_long_pred > 0):
                print('criteria met, attempting to place trade')
                latest_price_date = latest_data['snapshotTime'].iloc[-1]
                deal_reference = 'long'+latest_price_date.strftime("%Y%m%d%H%M%S")
                trade_response = ah.place_trade(api_key, cst, x_security_token,
                                                dealReference=deal_reference, epic="CS.D.GBPUSD.TODAY.IP", direction='BUY',
                                                expiry='DFB', orderType='MARKET', size=1,
                                                limitDistance=50, stopDistance=15)
                confirm_response = ah.confirm_trade(api_key, cst, x_security_token, dealReference=deal_reference)
                confirm_response_json = confirm_response.json()

                # if deal confirmed then store details in trades_dict
                deal_status = confirm_response_json.get('dealStatus', 'NOT_FOUND')
                if deal_status == 'NOT FOUND':
                    print('WARNING: deal confirmation for '+deal_reference+' not received')
                if deal_status in ['ACCEPTED', 'REJECTED']:
                    deal_close_time = latest_price_date + pd.Timedelta(minutes=40)
                    edited_response_json = confirm_response_json.copy()
                    edited_response_json['close_time'] = deal_close_time

                    if deal_status == 'ACCEPTED':
                        trades_dict_long[deal_reference] = edited_response_json
                        print('TRADE: placed trade '+deal_reference)
                    elif deal_status == 'REJECTED':
                        print('WARNING: deal '+deal_reference+' rejected')

                    # send details to db
                    edited_response_df = ah.generate_deal_details(edited_response_json)
                    try:
                        edited_response_df.to_sql(
                            name='voyager_1_trade_confirmations', con=sql_engine, schema='forex', if_exists='append', index=False)
                    except:
                        connect_string = 'mysql+pymysql://root:'+db_pw+'@localhost/forex'
                        sql_engine = sqlalchemy.create_engine(connect_string)
                        edited_response_df.to_sql(
                            name='voyager_1_trade_confirmations', con=sql_engine, schema='forex', if_exists='append', index=False)
            else:
                print('no trade (predictions too low)')
        else:
            print('no trade (positions full)')

        # short decision
        print('making decision short')
        print('lin reg short: ' + str(lin_reg_short_pred))
        print('log reg short: ' + str(log_reg_short_pred))
        print('trade mod short: ' + str(trade_mod_short_pred))
        if number_open_trades_short < max_open_short:
            if (lin_reg_short_pred > 0) and (log_reg_short_pred > 0.5) and (trade_mod_short_pred > 0):
                print('criteria met, attempting to place trade')
                latest_price_date = latest_data['snapshotTime'].iloc[-1]
                deal_reference = 'short' + latest_price_date.strftime("%Y%m%d%H%M%S")
                trade_response = ah.place_trade(api_key, cst, x_security_token,
                                                dealReference=deal_reference, epic="CS.D.GBPUSD.TODAY.IP", direction='SELL',
                                                expiry='DFB', orderType='MARKET', size=1,
                                                limitDistance=50, stopDistance=15)
                confirm_response = ah.confirm_trade(api_key, cst, x_security_token, dealReference=deal_reference)
                confirm_response_json = confirm_response.json()

                # if deal confirmed then store details in trades_dict
                deal_status = confirm_response_json.get('dealStatus', 'NOT_FOUND')
                if deal_status == 'NOT FOUND':
                    print('WARNING: deal confirmation for '+deal_reference+' not received')
                if deal_status in ['ACCEPTED', 'REJECTED']:
                    deal_close_time = latest_price_date + pd.Timedelta(minutes=40)
                    edited_response_json = confirm_response_json.copy()
                    edited_response_json['close_time'] = deal_close_time

                    if deal_status == 'ACCEPTED':
                        trades_dict_short[deal_reference] = edited_response_json
                        print('TRADE: placed trade ' + deal_reference)
                    elif deal_status == 'REJECTED':
                        print('WARNING: deal ' + deal_reference + ' rejected')

                    # send details to db
                    edited_response_df = ah.generate_deal_details(edited_response_json)
                    try:
                        edited_response_df.to_sql(
                            name='voyager_1_trade_confirmations', con=sql_engine, schema='forex', if_exists='append', index=False)
                    except:
                        connect_string = 'mysql+pymysql://root:'+db_pw+'@localhost/forex'
                        sql_engine = sqlalchemy.create_engine(connect_string)
                        edited_response_df.to_sql(
                            name='voyager_1_trade_confirmations', con=sql_engine, schema='forex', if_exists='append', index=False)
            else:
                print('no trade (predictions too low)')
        else:
            print('no trade (positions full)')

        # send prices to db (doing this after to minimise time between getting latest data and placing trades)
        print('sending latest prices to db')
        try:
            current_prices_df.to_sql(name='gbpusd_spread_data', con=sql_engine, schema='forex', if_exists='append', index=False)
        except:
            connect_string = 'mysql+pymysql://root:'+db_pw+'@localhost/forex'
            sql_engine = sqlalchemy.create_engine(connect_string)
            current_prices_df.to_sql(name='gbpusd_spread_data', con=sql_engine, schema='forex', if_exists='append',
                                     index=False)

    latest_data = latest_data.drop_duplicates(subset=['snapshotTime', 'closePrice_bid', 'closePrice_ask'])

    time.sleep(5)










