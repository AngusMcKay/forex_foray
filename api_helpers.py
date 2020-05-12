"""
api functions for getting data from IG
"""

import requests
import json
import pandas as pd
import numpy as np


resolutions = ["SECOND", "MINUTE", "MINUTE_2", "MINUTE_3", "MINUTE_5", "MINUTE_10", "MINUTE_15", "MINUTE_30",
               "HOUR", "HOUR_2", "HOUR_3", "HOUR_4", "DAY", "WEEK", "MONTH"]

epics = {
    'GBPUSD_spread': 'CS.D.GBPUSD.TODAY.IP',
    'GBPUSD_CFD': 'CS.D.GBPUSD.CFD.IP'
}


def get_creds(id, pw, api_key):
    """
    gets required credentials for use in later calls
    """

    url = 'https://demo-api.ig.com/gateway/deal/session'

    body = {
        'identifier': id,
        'password': pw
    }

    header = {
        'Content-Type': 'application/json',
        'Accept': 'application/json',
        'VERSION': '2',
        'X-IG-API-KEY': api_key
    }

    response = requests.post(url, data=json.dumps(body), headers=header)
    cst = response.headers['CST']
    x_security_token = response.headers['X-SECURITY-TOKEN']

    return cst, x_security_token


def get_market_deets(api_key, cst, x_security_token, epic):
    """
    Get details for a market
    """
    url = "https://demo-api.ig.com/gateway/deal/markets/" + epic

    header = {
        'Content-Type': 'application/json',
        'Accept': 'application/json',
        'VERSION': '3',
        'X-IG-API-KEY': api_key,
        'CST': cst,
        'X-SECURITY-TOKEN': x_security_token
    }

    response = requests.get(url, headers=header)
    response_json = response.json()

    return response_json


def get_prices(api_key, cst, x_security_token, epic, date_from, date_to, res, qty_max='100000'):
    """
    Note: max doesn't do anything just now because dates are set
    """
    base_url = "https://demo-api.ig.com/gateway/deal/prices/"
    params = epic + "?resolution=" + res + "&from=" + date_from + "&to=" + date_to + "&max=" + str(qty_max) + "&pageSize=0&pageNumber=1"
    url = base_url + params

    header = {
        'Content-Type': 'application/json',
        'Accept': 'application/json',
        'VERSION': '3',
        'X-IG-API-KEY': api_key,
        'CST': cst,
        'X-SECURITY-TOKEN': x_security_token
    }

    response = requests.get(url, headers=header)
    response_json = response.json()

    remaining_allowance = response_json['metadata']['allowance']['remainingAllowance']
    print('remaining allowance: ' + str(remaining_allowance))

    prices = response_json['prices']

    return prices


def populate_missing_values(prices_df, bid_col, ask_col, populate_forward_and_backward=False):
    # where one of ask or bid is missing, add/subtract spread to/from the other
    spread = np.mean(
        prices_df.loc[
            (prices_df[bid_col].notnull()) & (prices_df[ask_col].notnull()), bid_col] -
        prices_df.loc[
            (prices_df[bid_col].notnull()) & (prices_df[ask_col].notnull()), ask_col]
    )

    prices_df.loc[
        (prices_df[bid_col].notnull()) & (prices_df[ask_col].isnull()), ask_col] = (
            prices_df.loc[
                (prices_df[
                     bid_col].notnull()) & (
                    prices_df[
                        ask_col].isnull()), bid_col] + spread)

    prices_df.loc[
        (prices_df[bid_col].isnull()) & (prices_df[ask_col].notnull()), bid_col] = (
            prices_df.loc[
                (prices_df[
                     bid_col].isnull()) & (
                    prices_df[
                        ask_col].notnull()), ask_col] - spread)

    # where both missing, populate with previous price
    prices_df['row_id'] = range(len(prices_df))
    missing_row_ids = prices_df.loc[(prices_df[bid_col].isnull()) & (prices_df[ask_col].isnull()), 'row_id']
    for i in missing_row_ids:
        prices_df.loc[prices_df['row_id'] == i, bid_col] = prices_df.loc[prices_df['row_id'] == i - 1, bid_col]
        prices_df.loc[prices_df['row_id'] == i, ask_col] = prices_df.loc[prices_df['row_id'] == i - 1, ask_col]

    if populate_forward_and_backward:
        missing_row_ids = prices_df.loc[(prices_df[bid_col].isnull()) & (prices_df[ask_col].isnull()), 'row_id']
        for i in missing_row_ids[::-1]:
            prices_df.loc[prices_df['row_id'] == i, bid_col] = prices_df.loc[prices_df['row_id'] == i + 1, bid_col]
            prices_df.loc[prices_df['row_id'] == i, ask_col] = prices_df.loc[prices_df['row_id'] == i + 1, ask_col]

    prices_df = prices_df.drop(columns='row_id')

    return prices_df


def convert_prices(prices):
    prices_list = []
    for i, p in enumerate(prices):
        try:
            prices_list.append(
                [
                    prices[i]['snapshotTime'],
                    prices[i]['snapshotTimeUTC'],
                    prices[i]['openPrice']['bid'],
                    prices[i]['openPrice']['ask'],
                    prices[i]['openPrice']['lastTraded'],
                    prices[i]['closePrice']['bid'],
                    prices[i]['closePrice']['ask'],
                    prices[i]['closePrice']['lastTraded'],
                    prices[i]['highPrice']['bid'],
                    prices[i]['highPrice']['ask'],
                    prices[i]['highPrice']['lastTraded'],
                    prices[i]['lowPrice']['bid'],
                    prices[i]['lowPrice']['ask'],
                    prices[i]['lowPrice']['lastTraded'],
                    prices[i]['lastTradedVolume'],

                ]
            )
        except KeyError:
            print("error getting " + str(i) + "th price")

    prices_df = pd.DataFrame(
        prices_list,
        columns=[
            'snapshotTime',
            'snapshotTimeUTC',
            'openPrice_bid',
            'openPrice_ask',
            'openPrice_last',
            'closePrice_bid',
            'closePrice_ask',
            'closePrice_last',
            'highPrice_bid',
            'highPrice_ask',
            'highPrice_lastTraded',
            'lowPrice_bid',
            'lowPrice_ask',
            'lowPrice_lastTraded',
            'lastTradedVolume'
        ]
    )

    # populate missing values
    prices_df = populate_missing_values(
        prices_df, 'openPrice_bid', 'openPrice_ask', populate_forward_and_backward=True)

    prices_df = populate_missing_values(
        prices_df, 'closePrice_bid', 'closePrice_ask', populate_forward_and_backward=True)

    prices_df = populate_missing_values(
        prices_df, 'highPrice_bid', 'highPrice_ask', populate_forward_and_backward=True)

    prices_df = populate_missing_values(
        prices_df, 'lowPrice_bid', 'lowPrice_ask', populate_forward_and_backward=True)

    prices_df['openPrice_mid'] = (prices_df['openPrice_bid'] + prices_df['openPrice_ask']) / 2
    prices_df['closePrice_mid'] = (prices_df['closePrice_bid'] + prices_df['closePrice_ask']) / 2
    prices_df['highPrice_mid'] = (prices_df['highPrice_bid'] + prices_df['highPrice_ask']) / 2
    prices_df['lowPrice_mid'] = (prices_df['lowPrice_bid'] + prices_df['lowPrice_ask']) / 2

    prices_df['snapshotTime'] = pd.to_datetime(prices_df['snapshotTime'])

    return prices_df


def place_trade(api_key, cst, x_security_token, dealReference, epic, direction, expiry,
                orderType, size, level=None, timeInForce='EXECUTE_AND_ELIMINATE', limitDistance=False, stopDistance=False,
                currencyCode='GBP', forceOpen=True, guaranteedStop=False, trailingStop=False):

    """
    orderType can be 'LIMIT', 'MARKET' or 'QUOTE'
    Need to set level if orderType = 'LIMIT'
    if current price is more favourable than level then it is a stop order, otherwise is a limit order
    expiry is the date that position automatically closes (will need to manage timeframes manually)
    timeInForce is either 'EXECUTE_AND_ELIMINATE' or 'FILL_OR_KILL'
    see more notes at https://labs.ig.com/rest-trading-api-reference/service-detail?id=542
    """

    url = "https://demo-api.ig.com/gateway/deal/positions/otc/"

    body = {
        'dealReference': dealReference,
        'epic': epic,
        'direction': direction,
        'expiry': expiry,
        'orderType': orderType,
        'size': size,
        'currencyCode': currencyCode,
        'forceOpen': forceOpen,
        'guaranteedStop': guaranteedStop,
        'trailingStop': trailingStop,
        'timeInForce': timeInForce
    }

    if limitDistance is not False:
        body['limitDistance'] = limitDistance

    if stopDistance is not False:
        body['stopDistance'] = stopDistance

    if orderType == 'LIMIT':
        body['level'] = level


    header = {
        'Content-Type': 'application/json',
        'Accept': 'application/json',
        'VERSION': '2',
        'X-IG-API-KEY': api_key,
        'CST': cst,
        'X-SECURITY-TOKEN': x_security_token
    }

    response = requests.post(url, data=json.dumps(body), headers=header)

    return response


def confirm_trade(api_key, cst, x_security_token, dealReference):

    url = "https://demo-api.ig.com/gateway/deal/confirms/" + dealReference

    header = {
        'Content-Type': 'application/json',
        'Accept': 'application/json',
        'VERSION': '1',
        'X-IG-API-KEY': api_key,
        'CST': cst,
        'X-SECURITY-TOKEN': x_security_token
    }

    response = requests.get(url, headers=header)

    return response


def close_trade(api_key, cst, x_security_token, dealId, direction, size):

    url = "https://demo-api.ig.com/gateway/deal/positions/otc/"

    header = {
        'Content-Type': 'application/json',
        'Accept': 'application/json',
        'VERSION': '1',
        'X-IG-API-KEY': api_key,
        'CST': cst,
        'X-SECURITY-TOKEN': x_security_token,
        '_method': 'DELETE'
    }

    body = {
        'dealId': dealId,
        'orderType': 'MARKET',
        'direction': direction,
        'size': size
    }

    response = requests.post(url, data=json.dumps(body), headers=header)

    return response


def generate_deal_details(confirmation_response_json):
    return pd.DataFrame({
        'date': confirmation_response_json.get('date', None),
        'status': confirmation_response_json.get('status', None),
        'reason': confirmation_response_json.get('reason', None),
        'dealStatus': confirmation_response_json.get('dealStatus', None),
        'epic': confirmation_response_json.get('epic', None),
        'expiry': confirmation_response_json.get('expiry', None),
        'dealReference': confirmation_response_json.get('dealReference', None),
        'dealId': confirmation_response_json.get('dealId', None),
        'affectedDeals': str(confirmation_response_json.get('affectedDeals', None)),
        'level': confirmation_response_json.get('level', None),
        'size': confirmation_response_json.get('size', None),
        'direction': confirmation_response_json.get('direction', None),
        'stopLevel': confirmation_response_json.get('stopLevel', None),
        'limitLevel': confirmation_response_json.get('limitLevel', None),
        'stopDistance': confirmation_response_json.get('stopDistance', None),
        'limitDistance': confirmation_response_json.get('limitDistance', None),
        'guaranteedStop': confirmation_response_json.get('guaranteedStop', None),
        'trailingStop': confirmation_response_json.get('trailingStop', None),
        'profit': confirmation_response_json.get('profit', None),
        'profitCurrency': confirmation_response_json.get('profitCurrency', None),
        'close_time': confirmation_response_json.get('close_time', None)
    }, index=[0])


def open_position_details(api_key, cst, x_security_token, dealId):

    url = "https://demo-api.ig.com/gateway/deal/positions/" + dealId

    header = {
        'Content-Type': 'application/json',
        'Accept': 'application/json',
        'VERSION': '2',
        'X-IG-API-KEY': api_key,
        'CST': cst,
        'X-SECURITY-TOKEN': x_security_token
    }

    response = requests.get(url, headers=header)

    return response


def all_open_positions_details(api_key, cst, x_security_token):

    url = "https://demo-api.ig.com/gateway/deal/positions"

    header = {
        'Content-Type': 'application/json',
        'Accept': 'application/json',
        'VERSION': '2',
        'X-IG-API-KEY': api_key,
        'CST': cst,
        'X-SECURITY-TOKEN': x_security_token
    }

    response = requests.get(url, headers=header)

    return response



