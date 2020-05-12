"""
helpers for analysing past data
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

class GenerateFeatures:

    """
    functions to apply to a series of prices to calculate features based on latest prices

    series needs to be a pandas dataframe - could run faster by avoiding pandas in future upgrade, but just mimicking
    the historic analysis for now
    """

    def __init__(self, series):
        self.series = series

    def add_previous_price_diff(self, price_col, lookback):
        current_price = self.series[price_col].iloc[-1]
        previous_price = self.series[price_col].iloc[-(lookback+1)]

        self.series.at[self.series.index[-1], 'previous_price_'+str(lookback)] = previous_price

        self.series.at[self.series.index[-1], 'previous_price_'+str(lookback)+'_diff'] = (
                current_price - previous_price)

    def add_lowest_price_in_last_x_diff(self, price_col, lookback):
        self.series.at[self.series.index[-1], 'lowest_last_'+str(lookback)] = min(
            self.series[price_col].iloc[-(1+lookback):-1])

        self.series.at[self.series.index[-1], 'lowest_last_'+str(lookback)+'_diff'] = (
                self.series[price_col].iloc[-1]
                - self.series['lowest_last_'+str(lookback)].iloc[-1]
        )

        self.series.at[self.series.index[-1], 'lowest_last_'+str(lookback)+'_timedelta'] = (
            lookback - np.argmin(list(self.series[price_col])[-(1+lookback):-1]))

    def add_highest_price_in_last_x_diff(self, price_col, lookback):
        self.series.at[self.series.index[-1], 'highest_last_'+str(lookback)] = max(
            self.series[price_col].iloc[-(1+lookback):-1])

        self.series.at[self.series.index[-1], 'highest_last_'+str(lookback)+'_diff'] = (
                self.series[price_col].iloc[-1]
                - self.series['highest_last_'+str(lookback)].iloc[-1]
        )

        self.series.at[self.series.index[-1], 'highest_last_' + str(lookback) + '_timedelta'] = (
            lookback - np.argmax(list(self.series[price_col])[-(1+lookback):-1]))

    def add_average_price_in_last_x_diff(self, price_col, lookback):
        self.series.at[self.series.index[-1], 'average_last_'+str(lookback)] = np.mean(
            self.series[price_col].iloc[-(1+lookback):-1])

        self.series.at[self.series.index[-1], 'average_last_'+str(lookback)+'_diff'] = (
                self.series[price_col].iloc[-1]
                - self.series['average_last_'+str(lookback)].iloc[-1]
        )

    def add_func_price_in_last_x_diff(self, price_col, lookback, func, name, freq=1, include_current_price=True):
        """
        freq is the number of points to pick up (1 is every point, 2 is every second and so on)
        include_current_price = True will mean that lookback + 1 points are taken into account
        """
        include_current_price = include_current_price*1  # convert to 0 or 1
        slice_end_point = len(self.series) - (1 - include_current_price)

        self.series.at[self.series.index[-1], name+'_last_'+str(lookback)] = func(
            self.series[price_col].iloc[-(1+lookback):slice_end_point:freq])

        self.series.at[self.series.index[-1], name+'_last_'+str(lookback)+'_diff'] = (
                self.series[price_col].iloc[-1]
                - self.series[name+'_last_'+str(lookback)].iloc[-1]
        )

    def add_spread(self, bid_price_col, ask_price_col):
        self.series.at[self.series.index[-1], 'spread'] = max(
            self.series[ask_price_col].iloc[-1] - self.series[bid_price_col].iloc[-1], 0)
        self.series.at[self.series.index[-1], 'spread_pc_ask'] = (
            self.series['spread'].iloc[-1]/self.series[ask_price_col].iloc[-1])

    def lin_reg_stats(self, x, y):
        x = np.array(x)
        y = np.array(y)
        n = len(x)
        if n == 1:
            return (y[0], 0, y[0])
        sum_xy = sum(x * y)
        sum_x = sum(x)
        sum_y = sum(y)
        sum_x2 = sum(x**2)

        incpt = (sum_y * sum_x2 - sum_x * sum_xy) / (n * sum_x2 - sum_x**2)

        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x**2)

        pred1 = incpt + slope * (n+1)

        return (incpt, slope, pred1)

    def regression_slope_last_x(self, price_col, lookback):
        x = range(lookback)

        reg_stats = self.lin_reg_stats(x, self.series[price_col].iloc[-(1+lookback):-1])

        self.series.at[self.series.index[-1], 'reg_incpt_last_' + str(lookback)] = reg_stats[0]
        self.series.at[self.series.index[-1], 'reg_slope_last_' + str(lookback)] = reg_stats[1]
        self.series.at[self.series.index[-1], 'reg_predt_last_' + str(lookback)] = reg_stats[2]
        self.series.at[self.series.index[-1], 'reg_predt_last_' + str(lookback)+'_diff'] = (
            reg_stats[2] - self.series[price_col].iloc[-1])

    def regression_slope_since_min_last_x(self, price_col, lookback):
        self.add_lowest_price_in_last_x_diff(price_col, lookback)

        reg_stats = self.lin_reg_stats(
            range(int(self.series['lowest_last_'+str(lookback)+'_timedelta'].iloc[-1])),
            self.series[price_col].iloc[-(1+int(self.series['lowest_last_'+str(lookback)+'_timedelta'].iloc[-1])):-1])

        self.series.at[self.series.index[-1], 'reg_incpt_since_min_last_' + str(lookback)] = reg_stats[0]
        self.series.at[self.series.index[-1], 'reg_slope_since_min_last_' + str(lookback)] = reg_stats[1]
        self.series.at[self.series.index[-1], 'reg_predt_since_min_last_' + str(lookback)] = reg_stats[2]
        self.series.at[self.series.index[-1], 'reg_predt_since_min_last_' + str(lookback)+'_diff'] = (
            reg_stats[2] - self.series[price_col].iloc[-1])

    def regression_slope_since_max_last_x(self, price_col, lookback):
        self.add_highest_price_in_last_x_diff(price_col, lookback)

        reg_stats = self.lin_reg_stats(
            range(int(self.series['highest_last_'+str(lookback)+'_timedelta'].iloc[-1])),
            self.series[price_col].iloc[-(1+int(self.series['highest_last_'+str(lookback)+'_timedelta'].iloc[-1])):-1])

        self.series.at[self.series.index[lookback:], 'reg_incpt_since_max_last_' + str(lookback)] = reg_stats[0]
        self.series.at[self.series.index[lookback:], 'reg_slope_since_max_last_' + str(lookback)] = reg_stats[1]
        self.series.at[self.series.index[lookback:], 'reg_predt_since_max_last_' + str(lookback)] = reg_stats[2]
        self.series.at[self.series.index[lookback:], 'reg_predt_since_max_last_' + str(lookback)+'_diff'] = (
            reg_stats[2] - self.series[price_col].iloc[-1])

    def rise(self, price_col):
        # Note: need to leave these acting on every row so that consecutive rises and falls can be calculated
        self.series['rise'] = None
        self.series.loc[self.series.index[1:], 'rise'] = (
                np.array(self.series[price_col].iloc[:-1]) < np.array(self.series[price_col].iloc[1:]))*1

    def fall(self, price_col):
        # Note: need to leave these acting on every row so that consecutive rises and falls can be calculated
        self.series['fall'] = None
        self.series.loc[self.series.index[1:], 'fall'] = (
                np.array(self.series[price_col].iloc[:-1]) > np.array(self.series[price_col].iloc[1:]))*1

    def consecutive_rises(self, price_col):
        # Leave as is (acting on every row) for now, can reformat if using it is too slow
        if 'rise' not in self.series.columns:
            self.rise(price_col)
        self.series['consecutive_rises'] = None
        for i in range(1, len(self.series)):
            if self.series['rise'].iat[i] == 0:
                self.series['consecutive_rises'].iat[i] = 0
            elif self.series['consecutive_rises'].iat[i-1] is None:
                pass
            else:
                self.series['consecutive_rises'].iat[i] = self.series['consecutive_rises'].iat[i-1] + 1

    def consecutive_falls(self, price_col):
        # Leave as is (acting on every row) for now, can reformat if using it is too slow
        if 'fall' not in self.series.columns:
            self.fall(price_col)
        self.series['consecutive_falls'] = None
        for i in range(1, len(self.series)):
            if self.series['fall'].iat[i] == 0:
                self.series['consecutive_falls'].iat[i] = 0
            elif self.series['consecutive_falls'].iat[i-1] is None:
                pass
            else:
                self.series['consecutive_falls'].iat[i] = self.series['consecutive_falls'].iat[i-1] + 1

    def number_times_passed_price_last_x_diff(self, price_col, lookback, price_passed_col=None, include_current=True):
        '''
        set price_passed_col to whichever column to reference to find price point passed. Leave as None for how many
        times current price has been passed.
        '''
        if price_passed_col is None:
            price_passed_col = price_col
        self.series[price_passed_col+'_passed_last_'+str(lookback)] = None

        include_current = include_current*1     # if true then will take into account lookback + 1 points but this
                                                # will ony be lookback number of price moves
        slice_end_point = len(self.series) - (1 - include_current)

        def times_passed_p(prices, p):
            # also counts if just touched and moved back up or down after (due to <= and >= for start_prices)
            prices = np.array(prices)
            start_prices = prices[:-1]
            end_prices = prices[1:]
            return sum(((start_prices <= p) & (end_prices > p)) | ((start_prices >= p) & (end_prices < p)))

        self.series.at[self.series.index[-1], price_passed_col+'_passed_last_'+str(lookback)] = times_passed_p(
            self.series[price_col].iloc[-(1+lookback):slice_end_point], self.series[price_passed_col].iloc[-1])

    def support_resistance_last_x(self, price_col, lookback, pc_within_range, pc_name_str, include_current=True):
        """
        looks for min-max range from lookback prices, finds number of times near min or max (near defined as within %
        of min-max range from min or max) and rebounds back out again
        pc_name_str is for adding to column name to make sure reliable (and flexible)
        if include_current = True it still doesn't take it into account for the period max and mins, only means latest
                potential 'rebound' to current price taken into account, this allows to be able to check if current
                price has broken out of recent support or resistance
        """
        self.add_lowest_price_in_last_x_diff(price_col, lookback)
        self.add_highest_price_in_last_x_diff(price_col, lookback)

        include_current = include_current * 1   # if true then will take into account lookback + 1 points but this
                                                # will ony be lookback number of price moves
                                                # and note lowest/highest last does not include current price
        slice_end_point = len(self.series) - (1 + include_current)

        self.series.at[self.series.index[-1], 'min_max_range_last_'+str(lookback)] = (
            self.series['highest_last_'+str(lookback)].iloc[-1] - self.series['lowest_last_'+str(lookback)].iloc[-1])

        def support_rebounds_from_low(prices, low, range, pc_within_range):
            '''
            only count rebound if comes back pc_within_range from current low, otherwise ends up counting loads when
            oscillating around the low_range price-point
            '''
            prices = np.array(prices)
            low_range = low + range*pc_within_range
            current_low = max(prices)
            rebounds = 0
            for p in prices:
                if p <= low_range:
                    current_low = min(p, current_low)
                if p > current_low + range*pc_within_range:
                    rebounds += 1
                    current_low = max(prices)

            return rebounds

        def resistance_rebounds_from_high(prices, high, range, pc_within_range):
            '''
            only count rebound if comes back pc_within_range from current low, otherwise ends up counting loads when
            oscillating around the low_range price-point
            '''
            prices = np.array(prices)
            high_range = high - range * pc_within_range
            current_high = min(prices)
            rebounds = 0
            for p in prices:
                if p >= high_range:
                    current_high = max(p, current_high)
                if p < current_high - range * pc_within_range:
                    rebounds += 1
                    current_high = min(prices)

            return rebounds

        self.series.at[
            self.series.index[-1], 'support_rebounds_'+str(pc_name_str)+'_last_'+str(lookback)] = (
            support_rebounds_from_low(self.series[price_col].iloc[-(1+lookback):slice_end_point],
                                      self.series['lowest_last_'+str(lookback)].iloc[-1],
                                      self.series['min_max_range_last_'+str(lookback)].iloc[-1],
                                      pc_within_range)
        )

        self.series.at[
            self.series.index[-1], 'resistance_rebounds_'+str(pc_name_str)+'_last_'+str(lookback)] = (
            resistance_rebounds_from_high(self.series[price_col].iloc[-(1+lookback):slice_end_point],
                                          self.series['highest_last_'+str(lookback)].iloc[-1],
                                          self.series['min_max_range_last_'+str(lookback)].iloc[-1],
                                          pc_within_range)
        )

        # add if resistance/support hit 2 or more and 3 or more times
        number_resistance_rebounds = self.series[
            'resistance_rebounds_' + str(pc_name_str) + '_last_' + str(lookback)].iloc[-1]
        number_support_rebounds = self.series[
            'support_rebounds_' + str(pc_name_str) + '_last_' + str(lookback)].iloc[-1]

        # resistance rebounds 2 and 3
        if number_resistance_rebounds >= 2:
            self.series.at[
                self.series.index[-1], 'resistance_2_' + str(pc_name_str) + '_last_' + str(lookback)] = (
                self.series['highest_last_' + str(lookback)].iloc[-1])
        else:
            self.series.at[
                self.series.index[-1], 'resistance_2_' + str(pc_name_str) + '_last_' + str(lookback)] = np.nan

        self.series[self.series.index[-1], 'resistance_2_' + str(pc_name_str) + '_last_' + str(lookback) + '_diff'] = (
                self.series[price_col].iloc[-1] -
                self.series['resistance_2_'+str(pc_name_str)+'_last_'+str(lookback)+'_diff'].iloc[-1]
        )

        if number_resistance_rebounds >= 3:
            self.series.at[
                self.series.index[-1], 'resistance_3_' + str(pc_name_str) + '_last_' + str(lookback)] = (
                self.series['highest_last_' + str(lookback)].iloc[-1])
        else:
            self.series.at[
                self.series.index[-1], 'resistance_3_' + str(pc_name_str) + '_last_' + str(lookback)] = np.nan

        self.series[self.series.index[-1], 'resistance_3_' + str(pc_name_str) + '_last_' + str(lookback) + '_diff'] = (
                self.series[price_col].iloc[-1] -
                self.series['resistance_3_'+str(pc_name_str)+'_last_'+str(lookback)+'_diff'].iloc[-1]
        )

        # support rebounds 2 and 3
        if number_support_rebounds >= 2:
            self.series.at[
                self.series.index[-1], 'support_2_' + str(pc_name_str) + '_last_' + str(lookback)] = (
                self.series['lowest_last_' + str(lookback)].iloc[-1])
        else:
            self.series.at[
                self.series.index[-1], 'support_2_' + str(pc_name_str) + '_last_' + str(lookback)] = np.nan

        self.series[self.series.index[-1], 'support_2_' + str(pc_name_str) + '_last_' + str(lookback) + '_diff'] = (
                self.series[price_col].iloc[-1] -
                self.series['support_2_' + str(pc_name_str) + '_last_' + str(lookback) + '_diff'].iloc[-1]
        )

        if number_support_rebounds >= 3:
            self.series.at[
                self.series.index[-1], 'support_3_' + str(pc_name_str) + '_last_' + str(lookback)] = (
                self.series['lowest_last_' + str(lookback)].iloc[-1])
        else:
            self.series.at[
                self.series.index[-1], 'support_3_' + str(pc_name_str) + '_last_' + str(lookback)] = np.nan

        self.series[self.series.index[-1], 'support_3_' + str(pc_name_str) + '_last_' + str(lookback) + '_diff'] = (
                self.series[price_col].iloc[-1] -
                self.series['support_3_' + str(pc_name_str) + '_last_' + str(lookback) + '_diff'].iloc[-1]
        )

    def add_average_price_in_last_x_diff_all_rows(self, price_col, lookback):
        self.series['average_last_'+str(lookback)] = None
        self.series['average_last_'+str(lookback)+'_diff'] = None

        self.series.loc[self.series.index[lookback:], 'average_last_'+str(lookback)] = [
            np.mean(self.series[price_col].iloc[r-lookback:r]) for r in range(lookback, len(self.series))]

        self.series.loc[self.series.index[lookback:], 'average_last_'+str(lookback)+'_diff'] = (
                self.series[price_col].iloc[lookback:]
                - self.series['average_last_'+str(lookback)].iloc[lookback:]
        )

    def moving_avg_short_vs_long(self, price_col, lookback_short, lookback_long):
        """
        NOTE: have to do for all rows because uses created values from past times
        Could speed up later section by just applying to last row, but only do this if makes a big differnce
        adds
        - difference (short - long) moving average
        - time this side
        - average difference
        - median difference
        """
        self.add_average_price_in_last_x_diff_all_rows(price_col, lookback_short)
        self.add_average_price_in_last_x_diff_all_rows(price_col, lookback_long)

        self.series['average_last_'+str(lookback_short)+'_'+str(lookback_long)+'_diff'] = None
        self.series['average_last_'+str(lookback_short)+'_'+str(lookback_long)+'_period_this_side'] = None
        self.series['average_last_'+str(lookback_short)+'_'+str(lookback_long)+'_period_this_side_avg_diff'] = None
        self.series['average_last_'+str(lookback_short)+'_'+str(lookback_long)+'_period_this_side_median_diff'] = None

        self.series.loc[self.series.index[lookback_long:],
                        'average_last_'+str(lookback_short)+'_'+str(lookback_long)+'_diff'] = [
            (self.series['average_last_'+str(lookback_short)].iloc[r]
             - self.series['average_last_'+str(lookback_long)].iloc[r])
            for r in range(lookback_long, len(self.series))]

        # add count of how long short term avg is less than or greater than long term avg
        # and median and avg diff over each period
        previous_diffs = self.series['average_last_'+str(lookback_short)+'_'+str(lookback_long)+'_diff'].iloc[
            lookback_long:-1].values
        current_diffs = self.series['average_last_'+str(lookback_short)+'_'+str(lookback_long)+'_diff'].iloc[
            lookback_long+1:].values
        period_this_side_str = 'average_last_' + str(lookback_short) + '_' + str(lookback_long) + '_period_this_side'
        for i in range(len(current_diffs)):
            if current_diffs[i] == 0:
                self.series[period_this_side_str].iloc[lookback_long + i + 1] = 0
            elif (current_diffs[i] > 0) & (previous_diffs[i] <= 0):
                self.series[period_this_side_str].iloc[lookback_long + i + 1] = 1
            elif (current_diffs[i] < 0) & (previous_diffs[i] >= 0):
                self.series[period_this_side_str].iloc[lookback_long + i + 1] = 1
            elif self.series[period_this_side_str].iloc[lookback_long + i] is None:
                pass
            else:
                self.series[period_this_side_str].iloc[lookback_long + i + 1] = (
                        self.series[period_this_side_str].iloc[lookback_long + i] + 1)

            # add avg and median for period above below
            if self.series[period_this_side_str].iloc[lookback_long + i + 1] == 0:
                self.series[
                    'average_last_'+str(lookback_short)+'_'+str(lookback_long)+'_period_this_side_avg_diff'].iloc[
                    lookback_long + i + 1] = 0
                self.series[
                    'average_last_'+str(lookback_short)+'_'+str(lookback_long)+'_period_this_side_median_diff'].iloc[
                    lookback_long + i + 1] = 0
            elif self.series[period_this_side_str].iloc[lookback_long + i + 1] is None:
                pass
            else:
                # *Note: need to shift index lookup along 1 because want to look up to and including current position
                # hence the -1 and extra +1 at highlighted places
                period_this_side_lookback = self.series[period_this_side_str].iloc[lookback_long + i + 1] - 1  # *
                self.series[
                    'average_last_'+str(lookback_short)+'_'+str(lookback_long)+'_period_this_side_avg_diff'].iloc[
                    lookback_long + i + 1] = np.mean(
                    self.series['average_last_'+str(lookback_short)+'_'+str(lookback_long)+'_diff'].iloc[
                        lookback_long + i + 1 - period_this_side_lookback:lookback_long + i + 1 + 1])  # *

                self.series[
                    'average_last_'+str(lookback_short)+'_'+str(lookback_long)+'_period_this_side_median_diff'].iloc[
                    lookback_long + i + 1] = np.median(
                    self.series['average_last_'+str(lookback_short)+'_'+str(lookback_long)+'_diff'].iloc[
                        lookback_long + i + 1 - period_this_side_lookback:lookback_long + i + 1 + 1])  # *







