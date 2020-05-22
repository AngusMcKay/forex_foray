"""
helpers for analysing past data
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

class GenerateHistoricFeatures:

    """
    functions to apply to a series of prices to calculate features ('signals')
    at each time based on current and past prices

    also contains functions for calculating future outcomes

    features can then be used for back-testing trading algorithms

    series needs to be a pandas dataframe - could run faster by avoiding pandas, but fine for back testing
    """

    def __init__(self, series):
        self.series = series

    def add_previous_price_diff(self, price_col, lookback):
        self.series['previous_price_'+str(lookback)] = None
        self.series['previous_price_'+str(lookback)+'_diff'] = None

        current_price = self.series.loc[self.series.index[lookback:], price_col].values
        previous_price = self.series.loc[self.series.index[:-lookback], price_col].values

        self.series.loc[self.series.index[lookback:], 'previous_price_'+str(lookback)] = previous_price

        self.series.loc[self.series.index[lookback:], 'previous_price_'+str(lookback)+'_diff'] = (
                current_price - previous_price)

    def add_lowest_price_in_last_x_diff(self, price_col, lookback, include_time_since_low=True):
        """
        Note: including time since low slows down function a lot, but is needed for calculating regression slope since
        min
        """
        self.series['lowest_last_'+str(lookback)] = None
        self.series['lowest_last_'+str(lookback)+'_diff'] = None

        self.series.loc[self.series.index[lookback:], 'lowest_last_'+str(lookback)] = [
            min(self.series[price_col].iloc[r-lookback:r]) for r in range(lookback, len(self.series))]

        self.series.loc[self.series.index[lookback:], 'lowest_last_'+str(lookback)+'_diff'] = (
                self.series[price_col].iloc[lookback:]
                - self.series['lowest_last_'+str(lookback)].iloc[lookback:]
        )

        if include_time_since_low:
            self.series.loc[self.series.index[lookback:], 'lowest_last_'+str(lookback)+'_timedelta'] = [
                lookback - np.argmin(list(self.series[price_col])[r-lookback:r])
                for r in range(lookback, len(self.series))]

    def add_highest_price_in_last_x_diff(self, price_col, lookback, include_time_since_high=True):
        """
        Note: including time since high slows down function a lot, but is needed for calculating regression slope since
        max
        """
        self.series['highest_last_'+str(lookback)] = None
        self.series['highest_last_'+str(lookback)+'_diff'] = None

        self.series.loc[self.series.index[lookback:], 'highest_last_'+str(lookback)] = [
            max(self.series[price_col].iloc[r-lookback:r]) for r in range(lookback, len(self.series))]

        self.series.loc[self.series.index[lookback:], 'highest_last_'+str(lookback)+'_diff'] = (
                self.series[price_col].iloc[lookback:]
                - self.series['highest_last_'+str(lookback)].iloc[lookback:]
        )

        if include_time_since_high:
            self.series.loc[self.series.index[lookback:], 'highest_last_' + str(lookback) + '_timedelta'] = [
                lookback - np.argmax(list(self.series[price_col])[r-lookback:r])
                for r in range(lookback, len(self.series))]

    def add_average_price_in_last_x_diff(self, price_col, lookback):
        self.series['average_last_'+str(lookback)] = None
        self.series['average_last_'+str(lookback)+'_diff'] = None

        self.series.loc[self.series.index[lookback:], 'average_last_'+str(lookback)] = [
            np.mean(self.series[price_col].iloc[r-lookback:r]) for r in range(lookback, len(self.series))]

        self.series.loc[self.series.index[lookback:], 'average_last_'+str(lookback)+'_diff'] = (
                self.series[price_col].iloc[lookback:]
                - self.series['average_last_'+str(lookback)].iloc[lookback:]
        )

    def add_func_price_in_last_x_diff(self, price_col, lookback, func, name, freq=1, include_current_price=True):
        """
        freq is the number of points to pick up (1 is every point, 2 is every second and so on)
        include_current_price = True will mean that lookback + 1 points are taken into account
        """
        include_current_price = include_current_price*1  # convert to 0 or 1

        self.series[name+'_last_'+str(lookback)] = None
        self.series[name+'_last_'+str(lookback)+'_diff'] = None

        self.series.loc[self.series.index[lookback:], name+'_last_'+str(lookback)] = [
            func(self.series[price_col].iloc[r-lookback:r+include_current_price:freq])
            for r in range(lookback, len(self.series))]

        self.series.loc[self.series.index[lookback:], name+'_last_'+str(lookback)+'_diff'] = (
                self.series[price_col].iloc[lookback:]
                - self.series[name+'_last_'+str(lookback)].iloc[lookback:]
        )

    def add_func_col_in_last_x(self, col, lookback, func, name, freq=1, include_current_row=True):
        """
        simpler version of above that doesn't add difference, intended for use with non-price based columns
        """
        include_current_row = include_current_row*1  # convert to 0 or 1

        self.series[name+'_last_'+str(lookback)] = None

        self.series.loc[self.series.index[lookback:], name+'_last_'+str(lookback)] = [
            func(self.series[col].iloc[r-lookback:r+include_current_row:freq])
            for r in range(lookback, len(self.series))]

    def add_spread(self, bid_price_col, ask_price_col):
        self.series['spread'] = np.maximum(self.series[ask_price_col] - self.series[bid_price_col], 0)
        self.series['spread_pc_ask'] = self.series['spread']/self.series[ask_price_col]

    def lin_reg_stats(self, x, y):
        try:
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

        except TypeError:
            print('warning: unable to calculate regression for row, returning np.nans')
            return (np.nan, np.nan, np.nan)

    def regression_slope_last_x(self, price_col, lookback):
        x = range(lookback)

        reg_stats = np.array([
            self.lin_reg_stats(x, self.series[price_col].iloc[r-lookback:r]) for r in range(lookback, len(self.series))]
        )

        self.series.loc[self.series.index[lookback:], 'reg_incpt_last_' + str(lookback)] = reg_stats[:, 0]
        self.series.loc[self.series.index[lookback:], 'reg_slope_last_' + str(lookback)] = reg_stats[:, 1]
        self.series.loc[self.series.index[lookback:], 'reg_predt_last_' + str(lookback)] = reg_stats[:, 2]
        self.series.loc[self.series.index[lookback:], 'reg_predt_last_' + str(lookback)+'_diff'] = (
            reg_stats[:, 2] - self.series[price_col].iloc[lookback:].values)

    def regression_slope_last_x_nameable(self, col, lookback, name):
        x = range(lookback)

        reg_stats = np.array([
            self.lin_reg_stats(x, self.series[col].iloc[r-lookback:r]) for r in range(lookback, len(self.series))]
        )

        self.series.loc[self.series.index[lookback:], name+'_reg_incpt_last_' + str(lookback)] = reg_stats[:, 0]
        self.series.loc[self.series.index[lookback:], name+'_reg_slope_last_' + str(lookback)] = reg_stats[:, 1]
        self.series.loc[self.series.index[lookback:], name+'_reg_predt_last_' + str(lookback)] = reg_stats[:, 2]
        self.series.loc[self.series.index[lookback:], name+'_reg_predt_last_' + str(lookback)+'_diff'] = (
            reg_stats[:, 2] - self.series[col].iloc[lookback:].values)

    def regression_slope_since_min_last_x(self, price_col, lookback):
        if 'lowest_last_'+str(lookback)+'_timedelta' not in self.series.columns:
            self.add_lowest_price_in_last_x_diff(price_col, lookback)

        reg_stats = np.array([
            self.lin_reg_stats(range(int(self.series['lowest_last_'+str(lookback)+'_timedelta'].iloc[r])),
                               self.series[price_col].iloc[
                               r-int(self.series['lowest_last_'+str(lookback)+'_timedelta'].iloc[r]):r])
            for r in range(lookback, len(self.series))]
        )

        self.series.loc[self.series.index[lookback:], 'reg_incpt_since_min_last_' + str(lookback)] = reg_stats[:, 0]
        self.series.loc[self.series.index[lookback:], 'reg_slope_since_min_last_' + str(lookback)] = reg_stats[:, 1]
        self.series.loc[self.series.index[lookback:], 'reg_predt_since_min_last_' + str(lookback)] = reg_stats[:, 2]
        self.series.loc[self.series.index[lookback:], 'reg_predt_since_min_last_' + str(lookback)+'_diff'] = (
            reg_stats[:, 2] - self.series[price_col].iloc[lookback:].values)

    def regression_slope_since_max_last_x(self, price_col, lookback):
        if 'highest_last_'+str(lookback)+'_timedelta' not in self.series.columns:
            self.add_highest_price_in_last_x_diff(price_col, lookback)

        reg_stats = np.array([
            self.lin_reg_stats(range(int(self.series['highest_last_'+str(lookback)+'_timedelta'].iloc[r])),
                               self.series[price_col].iloc[
                               r-int(self.series['highest_last_'+str(lookback)+'_timedelta'].iloc[r]):r])
            for r in range(lookback, len(self.series))]
         )

        self.series.loc[self.series.index[lookback:], 'reg_incpt_since_max_last_' + str(lookback)] = reg_stats[:, 0]
        self.series.loc[self.series.index[lookback:], 'reg_slope_since_max_last_' + str(lookback)] = reg_stats[:, 1]
        self.series.loc[self.series.index[lookback:], 'reg_predt_since_max_last_' + str(lookback)] = reg_stats[:, 2]
        self.series.loc[self.series.index[lookback:], 'reg_predt_since_max_last_' + str(lookback)+'_diff'] = (
            reg_stats[:, 2] - self.series[price_col].iloc[lookback:].values)

    def quantile_regression_slopes_and_wedge(self, price_col, lookback, q, name):
        '''
        can't find a better way than to do this using statsmodels, which needs a pandas dataframe as input :(
        takes ages, need a better solution if wanting to use this
        '''
        q = min(q, 1-q)
        p = 1-q
        x = range(lookback)
        def quant_reg_coeff(x, y, qu):
            df = pd.DataFrame({'x': x, 'y': y})
            qrmod = smf.quantreg('y ~ x', df).fit(qu)
            return qrmod.params['x']

        self.series['quant_reg_'+str(name)+'_last_'+str(lookback)+'_lower'] = None
        self.series.loc[self.series.index[lookback:], 'quant_reg_'+str(name)+'_last_'+str(lookback)+'_lower'] = [
            quant_reg_coeff(x, self.series[price_col].iloc[r-lookback:r], q) for r in range(lookback, len(self.series))]

        self.series['quant_reg_' + str(name) + '_last_' + str(lookback) + '_upper'] = None
        self.series.loc[self.series.index[lookback:], 'quant_reg_'+str(name)+'_last_'+str(lookback)+'_upper'] = [
            quant_reg_coeff(x, self.series[price_col].iloc[r - lookback:r], p) for r in
            range(lookback, len(self.series))]

        self.series['quant_reg_' + str(name) + '_last_' + str(lookback) + '_wedge'] = (
            self.series['quant_reg_'+str(name)+'_last_'+str(lookback)+'_upper']
            - self.series['quant_reg_'+str(name)+'_last_'+str(lookback)+'_lower']
        )

    def rise(self, price_col):
        self.series['rise'] = None
        self.series.loc[self.series.index[1:], 'rise'] = (
                np.array(self.series[price_col].iloc[:-1]) < np.array(self.series[price_col].iloc[1:]))*1

    def fall(self, price_col):
        self.series['fall'] = None
        self.series.loc[self.series.index[1:], 'fall'] = (
                np.array(self.series[price_col].iloc[:-1]) > np.array(self.series[price_col].iloc[1:]))*1

    def consecutive_rises(self, price_col):
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

        def times_passed_p(prices, p):
            # also counts if just touched and moved back up or down after (due to <= and >= for start_prices)
            prices = np.array(prices)
            start_prices = prices[:-1]
            end_prices = prices[1:]
            return sum(((start_prices <= p) & (end_prices > p)) | ((start_prices >= p) & (end_prices < p)))

        self.series.loc[self.series.index[lookback:], price_passed_col+'_passed_last_'+str(lookback)] = [
            times_passed_p(self.series[price_col].iloc[r-lookback:r+include_current],
                           self.series[price_passed_col].iloc[r])
            for r in range(lookback, len(self.series))
        ]

    def support_resistance_last_x(self, price_col, lookback, pc_within_range, pc_name_str, include_current=True):
        """
        looks for min-max range from lookback prices, finds number of times near min or max (near defined as within %
        of min-max range from min or max) and rebounds back out again
        pc_name_str is for adding to column name to make sure reliable (and flexible)
        if include_current = True it still doesn't take it into account for the period max and mins, only means latest
                potential 'rebound' to current price taken into account, this allows to be able to check if current
                price has broken out of recent support or resistance
        """
        if 'lowest_last_'+str(lookback) not in self.series.columns:
            self.add_lowest_price_in_last_x_diff(price_col, lookback)
        if 'highest_last_'+str(lookback) not in self.series.columns:
            self.add_highest_price_in_last_x_diff(price_col, lookback)

        include_current = include_current * 1   # if true then will take into account lookback + 1 points but this
                                                # will ony be lookback number of price moves
                                                # and note lowest/highest last does not include current price

        self.series['min_max_range_last_'+str(lookback)] = (
            self.series['highest_last_'+str(lookback)] - self.series['lowest_last_'+str(lookback)])

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

            # prices = np.array(prices)
            # start_prices = prices[:-1]
            # end_prices = prices[1:]
            # support_range_max = low + range*pc_within_range
            # return sum((start_prices <= support_range_max) & (end_prices > support_range_max))

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

            # prices = np.array(prices)
            # start_prices = prices[:-1]
            # end_prices = prices[1:]
            # support_range_max = high - range*pc_within_range
            # return sum((start_prices >= support_range_max) & (end_prices < support_range_max))

        self.series.loc[
            self.series.index[lookback:], 'support_rebounds_'+str(pc_name_str)+'_last_'+str(lookback)] = [
            support_rebounds_from_low(self.series[price_col].iloc[r-lookback:r+include_current],
                                      self.series['lowest_last_'+str(lookback)].iloc[r],
                                      self.series['min_max_range_last_'+str(lookback)].iloc[r],
                                      pc_within_range)
            for r in range(lookback, len(self.series))
        ]

        self.series.loc[
            self.series.index[lookback:], 'resistance_rebounds_'+str(pc_name_str)+'_last_'+str(lookback)] = [
            resistance_rebounds_from_high(self.series[price_col].iloc[r-lookback:r+include_current],
                                          self.series['highest_last_'+str(lookback)].iloc[r],
                                          self.series['min_max_range_last_'+str(lookback)].iloc[r],
                                          pc_within_range)
            for r in range(lookback, len(self.series))
        ]

        # add if resistance/support hit 2 or more and 3 or more times
        self.series['resistance_2_'+str(pc_name_str)+'_last_'+str(lookback)] = np.nan
        self.series['resistance_2_'+str(pc_name_str)+'_last_'+str(lookback)+'_diff'] = np.nan
        self.series['resistance_3_'+str(pc_name_str)+'_last_'+str(lookback)] = np.nan
        self.series['resistance_3_'+str(pc_name_str)+'_last_'+str(lookback)+'_diff'] = np.nan

        self.series.loc[
            self.series['resistance_rebounds_'+str(pc_name_str)+'_last_'+str(lookback)] >= 2,
            'resistance_2_'+str(pc_name_str)+'_last_'+str(lookback)] = self.series.loc[
            self.series['resistance_rebounds_' + str(pc_name_str) + '_last_' + str(lookback)] >= 2,
            'highest_last_'+str(lookback)
        ]
        self.series['resistance_2_'+str(pc_name_str)+'_last_'+str(lookback)+'_diff'] = (
                self.series[price_col] - self.series['resistance_2_'+str(pc_name_str)+'_last_'+str(lookback)+'_diff']
        )

        self.series.loc[
            self.series['resistance_rebounds_' + str(pc_name_str) + '_last_' + str(lookback)] >= 3,
            'resistance_3_' + str(pc_name_str) + '_last_' + str(lookback)] = self.series.loc[
            self.series['resistance_rebounds_' + str(pc_name_str) + '_last_' + str(lookback)] >= 3,
            'highest_last_' + str(lookback)
        ]
        self.series['resistance_3_' + str(pc_name_str) + '_last_' + str(lookback) + '_diff'] = (
                self.series[price_col] - self.series['resistance_3_'+str(pc_name_str)+'_last_'+str(lookback)+'_diff']
        )

        self.series['support_2_' + str(pc_name_str) + '_last_' + str(lookback)] = np.nan
        self.series['support_2_' + str(pc_name_str) + '_last_' + str(lookback) + '_diff'] = np.nan
        self.series['support_3_' + str(pc_name_str) + '_last_' + str(lookback)] = np.nan
        self.series['support_3_' + str(pc_name_str) + '_last_' + str(lookback) + '_diff'] = np.nan

        self.series.loc[
            self.series['support_rebounds_' + str(pc_name_str) + '_last_' + str(lookback)] >= 2,
            'support_2_' + str(pc_name_str) + '_last_' + str(lookback)] = self.series.loc[
            self.series['support_rebounds_' + str(pc_name_str) + '_last_' + str(lookback)] >= 2,
            'lowest_last_' + str(lookback)
        ]
        self.series['support_2_' + str(pc_name_str) + '_last_' + str(lookback) + '_diff'] = (
                self.series[price_col] - self.series['support_2_'+str(pc_name_str)+'_last_'+str(lookback)+'_diff']
        )

        self.series.loc[
            self.series['support_rebounds_' + str(pc_name_str) + '_last_' + str(lookback)] >= 3,
            'support_3_' + str(pc_name_str) + '_last_' + str(lookback)] = self.series.loc[
            self.series['support_rebounds_' + str(pc_name_str) + '_last_' + str(lookback)] >= 3,
            'lowest_last_' + str(lookback)
        ]
        self.series['support_3_' + str(pc_name_str) + '_last_' + str(lookback) + '_diff'] = (
                self.series[price_col] - self.series['support_3_'+str(pc_name_str)+'_last_'+str(lookback)+'_diff']
        )

    def moving_avg_short_vs_long(self, price_col, lookback_short, lookback_long):
        """
        adds
        - difference (short - long) moving average
        - time this side
        - average difference
        - median difference
        """
        if 'average_last_'+str(lookback_short) not in self.series.columns:
            self.add_average_price_in_last_x_diff(price_col, lookback_short)
        if 'average_last_'+str(lookback_long) not in self.series.columns:
            self.add_average_price_in_last_x_diff(price_col, lookback_long)

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

    '''
    add outcomes
    '''

    def add_future_price_diff(self, start_price_col, end_price_col, lookforward, name):
        self.series[name+'_future_price_' + str(lookforward)] = None
        self.series[name+'_future_price_' + str(lookforward) + '_diff'] = None

        current_price = self.series.loc[self.series.index[:-lookforward], start_price_col].values
        future_price = self.series.loc[self.series.index[lookforward:], end_price_col].values

        self.series.loc[self.series.index[:-lookforward], name+'_future_price_' + str(lookforward)] = future_price

        self.series.loc[self.series.index[:-lookforward], name+'_future_price_' + str(lookforward) + '_diff'] = (
                future_price - current_price)

    def add_lowest_price_in_next_x_diff(self, start_price_col, end_price_high_col, end_price_low_col, lookforward):
        self.series['lowest_next_'+str(lookforward)] = None
        self.series['lowest_next_'+str(lookforward)+'_diff'] = None

        self.series.loc[self.series.index[:-lookforward], 'lowest_next_'+str(lookforward)] = [
            min(self.series[end_price_low_col].iloc[r+1:r+lookforward+1]) for r in range(len(self.series)-lookforward)]

        self.series.loc[self.series.index[:-lookforward], 'lowest_next_'+str(lookforward)+'_diff'] = (
                self.series['lowest_next_'+str(lookforward)].iloc[:-lookforward]
                - self.series[start_price_col].iloc[:-lookforward]
        )

        self.series.loc[self.series.index[:-lookforward], 'lowest_next_'+str(lookforward)+'_timedelta'] = [
            np.argmin(list(self.series[end_price_low_col])[r+1:r+lookforward+1]) + 1
            for r in range(len(self.series)-lookforward)]

        # note: prev high calc assumes that if in same period as low point then high occurs before the low
        self.series.loc[self.series.index[:-lookforward], 'lowest_next_' + str(lookforward) + '_prev_high'] = [
            max(self.series[end_price_high_col][
                r+1:r+int(self.series['lowest_next_'+str(lookforward)+'_timedelta'].iloc[r])+1])
            for r in range(len(self.series) - lookforward)]

        self.series.loc[self.series.index[:-lookforward], 'lowest_next_' + str(lookforward) + '_prev_high_diff'] = (
                self.series['lowest_next_' + str(lookforward) + '_prev_high'].iloc[:-lookforward]
                - self.series[start_price_col].iloc[:-lookforward]
        )

    def add_highest_price_in_next_x_diff(self, start_price_col, end_price_high_col, end_price_low_col, lookforward):
        self.series['highest_next_'+str(lookforward)] = None
        self.series['highest_next_'+str(lookforward)+'_diff'] = None

        self.series.loc[self.series.index[:-lookforward], 'highest_next_'+str(lookforward)] = [
            max(self.series[end_price_high_col].iloc[r+1:r+lookforward+1]) for r in range(len(self.series)-lookforward)]

        self.series.loc[self.series.index[:-lookforward], 'highest_next_'+str(lookforward)+'_diff'] = (
                self.series['highest_next_'+str(lookforward)].iloc[:-lookforward]
                - self.series[start_price_col].iloc[:-lookforward]
        )

        self.series.loc[self.series.index[:-lookforward], 'highest_next_' + str(lookforward) + '_timedelta'] = [
            np.argmax(list(self.series[end_price_high_col])[r+1:r+lookforward+1]) + 1
            for r in range(len(self.series)-lookforward)]

        # note: prev low calc assumes that if in same period as high point then low occurs before the high
        self.series.loc[self.series.index[:-lookforward], 'highest_next_' + str(lookforward) + '_prev_low'] = [
            min(self.series[end_price_low_col][
                r+1:r+int(self.series['highest_next_'+str(lookforward)+'_timedelta'].iloc[r])+1])
            for r in range(len(self.series) - lookforward)]

        self.series.loc[self.series.index[:-lookforward], 'highest_next_' + str(lookforward) + '_prev_low_diff'] = (
                self.series['highest_next_' + str(lookforward) + '_prev_low'].iloc[:-lookforward]
                - self.series[start_price_col].iloc[:-lookforward]
        )

    def add_func_price_in_next_x_diff(self, price_col, lookforward, func, name, freq=1):
        """
        freq is the number of points to pick up (1 is every point, 2 is every second and so on)
        """
        self.series[name+'_next_'+str(lookforward)] = None
        self.series[name+'_next_'+str(lookforward)+'_diff'] = None

        self.series.loc[self.series.index[:-lookforward], name+'_next_'+str(lookforward)] = [
            func(self.series[price_col].iloc[r+1:r+lookforward+1:freq])
            for r in range(len(self.series)-lookforward)]

        self.series.loc[self.series.index[:-lookforward], name+'_next_'+str(lookforward)+'_diff'] = (
                self.series[name+'_next_'+str(lookforward)].iloc[:-lookforward]
                - self.series[price_col].iloc[:-lookforward]
        )

    def add_price_after_breach_price_change(self, price_col, lookforward, price_change):
        self.series['price_after_breach_'+str(price_change)+'_next_'+str(lookforward)] = None
        self.series['price_after_breach_'+str(price_change)+'_next_'+str(lookforward)+'_diff'] = None
        self.series['price_after_breach_'+str(price_change)+'_next_'+str(lookforward)+'_timedelta'] = None

        def first_over_change(prices, current_price, price_change):
            """
            returns index and price in tuple, so make sure to only pick up whichever is required
            if no reaches then return (np.nan, np.nan)
            """
            limit = current_price + price_change
            if price_change > 0:
                breaches = [(i+1, p) for i, p in enumerate(prices) if p >= limit]
            else:
                breaches = [(i+1, p) for i, p in enumerate(prices) if p <= limit]

            if len(breaches) == 0:
                return (np.nan, np.nan)

            return breaches[0]

        self.series.loc[self.series.index[:-lookforward],
                        'price_after_breach_'+str(price_change)+'_next_'+str(lookforward)] = [
            first_over_change(self.series[price_col].iloc[r+1:r+lookforward+1],
                              self.series[price_col].iloc[r],
                              price_change)[1]
            for r in range(len(self.series) - lookforward)]

        self.series.loc[self.series.index[:-lookforward],
                        'price_after_breach_'+str(price_change)+'_next_'+str(lookforward)+'_diff'] = (
            self.series['price_after_breach_'+str(price_change)+'_next_'+str(lookforward)].iloc[:-lookforward]
            - self.series[price_col].iloc[:-lookforward]
        )

        self.series.loc[self.series.index[:-lookforward],
                        'price_after_breach_'+str(price_change)+'_next_'+str(lookforward)+'_timedelta'] = [
            first_over_change(self.series[price_col].iloc[r + 1:r + lookforward + 1],
                              self.series[price_col].iloc[r],
                              price_change)[0]
            for r in range(len(self.series) - lookforward)]

    def stoploss_limit_outcome(self, start_price_col, end_price_high_col, end_price_low_col, end_price_close_col,
                               lookforward, high_limit, low_limit, long_or_short, name):
        """
        start price should be ask for long position, bid for short position
        end prices should be bid for long position, ask for short position
        if breach high and low limits in same period then assumes worst case scenario
        high_limit and low_limit expressed as difference between current price
        """

        def first_breach_low(low_prices, start_price):
            breaches = [i + 1 for i, p in enumerate(low_prices) if p <= start_price + low_limit]
            if len(breaches) == 0:
                return False
            return breaches[0]

        def first_breach_high(high_prices, start_price):
            breaches = [i + 1 for i, p in enumerate(high_prices) if p >= start_price + high_limit]
            if len(breaches) == 0:
                return False
            return breaches[0]

        def calculate_return(low_prices, high_prices, start_price, no_breach_end_price):
            low_breach = first_breach_low(low_prices, start_price)
            high_breach = first_breach_high(high_prices, start_price)

            if not low_breach and not high_breach:
                return no_breach_end_price

            if long_or_short == 'long':
                if not low_breach:
                    return start_price + high_limit
                elif not high_breach:
                    return start_price + low_limit
                elif low_breach <= high_breach:
                    return start_price + low_limit
                elif high_breach < low_breach:
                    return start_price + high_limit

            if long_or_short == 'short':
                if not high_breach:
                    return start_price + low_limit
                elif not low_breach:
                    return start_price + high_limit
                elif high_breach <= low_breach:
                    return start_price + high_limit
                elif low_breach < high_breach:
                    return start_price + low_limit

        self.series.loc[self.series.index[:-lookforward],
                        long_or_short + '_' + name + '_next_' + str(lookforward)+'_end_price'] = [
            calculate_return(
                self.series[end_price_low_col].iloc[r + 1:r + lookforward + 1],
                self.series[end_price_high_col].iloc[r + 1:r + lookforward + 1],
                self.series[start_price_col].iloc[r],
                self.series[end_price_close_col].iloc[r + lookforward]
            )
            for r in range(len(self.series) - lookforward)]

        self.series.loc[self.series.index[:-lookforward],
                        long_or_short + '_' + name + '_next_' + str(lookforward) + '_end_price_diff'] = (
            self.series[long_or_short + '_' + name + '_next_' + str(lookforward) + '_end_price'].iloc[:-lookforward]
            - self.series[start_price_col].iloc[:-lookforward]
        )

        if long_or_short == 'long':
            self.series.loc[self.series.index[:-lookforward],
                            long_or_short+'_'+name+'_next_'+str(lookforward)+'_profit'] = (
                self.series[long_or_short+'_'+name+'_next_'+str(lookforward)+'_end_price_diff'].iloc[:-lookforward])
        elif long_or_short == 'short':
            self.series.loc[self.series.index[:-lookforward],
                            long_or_short+'_'+name+'_next_'+str(lookforward)+'_profit'] = (
                - self.series[long_or_short+'_'+name+'_next_'+str(lookforward)+'_end_price_diff'].iloc[:-lookforward])

    def stoploss_limit_outcome_variable_limits(
            self, start_price_col, end_price_high_col, end_price_low_col, end_price_close_col,
            lookforward, high_limit_col, low_limit_col, long_or_short, name):
        """
        start price should be ask for long position, bid for short position
        end prices should be bid for long position, ask for short position
        if breach high and low limits in same period then assumes worst case scenario
        high_limit and low_limit expressed as difference between current price
        """

        def first_breach_low(low_prices, start_price, low_limit):
            breaches = [i + 1 for i, p in enumerate(low_prices) if p <= start_price + low_limit]
            if len(breaches) == 0:
                return False
            return breaches[0]

        def first_breach_high(high_prices, start_price, high_limit):
            breaches = [i + 1 for i, p in enumerate(high_prices) if p >= start_price + high_limit]
            if len(breaches) == 0:
                return False
            return breaches[0]

        def calculate_return(low_prices, high_prices, start_price, no_breach_end_price,
                             low_limit_price, high_limit_price):
            try:
                low_limit = low_limit_price - start_price
            except TypeError:
                return np.nan

            try:
                high_limit = high_limit_price - start_price
            except TypeError:
                return np.nan

            if (low_limit >= 0) or (high_limit <= 0):
                return start_price

            low_breach = first_breach_low(low_prices, start_price, low_limit)
            high_breach = first_breach_high(high_prices, start_price, high_limit)

            if not low_breach and not high_breach:
                return no_breach_end_price

            if long_or_short == 'long':
                if not low_breach:
                    return start_price + high_limit
                elif not high_breach:
                    return start_price + low_limit
                elif low_breach <= high_breach:
                    return start_price + low_limit
                elif high_breach < low_breach:
                    return start_price + high_limit

            if long_or_short == 'short':
                if not high_breach:
                    return start_price + low_limit
                elif not low_breach:
                    return start_price + high_limit
                elif high_breach <= low_breach:
                    return start_price + high_limit
                elif low_breach < high_breach:
                    return start_price + low_limit

        self.series.loc[self.series.index[:-lookforward],
                        long_or_short + '_' + name + '_next_' + str(lookforward)+'_end_price'] = [
            calculate_return(
                self.series[end_price_low_col].iloc[r + 1:r + lookforward + 1],
                self.series[end_price_high_col].iloc[r + 1:r + lookforward + 1],
                self.series[start_price_col].iloc[r],
                self.series[end_price_close_col].iloc[r + lookforward],
                self.series[low_limit_col].iloc[r],
                self.series[high_limit_col].iloc[r]
            )
            for r in range(len(self.series) - lookforward)]

        self.series.loc[self.series.index[:-lookforward],
                        long_or_short + '_' + name + '_next_' + str(lookforward) + '_end_price_diff'] = (
            self.series[long_or_short + '_' + name + '_next_' + str(lookforward) + '_end_price'].iloc[:-lookforward]
            - self.series[start_price_col].iloc[:-lookforward]
        )

        if long_or_short == 'long':
            self.series.loc[self.series.index[:-lookforward],
                            long_or_short+'_'+name+'_next_'+str(lookforward)+'_profit'] = (
                self.series[long_or_short+'_'+name+'_next_'+str(lookforward)+'_end_price_diff'].iloc[:-lookforward])
        elif long_or_short == 'short':
            self.series.loc[self.series.index[:-lookforward],
                            long_or_short+'_'+name+'_next_'+str(lookforward)+'_profit'] = (
                - self.series[long_or_short+'_'+name+'_next_'+str(lookforward)+'_end_price_diff'].iloc[:-lookforward])






