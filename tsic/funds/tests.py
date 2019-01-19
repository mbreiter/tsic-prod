# from django.test import TestCase
from iexfinance import get_historical_data, Stock
from bs4 import BeautifulSoup
import requests
import random
import numpy as np
import math
import pandas as pd
from datetime import datetime
from dateutil.relativedelta import relativedelta
import statsmodels.tsa.stattools as ts
from scipy.stats.stats import pearsonr

def sp100_stocks(assets):
    # go to wikipedia to get the list of stocks in the SP100
    url = "https://en.wikipedia.org/wiki/S%26P_100"

    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'lxml')

    ticker_table = soup.find_all('table', class_="wikitable sortable")[0]

    rows = ticker_table.findChildren('tr')

    tickers = []

    for row in rows:
        cols = row.find_all('td')

        if cols:
            cols = [ele.text.strip() for ele in cols]

            if cols[0] not in assets:
                tickers.append(cols[0])

    return tickers

def get_data(index, sp100, start_date, end_date):
    '''
        :description: returns price data over a specified period.

        :param tickers: list of the asset tickers for which data is pulled for
        :param index: Indicated which column in the resulting price data is gathered
        :param portfolio_id: id for the portfolio. If none, gets all assets
        :param start_date: Datetime indicating the start. If none, use the start
                           date as a year prior to the end date.
        :param end_date: Datetime indicating the end. If none, use the curent date

        :returns: price data
    '''
    data_index = ('open', 'high', 'low', 'close', 'volume')

    # we can choose to collect data on certain stocks by populating tickers.
    if not sp100:
        tickers = ['V','AXP','COT','JBLU','TRP','GGG','FL','AMGN']
    else:
        tickers = sp100_stocks(['V','AXP','COT','JBLU','TRP','GGG','FL','AMGN'])
        random.shuffle(tickers)
        tickers = tickers[0:5]

    price = pd.DataFrame()

    if end_date is None:
        end_date = datetime.now() - relativedelta(days=1)

    if start_date is None:
        start_date = end_date - relativedelta(years=1)

    for ticker in tickers:
        data = get_historical_data(ticker, start=start_date, end=end_date, output_format='pandas')
        price[ticker] = data[data_index[index]]

    return price

def get_pairs(assets, sp100_assets):
    # paired keeps track of the sp100 assets which are cointergrated with investor assets
    paired = []
    # asset_pairs is a 4-tuple with asset and sp100_asset correlation and beta information
    asset_pairs = []
    # initializing the functions return
    optimal_pairs = {}

    for asset in assets:
        prices = pd.DataFrame(assets[asset])
        prices["1"] = 1

        if prices.isnull().values.any():
            continue

        prices = prices.values

        # flag to check to see if we can find a pair for our asset
        asset_paired = False

        for sp100_asset in sp100_assets:
            sp100_price = pd.DataFrame(sp100_assets[sp100_asset]).values

            # test for any errors in pulling the data, if so, simply omit the sp100 stock
            if pd.DataFrame(sp100_price).isnull().values.any():
                continue

            # determine the spread by taking a regression of investor sp100
            # asset prices on tsic asset prices
            try:
                beta = np.linalg.pinv((prices.T).dot(prices)).dot(prices.T).dot(sp100_price)
            except:
                unpairable = (None, None, None)
                optimal_pairs[asset] = unpairable
                continue

            spread = sp100_price - prices.dot(beta)

            # calculuate the correlation coefficient from the last 6 months
            correlation = pearsonr(assets[asset].iloc[int(len(assets[asset])/2)::],
                                   sp100_assets[sp100_asset].iloc[int(len(sp100_assets[sp100_asset])/2)::])[0]

            delta_spread = np.diff(spread, axis=0)

            X = np.c_[ spread[1:len(spread)-1], delta_spread[0:len(delta_spread)-1], np.ones(len(spread)-2)]
            Y = spread[2::]

            beta_two = np.linalg.pinv((X.T).dot(X)).dot(X.T).dot(Y)

            # calculate the residual sum of squares
            RSS = np.power(Y - X.dot(beta_two), 2).sum(axis=0)

            # calculate the mean square error, MSE=RSS/(n-p-1)
            MSE = float(RSS) / (len(X) + len(beta_two) - 1)

            # calculate the variance for each of our OLS estimates
            var_ols = MSE * np.linalg.inv((X.T).dot(X))

            print("\nI CALCULATE THE T STATISTIC AS")
            # calculate our t-statistic
            t = float(beta_two[0] - 1) / (np.sqrt(var_ols[0][0]))
            print(t)

            # perform ADT test to test for stationarity
            result = ts.adfuller(spread.flatten(), 1)
            print(result)

            if result[0] < result[4]['5%']:
                # sufficient evidence from test to say that asset sp100_asset are
                # cointegrated -> spread between them is stationary

                # indicate to our flag that we found a pair
                asset_paired = True

                asset_pairs.append((asset, sp100_asset, correlation, beta[0][0]))
                paired.append(sp100_asset)

        if not asset_paired:
            unpairable = (None, None, None)
            optimal_pairs[asset] = unpairable

    print(asset_pairs)

    # we solve a mixed-integer linear program to optimize our selection of
    # correlated assets for our pairs trading strategy
    m = Model('maximize pair trading correlation')
    vars = pd.Series()

    # we will have a binary variable for eaching pairing in asset_pairs
    for tuple in asset_pairs:
        vars[tuple[0] + "-" + tuple[1]] = m.addVar(vtype=GRB.INTEGER,
                                                   obj=tuple[2],
                                                   name=tuple[0] + "-" + tuple[1])

    # gurobi requires an update to the model when variables are added
    m.update()

    print("\nTHE VARIABLES:")
    print(vars)

    # set the constraint allowing up to one paired asset per investor asset
    for asset in assets:

        constraint = vars.filter(like=asset + "-")

        if len(constraint.index) != 0:
            m.addConstr(constraint.sum() <= 1)

    # set constraint allowing up to one investor asset per paired asset
    for pair in paired:
        constraint = vars.filter(like="-" + pair)

        if len(constraint.index) != 0:
            m.addConstr(constraint.sum() <= 1)

    m.setParam('OutputFlag', 0)
    m.modelSense = GRB.MAXIMIZE

    # optimize:
    m.optimize()

    # if there is no solution, then the model has no attribute 'x'
    try:
        optimal_solution = m.x
    except:
        pass

    pair_prices = pd.DataFrame()

    for i in range(0, len(optimal_solution)):
        if optimal_solution[i] == 1:
            optimal_pairs[asset_pairs[i][0]] = asset_pairs[i][1::]
            pair_prices[asset_pairs[i][1]] = sp100_assets[asset_pairs[i][1]]

    print("\n The best pairs are: ")
    print(optimal_pairs)

    return optimal_pairs, pair_prices

if __name__ == "__main__":
    prices = get_data(3, False, None, None)
    sp100_prices = get_data(3, True, None, None)

    get_pairs(prices, sp100_prices)
