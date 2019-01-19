from django.shortcuts import render
import quandl
from alpha_vantage.timeseries import TimeSeries
from iexfinance import get_historical_data, Stock
from tsic.settings import APLHA_VANTAGE_KEY, CURRENCY_KEY, INTRIO_ID, INTRIO_PASS
from funds.blb import *
from funds.mvo import *
from funds.statistics import *
from funds.models import *
import pandas as pd
from math import *
import math
import numpy as np
from datetime import datetime
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay
from dateutil.relativedelta import relativedelta
from django.utils.timezone import make_aware
import pytz
import csv
from bs4 import BeautifulSoup
import requests
import random
import statsmodels.tsa.stattools as ts
from scipy.stats.stats import pearsonr
from scipy.stats.mstats import gmean
from scipy.stats import t
from scipy.special import gamma
import time
import itertools
from selenium import webdriver
from selenium.webdriver.chrome.options import Options

'''
**************************
OPTIMIZATION
**************************
'''
def optimize_pairs(assets, sp100_assets, pairs_date, portfolio):
    # KEEP TRACK OF WHICH SP100 ASSETS ARE COINTEGRATED WITH TSIC ASSETS
    paired = []

    # RETURN A 5 TUPLE WITH
    #   1) TSIC ASSET
    #   2) SP100 ASSET THAT IS OPTIMALLY COINTEGRATED WITH THE TSIC ASSET
    #   3) THE CORRELATION BETWEEN THE PAIR DURING THE LAST SIX MONTHS OF THE
    #      IN SAMPLE PERIOD
    #   4) THE HEDGING RATIO BETWEEN THE PAIRS, WHICH INDICATES HOW MUCH OF THE
    #      SP100 ASSET SHOULD BE SHORTED
    #   5) THE AMOUNT BY WHICH THE TSIC ASSET WILL OUT/UNDER PERFORM ITS COUPLED SP100 ASSET
    asset_pairs = []

    # INITIALIZING THE FUNCTIONS RETURN
    optimal_pairs = {}

    # INITIALIZING THE DATAFRAME OF RETURNED SP100 PRICE DATA
    pair_prices = pd.DataFrame()

    for asset in assets:
        prices = pd.DataFrame(assets[asset]).values

        # TEST FOR ANY ERRORS IN THE SP100 DATA, IF ANY, OMIT AND CONTINUE
        if pd.DataFrame(prices).isnull().values.any():
            unpairable = (None, None, None, None)
            optimal_pairs[asset] = unpairable
            continue

        # FLAG TO SEE IF WE CAN FIND A PAIR FOR OUR ASSET
        asset_paired = False

        for sp100_asset in sp100_assets:
            sp100_price = pd.DataFrame(sp100_assets[sp100_asset]).values

            # TEST FOR ANY ERRORS IN THE SP100 DATA, IF ANY, OMIT AND CONTINUE
            if pd.DataFrame(sp100_price).isnull().values.any():
                continue

            # CALCULATE THE SPREAD
            spread = sp100_price - prices

            # CALCULATE FROM THE FIRST SIX MONTHS OF IN-SAMPLE DATA, THE AVERAGE SPREAD AND
            # THE STANDARD DEVIATION FOR THE SPREAD
            spread_avg = spread[::int(len(spread)/2)].mean()
            spread_std = spread[::int(len(spread)/2)].std(ddof=1)

            # FROM THE LAST 6 MONTHS, CALCULATE THE AVERAGE SPREAD TO INFORM US ON WHICH
            # SP100 WILL BE OPTIMAL TO SHORT
            spread_6month = spread[int(len(spread)/2)::].mean()

            # STANDARDIZE THIS VARIABLE
            spread_z = (spread_6month - spread_avg) / spread_std

            # FOR THE PURPOSE OF GENERATING VIEWS, FIND THE AVERAGE RETURN OF THE SP100 ASSET
            # AVERAGE OVER THE LAST 6 MONTHS OF IN-SAMPLE DATA
            sp100_avg = get_returns(pd.DataFrame(sp100_assets[sp100_asset].iloc[int(len(spread)/2):len(spread)]), None).values.mean()

            # DETERMINE HOW MUCH THE TSIC ASSET WILL OUT/UNDER PERFORM THE SP100 ASSET
            #   -> WE BET THAT THE ASSETS WILL CONVERGE TO THE SPREAD AVERAGE
            #   --> (r_y)Y - (r_x)X = spread_avg
            #   ---> (r_x) - (r_y) =  (r_y)(Y/X - 1) - spread_avg/X
            view = sp100_avg * ((sp100_price[len(sp100_price)-1][0] / prices[len(prices)-1][0]) - 1) - \
                        spread_avg/prices[len(prices)-1][0]

            # FOR SENATIVITY PURPOSES, TAKE SPREAD_AVG +/- ONE STD FOR LOW/HIGH ESTIMATES
            view_low = sp100_avg * ((sp100_price[len(sp100_price)-1][0] / prices[len(prices)-1][0]) - 1) - \
                        (spread_avg+spread_std)/prices[len(prices)-1][0]
            view_high = sp100_avg * ((sp100_price[len(sp100_price)-1][0] / prices[len(prices)-1][0]) - 1) - \
                        (spread_avg-spread_std)/prices[len(prices)-1][0]

            # WE PASS THIS TUPLE ON AS THE LAST ENTRY IN THE OPTIMAL TUPLE RETURNED BY THIS
            # FUNCTION
            view_values = (view, view_low, view_high)

            # PERFORM THE ADT TEST USING THE FOLLOWING METHOD
            result = ts.adfuller(spread.flatten())

            if result[0] < result[4]['5%']:
                # HERE WE HAVE SUFFICIENT EVIDENCE TO SAY THAT THE ASSETS ARE COINTEGRATED
                asset_paired = True

                shorting_ratio = math.pi

                asset_pairs.append((asset, sp100_asset, spread_z, shorting_ratio, view_values))
                paired.append(sp100_asset)

        if not asset_paired:
            unpairable = (None, None, None, None)
            optimal_pairs[asset] = unpairable

    print("\n********** ********** **********")
    print("COINTERATION TEST RESULTS")
    print(asset_pairs)
    print("********** ********** **********\n")

    print("\n********** ********** **********")
    print("OPTIMIZE COINTEGRATED ASSETS")
    print("********** ********** **********\n")

    # SOLVE A MILP TO OPTIMIZE THE SELECTION OF CORRELATED ASSETS
    m = Model('maximize pair trading correlation')
    vars = pd.Series()

    # DEFINE A BINARY VARIABLE FOR EACH POTENTIAL PAIRING
    for tuple in asset_pairs:
        vars[tuple[0] + "-" + tuple[1]] = m.addVar(vtype=GRB.INTEGER,
                                                   obj=tuple[2],
                                                   name=tuple[0] + "-" + tuple[1])

    # UPDATE THE GUROBI MODEL
    m.update()

    # CONSTRAINT TO STRIVE FOR REPRESENTATION FROM AT MOST ONE TSIC ASSET
    #   ie. A TSIC ASSET CANNOT BE PAIRED WITH TWO (OR MORE) SP100 ASSETS
    for asset in assets:
        constraint = vars.filter(like=asset + "-")

        if len(constraint.index) != 0:
            m.addConstr(constraint.sum() <= 1)

    # CONSTRAINT TO STRIVE FOR REPRESENTATION FROM AT MOST ONE SP100 ASSET
    #   ie. AN SP100 ASSET CANNOT BE PAIRED WITH TWO (OR MORE) TSIC ASSETS
    for pair in paired:
        constraint = vars.filter(like="-" + pair)

        if len(constraint.index) != 0:
            m.addConstr(constraint.sum() <= 1)

    m.setParam('OutputFlag', 0)
    m.modelSense = GRB.MAXIMIZE

    # OPTIMIZE
    m.optimize()

    # IF THE MODEL IS INFEASIBLE, THEN GUROBI WILL NOT DEFINE AN ATTRRIBUTE X
    try:
        optimal_solution = m.x
    except:
        print("\nTHE PAIRS-TRADING OPTIMIZATION WAS INFEASIBLE - ONLY USE TSIC ASSETS\n")

        return None, None, None

    # INITIALIZE THE LIST OF PAIRED SP100 ASSETS
    best_paired = []

    # GET ALL OF THE OLD SP100 ASSETS WHICH WERE PAIRED WITH TSIC ASSETS
    old_paired = Asset.objects.filter(portfolio=portfolio, analyst_input=False)

    # REMOVE THE ASSOCIATION OF THESE ASSETS WITH THE PORTFOLIO, BUT KEEP THEM
    for item in old_paired:
        item.portfolio.remove(portfolio)
        item.save()

    for i in range(0, len(optimal_solution)):
        if optimal_solution[i] == 1:

            optimal_pairs[asset_pairs[i][0]] = asset_pairs[i][1::]

            best_paired.append(asset_pairs[i][1])

            pair_prices[asset_pairs[i][1]] = sp100_assets[asset_pairs[i][1]]

            # RETRIEVE THE TSIC ASSET
            tsic_asset = Asset.objects.get(ticker=asset_pairs[i][0])

            # CHECK IF THE PAIRED SP100 ASSET HAS ALREADY BEEN USED IN THE PORTFOLIO,
            check_pair = Asset.objects.filter(ticker=asset_pairs[i][1])

            if check_pair:
                check_pair[0].portfolio.add(portfolio)
                check_pair[0].coupled_asset = tsic_asset
                check_pair[0].save()

                tsic_asset.coupled_asset = check_pair[0]
                tsic_asset.save()
            else:
                # ADD THE SP100 ASSET TO THE PORTFOLIO
                pair_asset = Asset.objects.create(name=asset_pairs[i][1],
                                                  ticker=asset_pairs[i][1],
                                                  date_added=pairs_date,
                                                  analyst_input=False,
                                                  coupled_asset=tsic_asset,
                                                  value=0,
                                                  value_low=0,
                                                  value_high=0)

                pair_asset.portfolio.add(portfolio)
                pair_asset.save()

                # SET THE COUPLE ON THE TSIC ASSET
                tsic_asset.coupled_asset = pair_asset
                tsic_asset.save()

            # LAST THING TO DO IS TO CREATE THE RELATIVE VIEW BASED ON THE PREDICTED CONVERGENCE
            # OF THE SPREAD
            set_relative_views(portfolio, (asset_pairs[i][0], asset_pairs[i][1]), asset_pairs[i][4], pairs_date)


    print("\nTHE OPTIMAL PAIRS-TRADING STRATEGY IS")
    print(optimal_pairs)
    print("\n")

    return optimal_pairs, best_paired, pair_prices

def optimization_params(tsic_prices, sp100_prices, risk_free, portfolio):
    '''
        :description: sets the parameters for optimization and returns optimal
                      asset allocations

        :param tsic_prices: dataframe from in-sample period for tsic asset prices
        :param sp100_prices: dataframe from in-sample period for sp100 asset prices
        :param risk_free: dataframe from in-sample period for 26 week t-bill yields
        :param portfolio: the portfolio being optimized


        :returns select_sp100: a list containing the sp100 assets used in the optimization
        :reuturns optimal_portfolio: list of asset allocations in the optiaml portfolio
    '''

    if portfolio.fund.name == "SPY":
        return None, {"SPY":1}

    if tsic_prices is None:
        tsic_prices = get_data(None, 3, portfolio.id, None, None)

    # RETRIEVE THE LAST DATE IN THE IN-SAMPLE PERIOD, USED WHEN ADDING PAIRED ASSETS
    in_end = datetime.strptime(tsic_prices.iloc[len(tsic_prices)-1].name, "%Y-%m-%d")

    # MAKE INACTIVE ALL OLD VIEWS
    View.objects.filter(portfolio=portfolio, active=True).update(active=False)

    # SET THE ABSOLUTE VIEWS FOR THE TSIC ASSETS, BACKED BY INFORMATION FROM ANALYSTS
    set_absolute_views(portfolio, list(tsic_prices.columns),
                       tsic_prices.iloc[len(tsic_prices)-1], in_end)

    # WE IMPLEMENT A PAIRS-TRADING STRATEGY, SO FIND THE OPTIMAL ASSETS FROM THE
    # SP100 TO GO LONG/SHORT AGAINST OUR TSIC ASSETS
    optimal_pairs, select_sp100, pair_prices = optimize_pairs(tsic_prices, sp100_prices,
                                                             in_end, portfolio)

    # COMBINE OUR TSIC ASSET PRICES WITH OUR OPTIMIZED PAIRS ASSET PRICES
    prices = pd.concat([tsic_prices, pair_prices], axis=1, sort=False)

    # THE SPOT PRICES WILL BE THE PRICE ON THE LAST DAY ON THE IN-SAMPLE PERIOD
    spot_price = prices.iloc[len(prices)-1]

    # CALCULATE THE CORRECTED RETURNS FOR ALL OF THE ASSETS
    returns = get_returns(prices, risk_free)

    # SINCE OUR RISK FREE RATE WILL BE USED TO COMPOUND OUR CASH HOLDING, RENAME
    # THE COLUMN TO CASH
    risk_free.columns = ['CASH']

    # CONCATENATE THE RISK FREE RETURNS WITH THE ASSET PRICE RETURN DATAFRAME
    returns = pd.concat([risk_free, returns], axis=1, sort=False)
    returns = returns.dropna(axis=0, how='any')

    # RECORD THE ASSETS BEING OPTIMIZED
    assets = returns.columns

    # CALCULAUTING SUMMARY STATISTICS
    #   - COVARIANCE
    covariance = returns.cov()

    #   - GEOMETRIC RETURNS
    geometric_returns = gmean(returns+1, axis=0)-1

    expected_returns = pd.Series()
    for i in range(0, len(geometric_returns)):
        expected_returns[assets[i]] = geometric_returns[i]

    # TODO: GET THE FACTOR LOADINGS BASED ON OUR FACTOR MODEL
    factors, beta = get_factors(returns, covariance)

    # GET THE OPTIMIZATION MODEL FOR THE FUND
    model = Optimization.objects.get(id=portfolio.fund.optimization_model_id).key
    optimization_model = Optimization.KEY_CHOICES[model][1]

    # GET THE OBJECTIVE OPTIMIZATION, WHICH WILL DECIDE THE UTILITY FUNCTION
    objective = Fund.OBJECTIVE_CHOICES[portfolio.fund.objective][1]

    # TODO: INCORPORATE USER SELECT CONSTRAINTS AND HEDGING REQUIREMENTS
    constraints = None
    hedge = None

    optimal_portfolio = optimize(portfolio.fund, optimal_pairs, factors, beta,
                                 expected_returns, covariance, risk_free, spot_price,
                                 optimization_model, objective, constraints, hedge)

    return select_sp100, optimal_portfolio

def optimize(fund, optimal_pairs, factors, beta, expected_returns,
             covariance, risk_free, price, optimization_model,
             objective, constraints, hedge):
    '''
        :description: handles the optimization using gurobi

        :param assets: list of the asset tickers
        :param factors: Dataframe which contains the relative data on the factors
                        for a generic factor model.
        :param beta: Dataframe which contains the regression coefficients for the
                     generic factor model.
        :param expected_returns: Dataframe for the expected returns
        :param covariance: Dataframe for the covariance matrix
        :param optimization_model: Specified by the fund
        :param objective: Specified by the fund.
        :param constraints: List which specifies which constraints are applied
        :param hedge: Object which specifies which how the portfolio is hedged

        :returns: Series of optimal values of the funds portfolio
    '''

    # OUR BENCHMARK PORTFOLIO IS THE SPY INDEX WHICH TRACKS THE SPY ... BENCHMARK IS THE
    # LONG PORFTOLIO ONLY IN SPY
    if optimization_model == "SPY":
        return {'SPY': 1}

    assets = list(covariance.index)

    # OPTIMIZE BASED ON THE SPECIFIED MODEL
    if optimization_model == "mvo":
        optimal_list = mvo(objective, expected_returns, covariance, assets, optimal_pairs)
    else:
        optimal_list = blb(fund, assets, optimal_pairs, expected_returns, covariance, risk_free)

    print("\nOPTIMIZATION RESULTS:")
    optimal_dict = dict( (assets[i], optimal_list[i]) for i in range(0, len(assets)) )
    print(optimal_dict)
    print("\n")

    return optimal_dict

def get_factors(returns, covariance):
    return None, None

def hedge(portfolio):
    pass

def update_weights(bond_fv, out_sample_prices, optimal_portfolio, portfolio, update_date, bond_yield):
    '''
        :description: calulates the quantities to be held for each held asset during
                      a rebalance.

        :param:

        :returns:
    '''

    # SEE IF WE ARE INITIALIZING THE PORTFOLIO OR WHETHER WE ARE REBALANCING IT
    old_portfolio_weights = Weight.objects.filter(portfolio=portfolio)

    if old_portfolio_weights:
        # HERE WE ARE VALUATING THE PORTFOLIO BASED ON THE OLD HOLDINGS
        old_prices = out_sample_prices.copy()

        # SLIGHT CORRECTION TO MAKE WITH THE BONDS ... WE WILL SELL AT FACE VALUE SINCE
        # WE ARE HODLING IT ... OUT_SAMPLE_PRICES DOES THE DISCOUNTING ...
        #   - NEED TO BE MINDFUL OF HOW WE CALCULATE THE VALUE ... SINCE WE ARE NOT ALWAYS
        #     REBALANCING THE PORTFOLIO WHEN THE BOND MATURES, THE PRICE WILL NOT NECESSARY
        #     BE THE SAME AS ITS FACE VALUE ...
        target_rebalance = portfolio.fund.last_rebalanced + relativedelta(weeks=52)

        try:
            time_to_maturity = (update_date - target_rebalance).days
        except:
            time_to_maturity = (update_date - target_rebalance.date()).days

        old_prices['CASH'] = min(bond_fv / math.pow(1 + bond_yield, float(time_to_maturity)), bond_fv)

        value = get_portfolio_value(portfolio, old_prices)
    else:
        # HERE WE ARE INITIALIZING THE PORTFOLIO
        value = float(portfolio.fund.initial_capital)

    # IF NO DATE IS PASSED WE ARE NOT BACK TESTING AND ARE RELYING ON REAL TIME UPDATES
    if update_date is None:
        update_date = datetime.now()

    current_price = dict( (optimal, out_sample_prices[optimal]) for optimal in optimal_portfolio )

    # REMOVE THE CURRENT WEIGHT STATUS ON ALL OF THE OLD ASSETS
    Weight.objects.filter(portfolio=portfolio, current_weight=True).update(current_weight=False)

    for optimal_asset in optimal_portfolio:
        asset = Asset.objects.get(ticker=optimal_asset)

        new_weight = Weight.objects.create(asset=asset,
                                           weight=optimal_portfolio[optimal_asset],
                                           portfolio=portfolio,
                                           date_allocated=update_date,
                                           current_weight=True,
                                           quantity=(value * optimal_portfolio[optimal_asset]) \
                                                    / current_price[optimal_asset])

    # UPDATE THE FUND
    fund = portfolio.fund
    fund.last_rebalanced = update_date
    fund.save()

'''
**************************
PRICE/DATA SCRAPING
**************************
'''
def get_data(tickers, index, portfolio_id, start_date, end_date):
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

    # IF NO TICKERS ARE PROVIDED, GET TSIC PRICES FOR ASSETS IN THE PORTFOLIO
    if tickers is None:

        if portfolio_id is None:
            assets = Asset.objects.all()
        else:
            assets = Asset.objects.filter(portfolio=portfolio_id,
                                          analyst_input=True)

        tickers = [x.ticker for x in assets if x.ticker != "CASH"]

    price = pd.DataFrame()

    if end_date is None:
        end_date = datetime.now()

    if start_date is None:
        start_date = end_date - relativedelta(years=1)

    for ticker in tickers:
        data = get_historical_data(ticker, start=start_date, end=end_date, output_format='pandas')
        price[ticker] = data[data_index[index]]

    return price

def get_stock_stats(ticker):
    # we use the api from IEX to get comprehensive stats on our stocks
    # api endpoint
    api = "https://api.iextrading.com/1.0/stock/" + ticker + "/stats"

    r = requests.get(api)

    return r.json()

def get_risk_free(start_date, end_date, dates, offline):
    # scrape data for the 30 day t bill to use as the risk free rate.
    # all thats required is to append the year to the following url

    url = 'https://www.treasury.gov/resource-center/data-chart-center/interest-rates/Pages/TextView.aspx?data=billRatesYear&year='

    if not offline:
        risk_data = pd.DataFrame(columns=['risk_free'])
        risk_data.index.name = 'date'

        for date in dates:
            risk_data.loc[date] = np.NaN
    else:
        risk_data = pd.read_csv("website/static/documents/risk_free.csv", index_col="date", dtype={'INDEX':str})

        return risk_data

    scrape_year = start_date.date().year

    print(scrape_year)

    while scrape_year <= end_date.year:
        year_url = url + str(scrape_year)

        r = requests.get(year_url)
        soup = BeautifulSoup(r.content, 'lxml')

        rate_data = soup.find_all('td', class_='text_view_data')
        index = 0
        date_match = False

        for rates in rate_data:
            if index == 0:
                format_date = rates.text.split("/")
                rate_date = "20" + format_date[2] + "-" + format_date[0] + "-" + format_date[1]

                if rate_date in dates:
                    date_match = True
                else:
                    date_match = False

                index = index + 1

            elif index == 8:
                # index 8 corresponds to the 26 week T-bill
                if date_match:
                    rate = float(rates.text)
                    daily_rate = math.pow(1 + rate/100, 1/(26*7)) - 1

                    risk_data.loc[rate_date] = daily_rate

                index = index + 1

            elif index == 10:
                index = 0

            else:
                index = index + 1

        scrape_year = scrape_year + 1

    # there are some days in which there is no yield ... on these
    # occasions, use the last yield available with a forward fill
    risk_data.fillna(method='ffill', inplace=True)

    risk_data.to_csv("website/static/documents/risk_free.csv")

    return risk_data

def get_sp100_stocks(assets):
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

'''
**************************
CALCULATE and SUMMARIZE
**************************
'''
def set_absolute_views(portfolio, assets, spot_price, date):
    for asset in assets:
        stock = Asset.objects.get(ticker=asset)

        # GET THE INPUTTED PRICE RANGES FOR THE STOCK
        value = stock.value
        value_high = stock.value_high
        value_low = stock.value_low
        value_mid = (value_high+value_low)/2

        get_rebalance = portfolio.fund.rebalance_method
        rebalance_time = int(get_rebalance.name.split(".")[1])

        # COMPUTING THE VIEWED RETURNS BASED ON DAILY RETURNS
        viewed_ret = 1 / (20*rebalance_time) * math.log(value / spot_price[asset])

        # USE THE DIFFERENCE BETWEEN VALUE_HIGH AND VALUE_LOW AS THE BASE FOR OUR SENSITIVITY:
        #   -> WIDER RANGE MEANS MORE UNCERTAINTY
        range_sensitivity = value / (value + value_high - value_low)

        # SCALE THE SENSITIVITY BY SEEING WHERE THE VALUE FALLS IN THE HIGH-LOW RANGE:
        #   -> IF THE VALUE IS CLOSER TO EITHER VALUE_LOW OR VALUE_HIGH, THEN THERE IS LESS
        #      UNCERTAINTY AS WE ARE CLOSER TO EITHER EXTREME
        position_sensitivity = (abs(value - value_mid) + 1) / (value_high - value_low + 1)

        # CALCULATE THE UNCERTAINTY
        xi = 0.75 + (1 + position_sensitivity) * (1 - range_sensitivity)

        # DEFINE SOME RANDOM ELEMENT TO THE UNCERTAINTY
        random = np.random.uniform(low=0, high=1)

        View.objects.create(name="absolute-"+asset,
                            active=True,
                            date_observed=date,
                            portfolio=portfolio,
                            primary_asset=stock,
                            kind=0,
                            value=viewed_ret,
                            sensitivty=abs(1*xi + 0*random))

def set_relative_views(portfolio, asset_pairs, value, date):
    # FIND THE PRIMARY AND SECONDARY ASSETS, TO BE USED LATER ON TO CONSTRUCT OUR PICK MATRIX
    primary_asset = Asset.objects.get(ticker=asset_pairs[0])
    secondary_asset = Asset.objects.get(ticker=asset_pairs[1])

    # DEFINE OUR VALUE SENSITIVITIES, ADJUSTED FOR THE INVESTMENT PERIOD
    view = 1/(20*6) * value[0]
    view_low = 1/(20*6) * value[1]
    view_high = 1/(20*6) * value[2]
    view_mid = (view_low+view_high)/2

    # USE THE DIFFERENCE BETWEEN VALUE_HIGH AND VALUE_LOW AS THE BASE FOR OUR SENSITIVITY:
    #   -> WIDER RANGE MEANS MORE UNCERTAINTY
    range_sensitivity = view / (view + view_high - view_low)

    # SCALE THE SENSITIVITY BY SEEING WHERE THE VALUE FALLS IN THE HIGH-LOW RANGE:
    #   -> IF THE VALUE IS CLOSER TO EITHER VALUE_LOW OR VALUE_HIGH, THEN THERE IS LESS
    #      UNCERTAINTY AS WE ARE CLOSER TO EITHER EXTREMUM
    position_sensitivity = (abs(view - view_mid) + 1) / (view_high - view_low + 1)

    # CALCULATE THE UNCERTAINTY
    xi = 0.75 + (1 + position_sensitivity) * (1 - range_sensitivity)

    # DEFINE SOME RANDOM ELEMENT TO THE UNCERTAINTY
    random = np.random.uniform(low=0, high=1)

    View.objects.create(name="relative-%s-%s" % (asset_pairs[0], asset_pairs[1]),
                        active=True,
                        date_observed=date,
                        portfolio=portfolio,
                        primary_asset=primary_asset,
                        secondary_asset=secondary_asset,
                        kind=1,
                        value=view,
                        sensitivty=abs(1*xi + 0*random))

def get_returns(data, risk_free):
    '''
        :description: calculates the daily log returns from the price data provided.

        :param tickers: price data in a Dataframe

        :returns: returns
    '''

    returns = np.log(data / data.shift(1))

    # HACKY WAY TO INCLUDE EXCESS RETURNS OR NOT
    if isinstance(risk_free, pd.DataFrame):
        excess_returns = returns[1:].sub(risk_free['risk_free'][1:], axis=0)
    else:
        excess_returns = returns[1:]

    corrected_returns = pd.DataFrame()

    # we remove the serial correlation by performing AR(1)
    for asset in excess_returns:
        design_matrix = pd.DataFrame(excess_returns[asset][:-1])
        design_matrix["1"] = 1

        R = pd.DataFrame(excess_returns[asset][1:])

        beta = np.linalg.pinv((design_matrix.T).dot(design_matrix)).dot(design_matrix.T).dot(R)

        corrected_returns[asset] = (excess_returns[asset][1:] - beta[0] * excess_returns[asset][:-1] ) / (1 - beta[0])

    return corrected_returns[1:len(corrected_returns)-1]

def get_portfolio_value(portfolio, prices):

    # QUERY THE ACTIVE WEIGHTS
    weights = Weight.objects.filter(portfolio=portfolio, current_weight=True)

    # PUT THE QUANTITIES OF EACH ASSET INTO A LIST
    quantities = [ x.quantity for x in weights ]

    # IF PRICES IS A DATAFRAME, WE CALCULATE THE DAILY VALUE OF THE PORTFOLIO (DATAFRAME),
    # ELSE WE CALCULATE A SPOT VALUE OF THE PORTFOLIO (SERIES)
    if isinstance(prices, pd.DataFrame):
        current_prices = pd.DataFrame()
    else:
        current_prices = pd.Series()

    # SELECT THE ASSET PRICES THAT WE NEED
    for weight in weights:
        if isinstance(prices, pd.DataFrame):
            current_prices[weight.asset.ticker] = prices[weight.asset.ticker]
        else:
            current_prices.loc[weight.asset.ticker] = prices[weight.asset.ticker]

    # FOR THE DATAFRAMES, WE NEED TO SPECIFY THAT THE SUMMATION AXIS IS COLUMNS (INDEX=1)
    if isinstance(prices, pd.DataFrame):
        return current_prices.mul(quantities).sum(1)
    else:
        return current_prices.mul(quantities).sum()

def make_portfolio_summary(write_data, filename):
    with open("website/static/documents/" + filename, 'w') as file:
        writer = csv.writer(file)
        # writer.writerow(write_data.columns)
        writer.writerow(['date', 'value'])

        write_data.to_csv(file, header=False)

'''
**************************
OPTIMIZATION PROCEDURE
**************************
'''
def comply_dates(date, test_dates, backwards):
    while date.strftime('%Y-%m-%d') not in test_dates:
        if backwards:
            date = date - relativedelta(days=1)
        else:
            date = date + relativedelta(days=1)

    return date

def optimization(portfolio, start_date, end_date):
    '''
        :description: given a portfolio over a specified investment horizon,
                      track the daily value.

        :param start_date: starting date, datetime variable
        :param end_date: ending date, datetime variable

        :returns: nothing
    '''

    # RECORD THE START-TIME TO DETERMINE HOW LONG THE BACK-TEST TOOK
    start_clock = time.time()

    # ALL_VALUES WILL TRACK THE DAILY VALUE OF THE PORTFOLIO
    all_values = pd.Series()

    # GET THE TSIC ANALYST RECOMMENDED STOCKS
    ordered_assets = Asset.objects.filter(portfolio=portfolio,
                                          analyst_input=True).order_by('date_added')

    # GET THE TICKERS OF THE ASSETS THAT ARE RECOMMENDED FROM TSIC ANALYSTS
    assets = [x.ticker for x in ordered_assets if x.ticker != "CASH"]

    # CHECK TO SEE IF WE ARE REBALANCING OR PERFORMING A BACK-CHECK
    if start_date is None:
        back_check = True

        # IF DOING A BACK-TEST, THE STARTING DATE IS THE DATE THAT THE FIRST ASSET WAS ADDED
        start_date = ordered_assets[0].date_added.date()
    else:
        back_check = False
        start_date = start_date.date()

    # IF BACKCHECK, RESET: PAIRED-ASSETS, ASSET-ALLOCATION WEIGHTS, VIEWS AND PORTFOLIO STATISTICS
    if back_check:
        Asset.objects.filter(portfolio=portfolio, analyst_input=False).delete()
        Weight.objects.filter(portfolio=portfolio).delete()
        View.objects.filter(portfolio=portfolio).delete()
        PortfolioStatistics.objects.filter(portfolio=portfolio).delete()

    print("\n********** ********** **********")
    print("SCRAPING TSIC PRICING INFORMATION")
    print("********** ********** **********\n")

    # GET ALL OF THE RELEVANT PRICING DATA FOR THE TSIC ASSETS
    beginning = start_date - relativedelta(years=1, days=17)
    data = get_data(assets, 3, portfolio.id, beginning, None)

    # HERE WE UPDATE THE CSV DATA FOR THE TSIC ASSETS
    if back_check:
        data.to_csv("website/static/documents/"+ portfolio.fund.name +"_offline_data.csv")

    # data = pd.read_csv("website/static/documents/"+ portfolio.fund.name +"_offline_data.csv", index_col="date", header="infer")

    print("\nTSIC PRICING INFORMATION")
    print(data)
    print("\n")

    # GET TICKERS OF ASSETS CURRENTLY IN THE SP100
    sp100_assets = get_sp100_stocks(assets)

    print("\n********** ********** **********")
    print("SCRAPING SP100 PRICING INFORMATION")
    print("********** ********** **********\n")

    # SIMILARLY, SCRAPE PRICING DATA FOR ALL SP100 ASSETS OVER THE BACK-TEST PERIOD
    #   - SLEEP TO ENSURE THAT ANY API THROTTLING FROM IEX DOES NOT CAUSE A TIME-OUT
    time.sleep(1)
    sp100_data = get_data(sp100_assets, 3, portfolio.id, beginning, None)

    if back_check:
        sp100_data.to_csv("website/static/documents/sp100_offline_data.csv")

    print("\nSP100 PRICING INFORMATION")
    print(sp100_data)
    print("\n")

    # BASED ON THE ABOVE DATA, KEEP TRACK OF ALL DAYS WHICH HAVE PRICE DATA
    available_dates = list(data.index)
    last_date = datetime.strptime(data.iloc[len(data)-1].name, '%Y-%m-%d').date()

    # OUT-OF-SAMPLE DATES, NOTING THE REBALANCE PERIOD SPECIFIED BY THE FUND
    out_start = comply_dates(start_date, available_dates, True)

    # GET THE FUND TO DETERMINE REBALANCE FREQUENCY
    fund = Fund.objects.get(id=portfolio.fund.id)
    get_rebalance = fund.rebalance_method

    # IF THE FUND HAS NO REBALANCE PERIOD AND IS TO BE HELD AFTER INITIAL OPTIMIZATION,
    # SET THE END DATE TO MATCH THE LAST DATE, OR IF WE ARE REBALANCING ON THE CURRENT PERIOD
    if get_rebalance.name == "none" or not back_check:
        out_end = last_date
    else:
        rebalance_time = relativedelta(months=int(get_rebalance.name.split(".")[1]))
        out_end = comply_dates(out_start + rebalance_time, available_dates, True)

    # SAVE THE FUND'S START DATE
    fund.started = start_date
    fund.save()

    # IN-SAMPLE DATES
    in_start = datetime.strptime(data.iloc[0].name, '%Y-%m-%d')
    in_end = comply_dates(out_start - relativedelta(days=1), available_dates, True)

    # NOW GET THE DAILY-COMPOUNDED YIELDS ON A 26 WEEK T-BILL
    risk_free = get_risk_free(in_start, last_date, available_dates, False)

    print("\nBOND YIELDS")
    print(risk_free)
    print("\n")

    while out_start <= last_date:
        print("\n********** ********** **********")
        print("IN-SAMPLE PERIOD RUNS FROM " + in_start.strftime('%Y-%m-%d') + " to " + in_end.strftime('%Y-%m-%d'))
        print("********** ********** **********\n")

        # MAKE A COPY OF THE DATA THAT WE PULLED, MEANT TO BE MALLEABLE
        tsic_prices = data.copy()
        sp100_prices = sp100_data.copy()

        # CHECK IF WE ARE IN THE LAST INVESTMENT PERIOD
        if out_end > last_date:
            out_end = last_date

        # KEEP TRACK OF ALL TSIC ASSETS THAT WERE ADDED DURING THE IN-SAMPLE PERIOD
        in_period_assets = [x.ticker for x in ordered_assets \
                                if x.date_added.date() <= out_start and \
                                   x.ticker != "CASH"]

        # WE ONLY TRACK ASSETS WHICH HAVE BEEN ADDED PRIOR TO THE START OF OUR OUT-OF-SAMPLE PERIOD
        tsic_prices = tsic_prices[in_period_assets]

        # THE INDEXING IN OUR DATAFRAMES IS A DATE TYPE CASTED AS A STRING
        in_start_text = in_start.strftime('%Y-%m-%d')
        in_end_text = in_end.strftime('%Y-%m-%d')

        # GET THE IN-SAMPLE DATA FOR THE TSIC ASSETS, SP100 ASSETS AND RISK FREE
        in_sample_tsic = tsic_prices.loc[in_start_text:in_end_text]
        in_sample_sp100 = sp100_prices.loc[in_start_text:in_end_text]
        in_sample_rf = risk_free.loc[in_start_text:in_end_text]

        # OPTIMIZATION
        selected_sp100, optimal = optimization_params(in_sample_tsic, in_sample_sp100,
                                                      in_sample_rf, portfolio)

        print("\n********** ********** **********")
        print("OUT-OF-SAMPLE PERIOD RUNS FROM " + out_start.strftime('%Y-%m-%d') + " to " + out_end.strftime('%Y-%m-%d'))
        print("********** ********** **********\n")

        # THE INDEXING IN OUR DATAFRAMES IS A DATE, TYPE-CASTED AS A STRING
        out_start_text = out_start.strftime('%Y-%m-%d')
        out_end_text = out_end.strftime('%Y-%m-%d')

        # GET THE OUT-OF-SAMPLE DATA FOR THE TSIC ASSETS, SP100 ASSETS AND RISK FREE
        out_sample_tsic = tsic_prices.loc[out_start_text:out_end_text]
        out_sample_sp100 = sp100_prices.loc[out_start_text:out_end_text]
        out_sample_rf = risk_free.loc[out_start_text:out_end_text]

        # WE PARK OUR CASH INTO A 26 WEEK T-BILL, SO CALCULATE THE PRICES WITH A FACE VALUE OF 100
        out_sample_bond = pd.DataFrame(columns=["CASH"])
        face_value = 100

        # PRICE OF THE 26 WEEK T-BILL, ISSUED ON THE START OF THE OUT-OF-SAMPLE PERIOD
        for i in range(0, len(out_sample_rf)):
            date_text = out_sample_rf.iloc[i].name
            date_datetime = datetime.strptime(date_text, '%Y-%m-%d').date()

            rate = out_sample_rf.iloc[i]

            maturity_date = out_start + relativedelta(weeks=26)

            time_to_maturity = (maturity_date - date_datetime).days

            out_sample_bond.loc[date_text] = min(face_value / math.pow(1 + rate, float(time_to_maturity)), face_value)

        # CONCATENATE ALL PRICES INTO A SINGLE DATAFRAME
        out_sample_prices = pd.concat( [out_sample_bond, out_sample_tsic, out_sample_sp100],
                                        axis=1, sort=False)

        # RECORD THE SPOT PRICES AT THE BEGINNING OF THE OUT-OF-SAMPLE PERIOD
        beginning_spot = out_sample_prices.iloc[0]

        print("\n OUT OF SAMPLE ASSET PRICES (ALL ASSETS)")
        print(out_sample_prices)
        print("\n")

        # UPDATE THE WEIGHTS OF THE PORTFOLIO
        update_weights(face_value, beginning_spot, optimal, portfolio, out_start, rate)

        if not back_check:
            end_clock = time.time()

            print("\n********** ********** **********")
            print("FINISHED REBALANCING FOR PORTFOLIO " + portfolio.name + " IN " + str(end_clock-start_clock) + " SECONDS")
            print("********** ********** **********\n")

            return

        # GET THE DAILY VALUES OF THE PORTFOLIO OVER THE OUT-OF-SAMPLE PERIOD
        values = get_portfolio_value(portfolio, out_sample_prices)

        # CONCATENATE THE VALUES
        all_values = all_values.append(values)

        # IF WE ARE IN THE LAST OUT-OF-SAMPLE PERIOD, WE CAN TERMINATE
        if out_end >= last_date:
            break
        else:
            # UPDATE OUR IN-SAMPLE and OUT-OF-SAMPLE PERIODS
            out_start = comply_dates(out_end + relativedelta(days=1), available_dates, False)
            out_end = comply_dates(out_start + rebalance_time, available_dates, True)

            in_start = comply_dates(out_start - relativedelta(years=1, days=7), available_dates, True)
            in_end = comply_dates(out_start - relativedelta(days=7), available_dates, False)

    spy_prices = get_data(['SPY'], 3, None, datetime.strptime(all_values.index[0], "%Y-%m-%d"),
                                            datetime.strptime(all_values.index[len(all_values)-1], "%Y-%m-%d"))['SPY']

    make_statistics(portfolio, all_values, spy_prices, risk_free['risk_free'])

    # for index, value in all_values.iteritems():
    #     PortfolioStatistics.objects.create(name=portfolio.name + "/" + str(index),
    #                                        portfolio=portfolio,
    #                                        date=datetime.strptime(index, '%Y-%m-%d'),
    #                                        value=value)

    make_portfolio_summary(all_values, portfolio.fund.name + "_portfolio_stats.csv")

    end_clock = time.time()

    print("\n********** ********** **********")
    print("FINISHED BACKCHECK IN " + str(end_clock-start_clock) + " SECONDS")
    print("********** ********** **********\n")

    return
