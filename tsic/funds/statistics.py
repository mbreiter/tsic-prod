import pandas as pd
import numpy as np
import math

from funds.models import *
from datetime import datetime

def make_statistics(portfolio, portfolio_values, sp500_values, risk_free):
    # IF WE GET A SINGLE PORTFOLIO VALUE, THEN PUT THAT INTO A SERIES AND CONCATENATE IT
    # WITH THE REST OF THE VALUES FROM WHICH WE CAN PULL
    # pd.Series( dict( (x.date, x.value) for x in PortfolioStatistics  )

    # if isintance(portfolio_values, float):
    #     portfolio_stats = PortfolioStatistics.objects.filter(portfolio=portfolio)
    #
    #     dict( (x.date, x.value) for x in portfolio_stats )

    portfolio_returns = trailing_returns(portfolio_values, 1)
    market_returns = trailing_returns(sp500_values, 1)

    # CALCULATE THE FOLLOWING TRAILING RETURNS:
    #   - 30 DAYS
    #   - 120 DAYS
    trailing_30 = trailing_returns(portfolio_values, 30)
    trailing_120 = trailing_returns(portfolio_values, 120)

    # CALCULATE THE FOLLOWING ROLLING RETURNS:
    #   - 30 DAYS
    #   - 120 DAYS
    rolling_30 = rolling_returns(portfolio_values, 30)
    rolling_120 = rolling_returns(portfolio_values, 120)

    # BASED ON THE INITIAL CAPITAL, CALCULATE THE RETURNS SINCE INCEPTION OF THE PORTFOLIO
    initial_capital = float(portfolio.fund.initial_capital)
    returns_inception =  returns_since(portfolio_values, initial_capital)

    for index, value in portfolio_values.iteritems():

        # HANDLING OUR 30-DAY TRAILING AND ROLLING STATISTICS ... WE ALSO NOW CALCULATE THE
        # BETA OF THE PORTFOLIO SINCE THERE HAS BEEN SUFFICIENT DATA GATHERED TO MAKE SIGNIFICANT
        try:
            t_30 = trailing_30.loc[index]
            r_30 = rolling_30.loc[index]

            beta = calculate_beta(portfolio_values.loc[:index], sp500_values.loc[:index])
            alpha = calculate_alpha(beta, risk_free.loc[index],
                                    portfolio_returns.loc[index], market_returns.loc[index])

            # CALCULATE THE SHARPE RATIO
            sharpe_ratio = calculate_sharpe(portfolio_values.loc[:index],
                                            risk_free.loc[:index])
        except Exception as e:
            print(e)
            t_30, r_30, beta, alpha, sharpe_ratio = 0, 0, 0, 0, 0

        # HANDLING OUR 120-DAY TRAILING AND ROLLING STATISTICS
        try:
            t_120 = trailing_120.loc[index]
            r_120 = rolling_120.loc[index]
        except:
            t_120, r_120 = 0, 0

        # CALCULATE RETURNS SINCE INCEPTION OF THE PORTFOLIO
        r_inception = returns_inception.loc[index]

        PortfolioStatistics.objects.create(name=portfolio.name + "/" + str(index),
                                           portfolio=portfolio,
                                           date=datetime.strptime(index, '%Y-%m-%d'),
                                           value=value,
                                           trailing_30=t_30,
                                           trailing_120=t_120,
                                           rolling_30=r_30,
                                           rolling_120=r_120,
                                           returns_inception=r_inception,
                                           alpha=alpha,
                                           beta=beta,
                                           sharpe_ratio=sharpe_ratio)

    print("\nSUCESSFULLY CAPTURED THE PORTFOLIO STATISTICS\n")

def calculate_sharpe(values, risk_free):
    value_returns = trailing_returns(values, 1)
    excess_returns = value_returns - risk_free

    # ANNUALIZE THE MOST EXCESS RETURN ON THE GIVEN DATE, AND ANNUALIZE
    avg_excess = (1 + excess_returns.mean())**252 - 1

    # CALCULATE THE STANDARD DEIVATION OF THE PORTFOLIO, LIFE-TO-DATE, AND ANNUALIZE
    std = math.sqrt(252) * value_returns.std()

    return avg_excess/std

def calculate_alpha(beta, risk_free, portfolio_return, market_return):
    return (portfolio_return - risk_free) - beta*(market_return - risk_free)

def calculate_beta(values, sp500_values):
    value_returns = trailing_returns(values, 1)
    sp500_returns = trailing_returns(sp500_values, 1)

    # fails below
    cov = value_returns.cov(sp500_returns)
    var_m = sp500_returns.var()

    return cov/var_m

def returns_since(values, base):
    return values / base - 1

def trailing_returns(values, window):
    return (values / values.shift(window) - 1).dropna()

def rolling_returns(values, window):
    returns = trailing_returns(values, 1)

    return returns.rolling(window).mean().dropna()
