import pandas as pd
import numpy as np
import math
import itertools

from iexfinance import get_historical_data, Stock
from datetime import datetime
from scipy.optimize import minimize
from funds.models import *
from scipy.stats.mstats import gmean
from scipy.special import gamma
from scipy.stats import t

'''
**************************
BLACK-LITTERMAN-BAYES MODEL
**************************
'''
def blb(fund, assets, optimal_pairs, expected_returns, sigma, risk_free):

    # SET THE DEGREES OF FREEDOM
    #   -> TODO: FIND BETTER WAY TO SET DOF RATHER THAN JUST ARBITRARY
    df = 5

    # SELECT ALPHA
    alpha = 0.95

    # DEFINE THE DISPERSION MATRIX, WHICH IS PROPORTIONAL TO THE COVARIANCE MATRIX
    D = (df - 2)/df * sigma

    # CALCULATE THE RISK AVERSION COEFFICIENT
    risk_aversion = get_risk_aversion(risk_free)

    # CALCULATE THE CVAR COEFFICIENT
    A = get_CVaR_coef(df, alpha)

    # DETERMINE THE EQUILIBRIUM PORTFOLIO, A RISK-PARITY CVAR PORTFOLIO
    x_equil = equil_optimize(assets, A, expected_returns, sigma, ub=0.25, lb=-0.05)\

    print("\nTHE EQUILIBRIUM CVAR RISK PARTITY PORTFOLIO IS")
    print(x_equil)
    print("\n")

    # CALCULATE THE IMPLIED EQUILIBRIUM RETURNS UNDER OUR UTILITY FUNCTION
    mu_equil = get_equil_returns(risk_aversion, A, D, x_equil)

    # GET THE PORTFOLIO
    portfolio = Portfolio.objects.get(fund=fund)

    # GET THE VIEWS AND THE CORRESPONDING PARAMETERS
    v, P, omega = views_params(portfolio, assets, sigma)

    # CALCULATE THE BLACK-LITTERMAN PARAMETERS
    mu_bl, D_bl, sigma_bl = bl_param(v, P, omega, D, mu_equil, df)

    optimal_portfolio = blb_optimize(assets, mu_bl, sigma_bl, risk_aversion, A, optimal_pairs)

    return optimal_portfolio

def bl_param(v, P, omega, D, mu, df):
    assets = P.columns

    # k represents the number of views that we have
    k = len(P)

    # s defines the skew of the distribution
    s = 0

    v = v.values
    P = P.values
    omega = omega.values
    D = D.values
    mu = mu.values

    view_diff = v - P.dot(mu)
    view_var = np.linalg.pinv(omega + P.dot(D).dot(P.T))

    mu_bl = mu + D.dot(P.T).dot(view_var).dot(view_diff)
    D_bl = D - D.dot(P.T).dot(view_var).dot(P.dot(D))

    sigma_bl = math.pow(k + df -2, -1) * (s + (view_diff.T).dot(view_var).dot(view_diff) ) * D_bl

    return mu_bl, D_bl, sigma_bl

def views_params(portfolio, assets, covariance):

    # RETRIEVE ALL THE ACTIVE VIEWS
    views = View.objects.filter(portfolio=portfolio, active=True)

    # GET THE NUMBER OF VIEWS
    k = len(views)

    # GET THE NUMBER OF ASSETS
    n = len(assets)

    # IF K>N THEN WE TAKE K CHOOSE N COMBINATIONS OF VIEWS AND LATER, PERFORM BLB RESAMPLING
    # WITH ALL OF OUR VIEWS TAKEN INTO CONSIDERATION WITH EACH OTHER
    if k>=n:
        view_combos = itertools.combinations(views, n)
        k = n
    else:
        view_combos = itertools.combinations(views, k)

    # INTIALIZE THE LISTS THAT WILL HOLD OUR VIEW PARAMETERS FOR EACH COMBINATION
    P_combos = []
    omega_combos = []
    v_combos = []

    for view in view_combos:
        # GET THE NAMES OF THE VIEW_NAMES
        view_names = [x.name for x in view]

        # INITIALIZE OUR PICK MATRIX WITH ZEROS
        P = pd.DataFrame(np.zeros((k, n)), index=view_names, columns=assets)

        # INITIALIZE OUR VIEW UNCERTAINTY MATRIX WITH ZEROS
        omega = pd.DataFrame(np.zeros((k, k)), index=view_names, columns=view_names)

        # INITIALIZE THE VIEWS VECTOR, A PANDAS SERIES
        v = pd.Series()

        for item in view:
            # SET THE PRIMARY INDEX IN THE PICK MATRIX
            primary_index = assets.index(item.primary_asset.ticker)

            P.loc[item.name, assets[primary_index]] = 1

            # IF THIS VIEW IS A RELATIVE ONE, WE NEED TO DO THE SAME FOR THE SECONDARY ASSET
            if item.kind==1:
                secondary_index = assets.index(item.secondary_asset.ticker)

                P.loc[item.name, assets[primary_index]] = -1

            # SET THE VIEW VALUE
            v.loc[item.name] = item.value

            # SET THE SENSITIVITY
            omega.loc[item.name, item.name] = item.sensitivty * \
                                                P.loc[item.name] \
                                                .dot(covariance) \
                                                .dot(P.loc[item.name].T)

            P_combos.append(P)
            omega_combos.append(omega)
            v_combos.append(v)

    return v_combos[0], P_combos[0], omega_combos[0]

def get_risk_aversion(risk_free):
    # WE TAKE THE SPY INDEX AS A MARKET PROXY

    # START/END DATES CORRESPOND TO WHAT'S PASSED IN THE RISK_FREE DATAFRAME
    start_date = datetime.strptime(risk_free.iloc[0].name, '%Y-%m-%d')
    end_date = datetime.strptime(risk_free.iloc[len(risk_free)-1].name, '%Y-%m-%d')

    # RETRIEVE THE PRICES FOR THE SPY INDEX
    spy_prices = get_data(["SPY"], 3, None, start_date, end_date)

    # CHANGE THE NAME BACK TO RISK_FREE
    risk_free.columns=['risk_free']

    # CALCULATE EXCESS RETURNS ON THE SPY INDEX
    spy_returns = get_returns(spy_prices, risk_free)

    # CALCULATE THE EXPECTED EXCESS RETURN AND THE VARIANCE
    spy_georet = gmean(spy_returns+1, axis=0)[0]-1
    spy_var = spy_returns.var()[0]

    # RETURN THE RISK AVERSION COEFFICIENT
    return spy_georet / spy_var

def get_CVaR_coef(df, alpha):
    # the CVaR parameter is denoted "A" and is the coefficient on CVaR in the
    # mean-CVaR tradeoff optimization

    c = gamma((df+1) / 2) / ( math.sqrt(math.pi * df) * gamma(df/2) )

    # we find the 1-alpha quantitle from the standard t-distribution
    q = t.ppf(1-alpha, df)

    # calculate the CVaR coefficient
    A = c * df * math.pow(1 + q**2/df, (1-df)/2 ) / ( (1-alpha) * (df-1))

    return A

def get_equil_returns(risk_aversion, A, dispersion, weight):
    coefficient = risk_aversion*A / (1 + risk_aversion)
    denominator = 1 / np.sqrt((weight.T).dot(dispersion).dot(weight))
    vector = dispersion.dot(weight)

    return coefficient * denominator * vector

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

'''
**************************
SCIPY MEAN-CVaR OPTIMIZATION
**************************
'''
def blb_optimize(assets, mu, sigma, risk_aversion, A, optimal_pairs):
    # DEFINE OUR INITIAL GUESS FOR THE OPTIMIZATION AS THE EQUAL-WEIGHTING PORTFOLIO
    x0 = 1/len(assets) * np.ones(len(assets))

    # BACK OUT WHICH ASSETS CAME FROM THE SP100
    sp100_assets = [optimal_pairs[pair][0] for pair in optimal_pairs]

    lb = 0
    ub = 0.25

    # SET OUR BOUNDS FOR EACH ASSET
    bounds = [(-0.1, -0.015) if x in sp100_assets else (lb, ub) for x in assets]

    # FORMALIZE THE REQUIREMENT TO HOLD 10% IN A T-BILL
    cash_constaint = {'type':'eq', 'fun':blb_cash_constraint}

    # FORMALIZE OUR BUDGET CONSTRAINT
    budget_constraint = {'type':'eq', 'fun':blb_budget_constraint}

    constraints = [cash_constaint, budget_constraint]

    # OPTIMIZE
    solution = minimize(blb_objective, x0, args=(mu, risk_aversion, A, sigma), method='SLSQP',
                        bounds=bounds, constraints=constraints).x

    return solution

def blb_objective(x, mu, risk_aversion, A, sigma):
    sqrt = math.sqrt((x.T).dot(sigma).dot(x))

    # WE ARE LOOKING TO MAXIMIZE, SO WE MUST MULTIPLY BY -1 HERE AND THEN ADJUST THE SOLUTION
    return -1 * ((1-risk_aversion)*(mu.T).dot(x) - risk_aversion*A*math.sqrt(sqrt))

def blb_cash_constraint(x):
    return x[0] - 0.1

def blb_budget_constraint(x):
    weight_sum = 0

    for i in range(0, len(x)):
        weight_sum = weight_sum + x[i]

    return weight_sum - 1

'''
**************************
SCIPY EQUILIBRIUM OPTIMIZATION - CVaR RISK PARITY
**************************
'''

def equil_optimize(assets, A, mu, sigma, ub, lb):
    # DEFINE OUR INITIAL GUESS FOR THE OPTIMIZATION AS THE EQUAL-WEIGHTING PORTFOLIO
    x0 = 1/len(assets) * np.ones(len(assets)+1)

    # SET OUR BOUNDS
    bounds = [(lb, ub) for x in assets] + [(-100,100)]

    # FORMALIZE OUR BUDGET CONSTRAINT
    constraints = [{'type':'eq', 'fun':equil_budget_constraint}]

    solution = minimize(equil_objective, x0, args=(A, mu, sigma), method='SLSQP',
                        bounds=bounds, constraints=constraints).x

    equil_portfolio = pd.Series()

    test = solution[0:len(solution)-1]
    sigma_x = sigma.dot(test)
    denominator = math.sqrt((test.T).dot(sigma).dot(test))

    for i in range(0, len(assets)):
        equil_portfolio.loc[assets[i]]= solution[i]

    return equil_portfolio

def equil_objective(x, A, mu, sigma):
    objective = 0

    # WE DO THIS BECAUSE WE OPTIMIZE AN AUXILLARY VARIABLE THETA
    weights = x[0:len(x)-1]
    theta = x[len(x)-1]

    # NEED THESE TO SIMPLIFY THE FORMULATION SHOWN BELOW
    sigma_x = sigma.dot(weights)
    denominator = math.sqrt((weights.T).dot(sigma).dot(weights))

    for i in range(0, len(weights)):
        objective = objective + \
                    math.pow(0.5 * A * (weights[i]*sigma_x[i])/denominator - mu[i]*weights[i] - theta, 2)

    return objective

def equil_budget_constraint(x):
    weights = x[0:len(x)-1]
    weight_sum = 0

    for i in range(0, len(weights)):
        weight_sum = weight_sum + weights[i]

    return weight_sum - 1
