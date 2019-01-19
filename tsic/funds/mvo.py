import pandas as pd
import numpy as np
import math

from scipy.optimize import minimize
from funds.models import *

def mvo(objective, mu, sigma, assets, optimal_pairs):
    # DEFINE OUR ASSETS BY KIND
    cash_asset = ['CASH']
    pair_assets = [optimal_pairs[x][0] for x in optimal_pairs if optimal_pairs[x][0] != None]
    tsic_assets = sigma.index.difference(cash_asset+pair_assets)

    # CREATE THE GUROBI MODEL
    m = Model('portfolio-MVO')

    # ADD A GUROBI VARIABLE FOR EACH ASSET
    vars = pd.Series(m.addVars(assets, lb=-0.10, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS), index=assets)

    # DEFINE THE PORTFOLIO'S VARIANCE AND EXPECTED RETURN
    portfolio_risk = sigma.dot(vars).dot(vars)
    portfolio_return = mu.dot(vars)

    # SET THE OBJECTIVE ON THE OPTIMIZATION
    if objective == "maximize returns":
        m.setObjective(portfolio_return, GRB.MAXIMIZE)
    else:
        m.setObjective(portfolio_risk, GRB.MINIMIZE)

    # ENFORCE THE PORTFOLIO'S BUDGET CONSTRAINT
    m.addConstr(vars.sum() == 1, 'budget')

    # cash holding of 10% constraint
    m.addConstr(vars.iloc[0] == 0.1, 'cash_holdings')

    # pair_index tracks where in vars is the start of the paired assets
    pair_index = len(cash_asset) + len(tsic_assets)

    # diversity constraints, no asset can be held for more than 25%
    m.addConstrs((0 <= vars.iloc[i] <= 0.25 for i in range(1, pair_index)), name='diversity')

    # we want to be shorting our paired assets
    m.addConstrs((vars.iloc[i] <= -0.02 for i in range(pair_index, len(vars))), name='shorting_pairs')

    # optimize model
    m.setParam('OutputFlag', 0)
    m.optimize()

    optimal_list = m.x

    return optimal_list
