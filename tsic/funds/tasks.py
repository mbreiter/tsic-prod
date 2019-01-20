from celery import Celery
from celery.task.schedules import crontab
from celery.decorators import periodic_task
from celery.utils.log import get_task_logger
from dateutil.relativedelta import relativedelta
from django.core.mail import EmailMessage
from funds.views import *
from funds.models import *
from website.models import *
from tsic.settings import EMAIL_HOST_USER, MEDIA_URL
from iexfinance import Stock
from datetime import datetime, timedelta
from django.utils import timezone as timez
import pandas as pd
import csv
import time

celery = Celery(__name__)
celery.config_from_object(__name__)
logger = get_task_logger(__name__)

# note: celery is stupid and uses UTC as a timezone despite me configuring it otherwise ... add 4 hours

# UPDATES PORTFOLIO VALUES AT 00h17ET WEEK-DAILY
@periodic_task(
    run_every=(crontab(hour=4, minute=17, day_of_week='1-5')),
    name="update_portfolio_numbers",
    ignore_result=True
)
def update_portfolio_numbers():
    ''' several steps:
            1. GET PRICING DATA
            2. CALCULATE THE PORTFOLIO'S VALUES
            3. CALCULATE PORTFOLIO STATISTICS
            4. WRITE RESULTS TO SPREADSHEET
            5. SEND EMAILS
    '''
    now = datetime.now()

    # GET ALL PORTFOLIO'S, EACH OF WHICH MUST BE UPDATED
    portfolios = Portfolio.objects.all()

    for portfolio in portfolios:
        prices = pd.Series()

        # GET ASSETS BASED ON THE ACTIVE WEIGHTS
        weights = Weight.objects.filter(portfolio=portfolio, current_weight=True)
        for weight in weights:
            asset = weight.asset.ticker

            if asset != 'CASH':
                prices[asset] = Stock(asset).get_price()
            else:
                current_yield = np.NaN
                yield_date = now

                # THE US TREASURY ONLY UPDATES THE YIELDS ON TRADING DAYS
                while True:
                    current_yield = get_risk_free(yield_date, yield_date, [yield_date.strftime('%Y-%m-%d')],
                                                  offline=False).iloc[0]
                    if not current_yield.isnull().values.any():
                        break
                    else:
                        yield_date = yield_date - relativedelta(days=1)

                time_to_maturity = (portfolio.fund.last_rebalanced \
                                    + relativedelta(weeks=26) - timez.now()).days

                prices['CASH'] = 100 / math.pow(1 + current_yield, time_to_maturity)

        value = get_portfolio_value(portfolio, prices)

        PortfolioStatistics.objects.create(name=portfolio.name + "/" + now.strftime('%Y-%m-%d'),
                                           portfolio=portfolio,
                                           date=now,
                                           value=value)

        write_summary(portfolio, now, value)

    logger.info("Successfully updated portfolio numbers")
    send_daily_emails()

    # get the latests daily risk free rate
    get_risk_free(now, now, [now.date().strftime('%Y-%m-%d')], False)
    logger.info("Successfully updated risk free rate")

    return "DONE"

def write_summary(portfolio, date, portfolio_value):

    with open('website/static/documents/' + portfolio.fund.name + '_portfolio_stats.csv', 'a') as file:
        writer=csv.writer(file, delimiter=',')
        writer.writerow([date.date().strftime('%Y-%m-%d'), portfolio_value])

    logger.info("Successfully wrote to spreadsheets")

# IF NEEDED, REBALANCE THE PORTFOLIO AT 00h00ET WEEK-DAILY
@periodic_task(
    run_every=(crontab(hour=4, minute=00, day_of_week='1-5')),
    name="rebalance_portfolios",
    ignore_result=True
)
def rebalance_portfolios():

    current_date = datetime.now()
    portfolios = Portfolio.objects.exclude(name="SPY")
    spot_prices = pd.Series()

    # THE LIST BELOW WILL TRACK WHICH PORTFOLIOS NEED REBALANCING
    rebalance_list = []

    # DO A QUICK CHECK TO SEE WHICH PORTFOLIOS NEED REBALANCING
    for portfolio in portfolios:
        fund = Fund.objects.get(id=portfolio.fund.id)
        get_rebalance = fund.rebalance_method
        rebalance_time = relativedelta(months=int(get_rebalance.name.split(".")[1]))

        # DETERMINE ON WHAT DATE THE NEXT REBALANCE FALLS ON
        next_rebalance = fund.last_rebalanced + rebalance_time

        # IF TRUE, WE NEED TO REBALANCE THIS PORTFOLIO TODAY
        if current_date.date() > next_rebalance.date():
            rebalance_list.append(portfolio)

    # IF THERE ARE NO PORTFOLIOS TO BE REBALANCED, THEN WE CAN SAVE THE TIME PULLING
    # UNNECESSARY DATA AND RETURN
    if not rebalance_list:
        logger.info("No portfolios were updated today")
        return "DONE"
    else:
        for portfolio in rebalance_list:
            optimization(portfolio, current_date, None)

            logger.info("The portfolio " + portfolio.name + " was successfully rebalanced today")

        return "DONE"

def send_daily_emails():
    day = datetime.now()
    users = User.objects.filter(email_preferences=0)
    portfolios = Portfolio.objects.all()

    email_list = []

    for user in [user for user in users if user.email_preferences == 0]:
        email_list.append(user.email)

    message = "TSIC EOD Portfolio Summary | " +  day.date().strftime('%Y-%m-%d')
    email = EmailMessage("TSIC EOD Portfolio Summary | " +  day.date().strftime('%Y-%m-%d'),
                         message,
                         EMAIL_HOST_USER,
                         email_list)

    for portfolio in portfolios:
        email.attach_file("website/static/documents/" + portfolio.fund.name + "_portfolio_stats.csv")

    email.send()

    logger.info("The daily email was sent")
    return "DONE"
