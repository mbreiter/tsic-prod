
import plotly.offline
import pandas as pd
import statsmodels.tsa.stattools as ts

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from django.contrib.auth import login as auth_login
from django.contrib.auth.models import Group
from django.http import HttpResponse, HttpResponseRedirect
from django.shortcuts import render, render_to_response
from django.template import RequestContext
from .forms import UserForm, UserEditForm
from .models import *
from django.contrib.auth.decorators import login_required
from tsic.settings import USER_TYPES, MEDIA_URL
from funds.views import *
from funds.models import *
from plotly.graph_objs import Scatter, Pie, Layout

def home(request):
    return render(request, "tsic/home.html")

def membership(request):
    return render(request, "tsic/membership.html")

@login_required
def performance(request):
    fund = Fund.objects.get(name="TSIC")
    fund_portfolio = Portfolio.objects.get(fund=fund.id)

    # optimization(fund_portfolio, None, None)

    # dynamically generate traces for the graph
    portfolios = Portfolio.objects.all()
    plot_data = []

    for portfolio in portfolios:
        if portfolio.name == "MVO":
            continue
        # file = "website/static/documents/" + portfolio.fund.name + "_portfolio_stats.csv"
        portfolio_stats = PortfolioStatistics.objects.filter(portfolio=portfolio)

        dates = []
        rebalance_dates = []
        values = []
        annotations = []

        oldest_asset = Asset.objects.filter(portfolio=portfolio).order_by('date_added')[0]

        rebalance_weights = Weight.objects.filter(asset=oldest_asset)

        for weight in rebalance_weights:
            rebalance_dates.append(weight.date_allocated.strftime('%Y-%m-%d'))

        for stat in portfolio_stats:
            dates.append(stat.date)
            values.append(stat.value)

            # IF YOU WANT TO ANNOTATE REBALANCES
            # if stat.date.strftime('%Y-%m-%d') in rebalance_dates:
            #     annotations.append(dict(x=stat.date.strftime('%Y-%m-%d'),
            #                             y=stat.value,
            #                             xref='x',
            #                             yref='y',
            #                             text='Rebalanced',
            #                             showarrow=True))

        trace = plotly.graph_objs.Scatter(x = dates,
                                          y = values,
                                          mode = 'lines',
                                          name = portfolio.fund.name)

        plot_data.append(trace)

    plot = plotly.offline.plot({"data": plot_data,
                                "layout": Layout(annotations=annotations)},
                                output_type='div',
                                include_plotlyjs=False,
                                show_link=False,
                                config={"displayModeBar": False})

    tsic_fund = Fund.objects.get(name="TSIC")
    tsic = Portfolio.objects.get(fund=tsic_fund)

    get_weights = Weight.objects.filter(portfolio = tsic, current_weight=True)
    weights, labels = [], []

    for weight in get_weights:
        labels.append(weight.asset.ticker)
        weights.append(weight.weight)

    pie = plotly.offline.plot({"data": [Pie(labels=labels, values=weights)]},
                               # "layout": Layout(title="Portfolio Asset Breakdown")
                               output_type='div',
                               include_plotlyjs=False,
                               show_link=False,
                               config={"displayModeBar": False})

    return render(request, "tsic/funds.html", {'MEDIA_URL': MEDIA_URL, 'plot': plot, 'pie': pie})

@login_required
def research(request):
    get_categories = ReportCategory.objects.all().exclude(name="Featured")
    get_featured = ReportCategory.objects.get(name="Featured")

    categories = {}
    for category in get_categories:
        categories[category] = Report.objects.filter(category=category)

    featured = {}
    featured[get_featured] = Report.objects.filter(category=get_featured)

    context = {'featured':featured, 'categories': categories}

    return render(request, "tsic/research.html", context)

@login_required
def about(request):
    return render(request, "tsic/about.html")

@login_required
def edit_about(request):

    if request.method == "POST":
        edit_form = UserEditForm(data=request.POST, instance=request.user)

        if edit_form.is_valid():
            edit_form.save()
            return HttpResponseRedirect('/about')

        else:
            print(edit_form.errors)

    else:
        edit_form = UserEditForm()

    context = {'form': edit_form, 'preferences': User.EMAIL_PREFERENCES}

    return render(request, "tsic/about_edit.html", context)

@login_required
def tools(request):
    return render(request, "tsic/tools.html")

def register(request):
    registered = False

    if request.method == "POST":
        user_form = UserForm(data=request.POST)

        if user_form.is_valid():
            registered = True

            user = user_form.save()
            user.set_password(user.password)

            if user_form.staff_status == True:
                user.is_staff = True

            user.save()

            group = Group.objects.get(name=USER_TYPES[user_form.user_type])
            group.user_set.add(user)
            group.save()

            auth_login(request, user)
            return HttpResponseRedirect('/home')

        else:
            print(user_form.errors)

    else:
        user_form = UserForm()

    context = {'form': user_form, 'registered': registered, 'MEDIA_URL': MEDIA_URL}

    return render(request, "registration/register.html", context)

def error_404(request, exception, template_name='404.html'):
    data = {}
    return render(request, '404.html', data)

def error_500(request, exception, template_name='500.html'):
    data = {}
    return render(request, '500.html', data)
