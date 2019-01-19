from django.db import models
from django.utils import timezone as timez

class Rebalance(models.Model):
    name = models.CharField(max_length=250)
    id = models.AutoField(primary_key=True)
    strategy = models.TextField(null=True, blank=True)

    def __str__(self):
        return "%s" % self.name

    class Meta:
        ordering = ('name',)

class Optimization(models.Model):
    KEY_CHOICES = (
        (0, 'benchmark'),
        (1, 'mvo'),
        (2, 'blb'),
    )

    name = models.CharField(max_length=250)
    key = models.PositiveSmallIntegerField(choices=KEY_CHOICES,null=True, blank=True)
    id = models.AutoField(primary_key=True)
    strategy = models.TextField(null=True, blank=True)

    def __str__(self):
        return "%s" % self.name

    class Meta:
        ordering = ('name',)

class Fund(models.Model):

    OBJECTIVE_CHOICES = (
        (0, 'buy and hold'),
        (1, 'maximize returns'),
        (2, 'minimize volatility'),
        (3, 'maximize mean-CVaR tradeoff'),
    )

    name = models.CharField(max_length=250)
    id = models.AutoField(primary_key=True)
    started = models.DateTimeField()
    initial_capital = models.DecimalField(max_digits=8, decimal_places=2, default=1000)
    minimum_capital = models.DecimalField(max_digits=8, decimal_places=2, null=True, blank=True)
    fees = models.DecimalField(max_digits=8, decimal_places=2, null=True, blank=True)

    rebalance_method = models.ForeignKey(Rebalance, null=True, on_delete=models.SET_NULL)
    last_rebalanced = models.DateTimeField()
    payout_date = models.DateTimeField(null=True, blank=True)

    optimization_model = models.ForeignKey(Optimization, null=True, on_delete=models.SET_NULL)
    objective = models.PositiveSmallIntegerField(choices=OBJECTIVE_CHOICES,null=True, blank=True)

    def __str__(self):
        return "%s" % self.name

    class Meta:
        ordering = ('name',)
        verbose_name_plural = "funds"

class Portfolio(models.Model):
    name = models.CharField(max_length=250)
    id = models.AutoField(primary_key=True)
    fund = models.OneToOneField(Fund, on_delete=models.CASCADE)

    def __str__(self):
        return "%s" % self.name

    class Meta:
        ordering = ('name',)
        verbose_name_plural = "portfolios"

class PortfolioStatistics(models.Model):
    name = models.CharField(max_length=250)
    id = models.AutoField(primary_key=True)
    portfolio = models.ForeignKey(Portfolio, on_delete=models.CASCADE)
    date = models.DateTimeField()
    value = models.FloatField(default=0)

    trailing_30 = models.FloatField(default=0)
    trailing_120 = models.FloatField(default=0)

    rolling_30 = models.FloatField(default=0)
    rolling_120 = models.FloatField(default=0)

    returns_inception = models.FloatField(default=0)

    alpha = models.FloatField(default=0)
    beta = models.FloatField(default=0)

    sharpe_ratio = models.FloatField(default=0)

    def __str__(self):
        return "%s" % self.name

    class Meta:
        ordering = ('name',)
        verbose_name_plural = "Statistics"

class Asset(models.Model):

    ACTION_CHOICES = (
        (0, 'hold'),
        (1, 'buy'),
        (2, 'sell'),
    )

    name = models.CharField(max_length=250)
    ticker = models.CharField(max_length=10)
    action = models.PositiveSmallIntegerField(choices=ACTION_CHOICES, default=0)
    portfolio = models.ManyToManyField(Portfolio)
    date_added = models.DateTimeField(default=timez.now)

    analyst_input = models.BooleanField(default=True)
    coupled_asset = models.ForeignKey("self", unique=False, null=True, blank=True, on_delete=models.SET_NULL)

    value = models.FloatField(default=1, blank=True)
    value_low = models.FloatField(default=1, blank=True)
    value_high = models.FloatField(default=1, blank=True)

    def __str__(self):
        return "%s" % self.ticker

    class Meta:
        ordering = ('ticker',)
        verbose_name_plural = "assets"

class View(models.Model):

    ABSOLUTE_RELATIVE = (
        (0, 'absolute'),
        (1, 'relative'),
    )
    name = models.CharField(max_length=250)
    id = models.AutoField(primary_key=True)
    active = models.BooleanField(default=True)
    date_observed = models.DateTimeField(default=timez.now)

    portfolio = models.ForeignKey(Portfolio, null=True, on_delete=models.CASCADE)
    primary_asset = models.ForeignKey(Asset, on_delete=models.CASCADE,
                                             related_name="primary_asset")
    secondary_asset = models.ForeignKey(Asset, null=True, on_delete=models.CASCADE,
                                             related_name="secondary_asset")

    kind = models.PositiveSmallIntegerField(choices=ABSOLUTE_RELATIVE, default=0)

    value = models.FloatField(default=0)
    sensitivty = models.FloatField(default=0)

    def __str__(self):
        return "%s" % self.name

    class Meta:
        ordering = ('name',)
        verbose_name_plural = "Views"

class Weight(models.Model):
    asset = models.ForeignKey(Asset, on_delete=models.CASCADE)
    id = models.AutoField(primary_key=True)
    weight = models.FloatField(default=0)
    quantity = models.FloatField(default=0)
    portfolio = models.ForeignKey(Portfolio, on_delete=models.CASCADE)
    date_allocated = models.DateTimeField(null=True, blank=True)
    current_weight = models.BooleanField(default=True)

    def __str__(self):
        return "%s" % self.asset.ticker

    class Meta:
        ordering = ('asset',)
        verbose_name_plural = "weights"
