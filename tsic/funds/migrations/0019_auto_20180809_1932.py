# Generated by Django 2.0.7 on 2018-08-09 23:32

import datetime
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('funds', '0018_auto_20180804_1600'),
    ]

    operations = [
        migrations.AddField(
            model_name='asset',
            name='analyst_input',
            field=models.BooleanField(default=True),
        ),
        migrations.AlterField(
            model_name='asset',
            name='date_added',
            field=models.DateTimeField(default=datetime.datetime(2018, 8, 9, 19, 32, 14, 321811)),
        ),
        migrations.AlterField(
            model_name='fund',
            name='objective',
            field=models.PositiveSmallIntegerField(blank=True, choices=[(0, 'buy and hold'), (1, 'maximize returns'), (2, 'minimize volatility'), (3, 'maximize mean-CVaR tradeoff')], null=True),
        ),
        migrations.AlterField(
            model_name='optimization',
            name='key',
            field=models.PositiveSmallIntegerField(blank=True, choices=[(0, 'benchmark'), (1, 'mvo'), (2, 'blb')], null=True),
        ),
    ]