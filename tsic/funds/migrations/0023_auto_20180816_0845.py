# Generated by Django 2.0.7 on 2018-08-16 12:45

import datetime
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('funds', '0022_auto_20180813_1714'),
    ]

    operations = [
        migrations.AddField(
            model_name='fund',
            name='initial_capital',
            field=models.DecimalField(decimal_places=2, default=1000, max_digits=8),
        ),
        migrations.AlterField(
            model_name='asset',
            name='date_added',
            field=models.DateTimeField(default=datetime.datetime(2018, 8, 16, 8, 45, 29, 130602)),
        ),
        migrations.AlterField(
            model_name='fund',
            name='fees',
            field=models.DecimalField(blank=True, decimal_places=2, max_digits=8, null=True),
        ),
        migrations.AlterField(
            model_name='fund',
            name='minimum_capital',
            field=models.DecimalField(blank=True, decimal_places=2, max_digits=8, null=True),
        ),
    ]