# Generated by Django 2.0.7 on 2018-07-16 02:16

import datetime
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('funds', '0013_auto_20180715_2205'),
    ]

    operations = [
        migrations.AlterField(
            model_name='asset',
            name='date_added',
            field=models.DateTimeField(default=datetime.datetime(2018, 7, 15, 22, 16, 48, 759383)),
        ),
    ]
