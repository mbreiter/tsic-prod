# Generated by Django 2.0.7 on 2018-07-19 01:59

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('website', '0002_auto_20180718_2156'),
    ]

    operations = [
        migrations.AddField(
            model_name='user',
            name='email_preferences',
            field=models.PositiveSmallIntegerField(blank=True, choices=[(0, 'daily'), (1, 'monthly'), (2, 'semesterly'), (3, 'never')], default=0, null=True),
        ),
        migrations.AlterField(
            model_name='user',
            name='user_type',
            field=models.PositiveSmallIntegerField(blank=True, choices=[(1, 'analyst'), (2, 'associate'), (3, 'quant')], null=True),
        ),
    ]