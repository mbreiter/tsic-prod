# Generated by Django 2.0.7 on 2018-08-27 00:56

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('funds', '0030_view_portfolio'),
    ]

    operations = [
        migrations.AddField(
            model_name='portfoliostatistics',
            name='alpha',
            field=models.FloatField(default=0),
        ),
        migrations.AddField(
            model_name='portfoliostatistics',
            name='beta',
            field=models.FloatField(default=0),
        ),
        migrations.AddField(
            model_name='portfoliostatistics',
            name='returns_inception',
            field=models.FloatField(default=0),
        ),
        migrations.AddField(
            model_name='portfoliostatistics',
            name='rolling_120',
            field=models.FloatField(default=0),
        ),
        migrations.AddField(
            model_name='portfoliostatistics',
            name='rolling_30',
            field=models.FloatField(default=0),
        ),
        migrations.AddField(
            model_name='portfoliostatistics',
            name='trailing_120',
            field=models.FloatField(default=0),
        ),
        migrations.AddField(
            model_name='portfoliostatistics',
            name='trailing_30',
            field=models.FloatField(default=0),
        ),
    ]