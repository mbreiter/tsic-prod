# Generated by Django 2.0.7 on 2018-07-15 16:45

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('funds', '0005_weight_current_weight'),
    ]

    operations = [
        migrations.AddField(
            model_name='weight',
            name='quantity',
            field=models.FloatField(default=0),
        ),
    ]