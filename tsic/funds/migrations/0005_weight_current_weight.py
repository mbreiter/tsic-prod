# Generated by Django 2.0.7 on 2018-07-15 12:42

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('funds', '0004_weight_date_allocated'),
    ]

    operations = [
        migrations.AddField(
            model_name='weight',
            name='current_weight',
            field=models.BooleanField(default=True),
        ),
    ]
