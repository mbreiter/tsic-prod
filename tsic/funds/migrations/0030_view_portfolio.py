# Generated by Django 2.0.7 on 2018-08-19 09:46

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('funds', '0029_auto_20180818_2130'),
    ]

    operations = [
        migrations.AddField(
            model_name='view',
            name='portfolio',
            field=models.ForeignKey(null=True, on_delete=django.db.models.deletion.CASCADE, to='funds.Portfolio'),
        ),
    ]
