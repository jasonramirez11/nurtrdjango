# Generated by Django 5.0.4 on 2025-03-29 10:59

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('users', '0008_customuser_favorites'),
    ]

    operations = [
        migrations.AlterField(
            model_name='customuser',
            name='favorites',
            field=models.TextField(default='{}'),
        ),
    ]
