# Generated by Django 5.0.4 on 2025-06-30 00:26

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('NurtrDjango', '0007_add_personalized_explanation'),
    ]

    operations = [
        migrations.CreateModel(
            name='EmailSubscription',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('email', models.EmailField(db_index=True, max_length=254, unique=True)),
                ('is_active', models.BooleanField(default=True)),
                ('brevo_contact_id', models.CharField(blank=True, help_text='Brevo contact ID for this subscriber', max_length=255, null=True)),
                ('source', models.CharField(default='website', help_text='Source of the subscription (website, api, etc.)', max_length=100)),
                ('subscribed_at', models.DateTimeField(auto_now_add=True)),
                ('unsubscribed_at', models.DateTimeField(blank=True, null=True)),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('updated_at', models.DateTimeField(auto_now=True)),
            ],
            options={
                'verbose_name': 'Email Subscription',
                'verbose_name_plural': 'Email Subscriptions',
                'ordering': ['-subscribed_at'],
            },
        ),
    ]
