from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('users', '0004_alter_customuser_managers_child'),
    ]

    operations = [
        migrations.AddField(
            model_name='child',
            name='interests',
            field=models.JSONField(default=list),
        ),
        migrations.RemoveField(
            model_name='child',
            name='outdoor_activities',
        ),
        migrations.RemoveField(
            model_name='child',
            name='indoor_activities',
        ),
    ]