from django.contrib import admin
from users.models import  Child


@admin.register(Child)
class ChildAdmin(admin.ModelAdmin):
    list_display = ('id', 'name', 'age', 'user')
    search_fields = ('name', 'user__email')
    list_filter = ('age',)
