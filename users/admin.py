# admin.py
from django.contrib import admin
from django.contrib.auth.admin import UserAdmin
from .models import CustomUser

class CustomUserAdmin(UserAdmin):
    model = CustomUser

    list_display = ("email", "name", "phone", "joining_date", "is_active", "is_staff")  # Show these columns in the list view
    search_fields = ("email", "name", "phone")  # Enable search by these fields
    ordering = ("-id",)

    fieldsets = (
        (None, {"fields": ("email", "password")}),
        ("Personal Info", {"fields": ("name", "phone", "joining_date")}),
        ("Permissions", {"fields": ("is_active", "is_staff", "is_superuser", "groups", "user_permissions")}),
    )

    add_fieldsets = (
        (None, {
            "classes": ("wide",),
            "fields": ("email", "password1", "password2", "name", "phone", "is_staff", "is_superuser"),
        }),
    )

    readonly_fields = ("joining_date",)  # Prevent editing the joining date

    filter_horizontal = ("groups", "user_permissions")  # Improves UI for permissions

admin.site.register(CustomUser, CustomUserAdmin)
