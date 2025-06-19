from django.contrib.auth.models import AbstractBaseUser, BaseUserManager, PermissionsMixin ,UserManager
from django.db import models
from django.contrib.auth.password_validation import validate_password
from django.core.exceptions import ValidationError

class UserManager(BaseUserManager):
    def create_user(self, email, password=None, **extra_fields):
        if not email:
            raise ValueError('The Email field must be set')
        email = self.normalize_email(email)
        user = self.model(email=email, **extra_fields)
        user.set_password(password)
        user.save(using=self._db)
        return user

    def create_superuser(self, email, password=None, **extra_fields):
        extra_fields.setdefault('is_staff', True)
        extra_fields.setdefault('is_superuser', True)

        # Ensure that email is set and the superuser has these flags
        if extra_fields.get('is_staff') is not True:
            raise ValueError('Superuser must have is_staff=True.')
        if extra_fields.get('is_superuser') is not True:
            raise ValueError('Superuser must have is_superuser=True.')

        return self.create_user(email, password, **extra_fields)

class CustomUser(AbstractBaseUser, PermissionsMixin):
    email = models.EmailField(unique=True)
    name = models.CharField(max_length=255, blank=True, null=True)
    phone = models.CharField(max_length=20, unique=True, blank=True, null=True)  # Set phone as optional
    joining_date = models.DateTimeField(auto_now_add=True)
    is_active = models.BooleanField(default=True)
    is_staff = models.BooleanField(default=False)
    favorites = models.TextField(default='{}')  # Store favorites as JSON string to preserve exact IDs
    
    # Location-based recommendation fields
    homebaseZipCode = models.CharField(max_length=10, blank=True, null=True, help_text="User's primary/home zip code for personalized recommendations")
    location_preferences = models.JSONField(default=list, help_text="User's preferred locations for recommendations (zip codes)")
    search_history_locations = models.JSONField(default=list, help_text="Historical locations user has searched (zip codes)")

    objects = UserManager()
    USERNAME_FIELD = 'email'

    def __str__(self):
        return f"{self.name} ({self.email}) - Joined on {self.joining_date.strftime('%Y-%m-%d')}"

class Child(models.Model):
    user = models.ForeignKey(CustomUser, on_delete=models.CASCADE, related_name='children')
    name = models.CharField(max_length=255)
    age = models.IntegerField()
    interests = models.JSONField(default=list)  # Replaced outdoor_activities and indoor_activities

    def __str__(self):
        return self.name