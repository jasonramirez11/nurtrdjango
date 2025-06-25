from django.db import models
from django.conf import settings

class UserSubscription(models.Model):
    PLAN_CHOICES = [
        ('free', 'Free'),
        ('premium_plus', 'Premium+'),
    ]
    
    user = models.OneToOneField(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name='subscription', null=True, blank=True)
    auth_token = models.TextField(blank=True)  # Store auth token for user identification
    stripe_customer_id = models.CharField(max_length=255, blank=True)
    stripe_subscription_id = models.CharField(max_length=255, blank=True)
    stripe_session_id = models.CharField(max_length=255, blank=True)
    is_active = models.BooleanField(default=False)
    plan_type = models.CharField(max_length=20, choices=PLAN_CHOICES, default='free')
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    def __str__(self):
        user_info = f"User {self.user.id}" if self.user else f"Token {self.auth_token[:20]}..."
        return f"{user_info} - {self.plan_type} ({'Active' if self.is_active else 'Inactive'})"
    
    def is_premium(self):
        return self.is_active and self.plan_type == 'premium_plus'
    
    class Meta:
        db_table = 'user_subscriptions'