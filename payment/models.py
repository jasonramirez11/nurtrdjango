from django.db import models
from django.conf import settings
from django.utils import timezone

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
    
    # Free trial fields
    trial_start_date = models.DateTimeField(null=True, blank=True, help_text="When the free trial started")
    trial_end_date = models.DateTimeField(null=True, blank=True, help_text="When the free trial ends")
    is_trial_active = models.BooleanField(default=False, help_text="Whether user is currently in free trial")
    
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    def __str__(self):
        user_info = f"User {self.user.id}" if self.user else f"Token {self.auth_token[:20]}..."
        trial_status = " (Trial)" if self.is_trial_active and not self.is_trial_expired() else ""
        return f"{user_info} - {self.plan_type} ({'Active' if self.is_active else 'Inactive'}){trial_status}"
    
    def is_trial_expired(self):
        """Check if the trial period has ended"""
        if not self.trial_end_date or not self.is_trial_active:
            return False
        return timezone.now() > self.trial_end_date
    
    def days_remaining_in_trial(self):
        """Get number of days remaining in trial"""
        if not self.trial_end_date or not self.is_trial_active or self.is_trial_expired():
            return 0
        delta = self.trial_end_date - timezone.now()
        return max(0, delta.days)
    
    def is_premium(self):
        """Legacy method - kept for backward compatibility"""
        return self.has_premium_access()
    
    def has_premium_access(self):
        """Check if user has premium access (either paid subscription or active trial)"""
        # Paid premium subscription
        if self.is_active and self.plan_type == 'premium_plus':
            return True
        
        # Active trial that hasn't expired
        if self.is_trial_active and not self.is_trial_expired():
            return True
        
        return False
    
    def get_access_type(self):
        """Return the type of access the user currently has"""
        if self.is_active and self.plan_type == 'premium_plus':
            return 'paid_premium'
        elif self.is_trial_active and not self.is_trial_expired():
            return 'trial'
        elif self.is_trial_active and self.is_trial_expired():
            return 'trial_expired'
        else:
            return 'free'
    
    class Meta:
        db_table = 'user_subscriptions'