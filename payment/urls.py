from django.urls import path
from .views import CreateCheckoutSessionView, StripeWebhookView, stripe_config, subscription_status

urlpatterns = [
    path('create-checkout-session/', CreateCheckoutSessionView.as_view(), name='create_checkout_session'),
    path('stripe-webhook/', StripeWebhookView.as_view(), name='stripe_webhook'),
    path('stripe-config/', stripe_config, name='stripe_config'),
    path('subscription-status/', subscription_status, name='subscription_status'),
]