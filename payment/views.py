import stripe
import json
from django.conf import settings
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from django.utils.decorators import method_decorator
from django.views import View
import os

# Set Stripe API key
stripe.api_key = os.getenv('STRIPE_SECRET_KEY')

@method_decorator(csrf_exempt, name='dispatch')
class CreateCheckoutSessionView(View):
    def post(self, request):
        try:
            # Use your existing DRF Token authentication system
            from rest_framework.authtoken.models import Token
            from django.contrib.auth import get_user_model
            
            auth_header = request.headers.get('Authorization', '')
            if not auth_header.startswith('Token '):
                return JsonResponse({'error': 'Authentication required - please sign in'}, status=401)
            
            token_key = auth_header.replace('Token ', '')
            if not token_key:
                return JsonResponse({'error': 'Authentication required - please sign in'}, status=401)
            
            # Validate token against your existing DRF Token system
            try:
                token_obj = Token.objects.get(key=token_key)
                user = token_obj.user  # Get the actual CustomUser
            except Token.DoesNotExist:
                return JsonResponse({'error': 'Invalid token - please sign in again'}, status=401)
            
            data = json.loads(request.body)
            amount = data.get('amount', 699)  # Default $6.99 in cents
            
            # Create a Checkout Session
            checkout_session = stripe.checkout.Session.create(
                payment_method_types=['card'],
                line_items=[{
                    'price_data': {
                        'currency': 'usd',
                        'product_data': {
                            'name': 'NURTR Premium+ Subscription',
                            'description': 'Unlimited access to premium features including curated activities, unlimited child profiles, and personalized dashboard.',
                        },
                        'unit_amount': amount,
                        'recurring': {
                            'interval': 'month'
                        }
                    },
                    'quantity': 1,
                }],
                mode='subscription',
                success_url=data.get('success_url', 'http://localhost:5173/dashboard?payment=success'),
                cancel_url=data.get('cancel_url', 'http://localhost:5173/?payment=cancelled'),
                customer_email=user.email,  # Pre-fill customer email
                metadata={
                    'subscription_type': 'premium_plus',
                    'user_id': str(user.id),  # Store actual user ID
                    'user_email': user.email,  # Store user email for verification
                },
                allow_promotion_codes=True,
            )

            return JsonResponse({
                'checkout_url': checkout_session.url,
                'session_id': checkout_session.id
            })

        except Exception as e:
            return JsonResponse({'error': str(e)}, status=400)

@method_decorator(csrf_exempt, name='dispatch')
class StripeWebhookView(View):
    def post(self, request):
        payload = request.body
        sig_header = request.META.get('HTTP_STRIPE_SIGNATURE')
        endpoint_secret = os.getenv('STRIPE_WEBHOOK_SECRET')

        # Debug logging
        print(f"🔍 Webhook received:")
        print(f"  - Signature header: {sig_header[:50] if sig_header else 'MISSING'}...")
        print(f"  - Webhook secret: {endpoint_secret[:20] if endpoint_secret else 'MISSING'}...")
        print(f"  - Full webhook secret: {endpoint_secret}")
        print(f"  - Payload length: {len(payload)} bytes")

        try:
            event = stripe.Webhook.construct_event(
                payload, sig_header, endpoint_secret
            )
            print(f"✅ Webhook signature verified successfully")
        except ValueError as e:
            print(f"❌ Webhook ValueError: {e}")
            return JsonResponse({'error': 'Invalid payload'}, status=400)
        except stripe.error.SignatureVerificationError as e:
            print(f"❌ Webhook SignatureVerificationError: {e}")
            return JsonResponse({'error': 'Invalid signature'}, status=400)

        # Handle the event
        if event['type'] == 'checkout.session.completed':
            session = event['data']['object']
            self.handle_successful_checkout(session)
        elif event['type'] == 'invoice.payment_succeeded':
            invoice = event['data']['object']
            self.handle_subscription_payment(invoice)
        elif event['type'] == 'invoice.payment_failed':
            invoice = event['data']['object']
            self.handle_failed_payment(invoice)
        else:
            print(f'Unhandled event type: {event["type"]}')

        return JsonResponse({'status': 'success'})

    def handle_successful_checkout(self, session):
        """Handle successful checkout session - initial subscription"""
        from .models import UserSubscription
        from django.contrib.auth import get_user_model
        
        print(f"Checkout completed for session: {session['id']}")
        user_id = session.get('metadata', {}).get('user_id')
        user_email = session.get('metadata', {}).get('user_email')
        
        if user_id:
            try:
                # Get the actual user
                User = get_user_model()
                user = User.objects.get(id=user_id)
                
                # Create or update user subscription linked to real user
                subscription, created = UserSubscription.objects.get_or_create(
                    user=user,
                    defaults={
                        'stripe_session_id': session['id'],
                        'stripe_customer_id': session.get('customer', ''),
                        'stripe_subscription_id': session.get('subscription', ''),
                        'is_active': True,
                        'plan_type': 'premium_plus'
                    }
                )
                
                if not created:
                    # Update existing subscription
                    subscription.stripe_session_id = session['id']
                    subscription.stripe_customer_id = session.get('customer', '')
                    subscription.stripe_subscription_id = session.get('subscription', '')
                    subscription.is_active = True
                    subscription.plan_type = 'premium_plus'
                    subscription.save()
                
                print(f"✅ User subscription {'created' if created else 'updated'} to Premium+ for user: {user.email} (ID: {user.id})")
                
            except User.DoesNotExist:
                print(f"❌ Warning: User with ID {user_id} not found in database")
        else:
            print("❌ Warning: No user_id found in session metadata")

    def handle_subscription_payment(self, invoice):
        """Handle successful subscription payment"""
        print(f"Subscription payment succeeded for invoice: {invoice['id']}")

    def handle_failed_payment(self, invoice):
        """Handle failed payment"""
        print(f"Payment failed for invoice: {invoice['id']}")

@require_http_methods(["GET"])
def stripe_config(request):
    """Return Stripe publishable key"""
    return JsonResponse({
        'publishable_key': os.getenv('STRIPE_PUBLISHABLE_KEY')
    })

@require_http_methods(["GET"])
def subscription_status(request):
    """Check subscription status for authenticated user"""
    from .models import UserSubscription
    from rest_framework.authtoken.models import Token
    
    auth_header = request.headers.get('Authorization', '')
    if not auth_header.startswith('Token '):
        return JsonResponse({'error': 'Authentication required'}, status=401)
    
    token_key = auth_header.replace('Token ', '')
    if not token_key:
        return JsonResponse({'error': 'Authentication required'}, status=401)
    
    # Validate token and get user
    try:
        token_obj = Token.objects.get(key=token_key)
        user = token_obj.user
    except Token.DoesNotExist:
        return JsonResponse({'error': 'Invalid token'}, status=401)
    
    # Check subscription for this specific user
    try:
        subscription = UserSubscription.objects.get(user=user)
        access_type = subscription.get_access_type()
        has_premium_access = subscription.has_premium_access()
        
        response_data = {
            'is_premium': has_premium_access,
            'plan_type': subscription.plan_type,
            'is_active': subscription.is_active,
            'has_access_to_dashboard': has_premium_access,  # Required for dashboard access
            'access_type': access_type,  # 'paid_premium', 'trial', 'trial_expired', or 'free'
            'user_id': user.id,
            'user_email': user.email
        }
        
        # Add trial-specific information if user is in trial
        if subscription.is_trial_active:
            response_data.update({
                'is_trial': True,
                'trial_expired': subscription.is_trial_expired(),
                'trial_days_remaining': subscription.days_remaining_in_trial(),
                'trial_end_date': subscription.trial_end_date.isoformat() if subscription.trial_end_date else None
            })
        else:
            response_data.update({
                'is_trial': False,
                'trial_expired': False,
                'trial_days_remaining': 0,
                'trial_end_date': None
            })
        
        return JsonResponse(response_data)
        
    except UserSubscription.DoesNotExist:
        return JsonResponse({
            'is_premium': False,
            'plan_type': 'free',
            'is_active': False,
            'has_access_to_dashboard': False,
            'access_type': 'free',
            'is_trial': False,
            'trial_expired': False,
            'trial_days_remaining': 0,
            'trial_end_date': None,
            'user_id': user.id,
            'user_email': user.email
        })