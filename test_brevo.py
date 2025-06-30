#!/usr/bin/env python3
"""
Test script to verify Brevo API connection and list setup.
Run this from the Django project root: python test_brevo.py
"""

import os
import django
from pathlib import Path

# Setup Django
BASE_DIR = Path(__file__).resolve().parent
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'NurtrDjango.settings')
django.setup()

from django.conf import settings
import sib_api_v3_sdk
from sib_api_v3_sdk.rest import ApiException

def test_brevo_connection():
    """Test Brevo API connection and list verification."""
    print("🔧 Testing Brevo API Connection...")
    print(f"API Key: {settings.BREVO_API_KEY[:20]}...")
    print(f"List ID: {settings.BREVO_LIST_ID}")
    
    try:
        # Configure Brevo API
        configuration = sib_api_v3_sdk.Configuration()
        configuration.api_key['api-key'] = settings.BREVO_API_KEY
        
        # Test 1: Get account info
        print("\n📊 Testing API connection...")
        account_api = sib_api_v3_sdk.AccountApi(sib_api_v3_sdk.ApiClient(configuration))
        account_info = account_api.get_account()
        print(f"✅ Connected to account: {account_info.email}")
        
        # Test 2: List all contact lists
        print("\n📋 Fetching all contact lists...")
        contacts_api = sib_api_v3_sdk.ContactsApi(sib_api_v3_sdk.ApiClient(configuration))
        lists = contacts_api.get_lists()
        
        print(f"Found {len(lists.lists)} contact lists:")
        for contact_list in lists.lists:
            print(f"  - ID: {contact_list.id}, Name: {contact_list.name}, Subscribers: {contact_list.unique_subscribers}")
        
        # Test 3: Verify target list exists
        target_list_id = int(settings.BREVO_LIST_ID)
        target_list = None
        for contact_list in lists.lists:
            if contact_list.id == target_list_id:
                target_list = contact_list
                break
        
        if target_list:
            print(f"\n✅ Target list found: '{target_list.name}' (ID: {target_list.id})")
            print(f"   Subscribers: {target_list.unique_subscribers}")
        else:
            print(f"\n❌ Target list ID {target_list_id} not found!")
            print("Available list IDs:", [lst.id for lst in lists.lists])
            return False
        
        # Test 4: Test adding a contact (dry run)
        test_email = "test@example.com"
        print(f"\n🧪 Testing contact creation with {test_email}...")
        
        create_contact = sib_api_v3_sdk.CreateContact(
            email=test_email,
            list_ids=[target_list_id]
        )
        
        try:
            response = contacts_api.create_contact(create_contact)
            print(f"✅ Test contact created successfully (ID: {response.id})")
            
            # Clean up - delete test contact
            try:
                contacts_api.delete_contact(test_email)
                print(f"🧹 Test contact cleaned up")
            except:
                print(f"⚠️  Could not clean up test contact (might not exist)")
                
        except ApiException as e:
            if e.status == 400:
                error_body = e.body
                if 'duplicate_parameter' in error_body:
                    print(f"✅ Contact already exists (this is expected)")
                else:
                    print(f"❌ API Error: {error_body}")
            else:
                print(f"❌ Unexpected API Error: {e}")
                return False
        
        print("\n🎉 All tests passed! Brevo integration is ready.")
        return True
        
    except Exception as e:
        print(f"\n❌ Connection failed: {e}")
        return False

if __name__ == "__main__":
    success = test_brevo_connection()
    if success:
        print("\n✅ You can now test the email subscription on your website!")
    else:
        print("\n❌ Please fix the issues above before testing the website.")