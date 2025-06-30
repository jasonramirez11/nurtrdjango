#!/usr/bin/env python3
"""
Debug script to test Brevo API configuration
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

print("🔧 Environment Variables Check:")
print(f"BREVO_API_KEY loaded: {os.getenv('BREVO_API_KEY')[:20]}..." if os.getenv('BREVO_API_KEY') else "NOT FOUND")
print(f"BREVO_LIST_ID: {os.getenv('BREVO_LIST_ID', 'NOT SET')}")

# Test the API key directly
try:
    import sib_api_v3_sdk
    from sib_api_v3_sdk.rest import ApiException
    
    # Configure API
    configuration = sib_api_v3_sdk.Configuration()
    configuration.api_key['api-key'] = os.getenv('BREVO_API_KEY')
    
    print(f"\n🔑 Testing API Key: {configuration.api_key['api-key'][:20]}...")
    
    # Test account info
    account_api = sib_api_v3_sdk.AccountApi(sib_api_v3_sdk.ApiClient(configuration))
    account_info = account_api.get_account()
    
    print(f"✅ API Key is VALID!")
    print(f"Account Email: {account_info.email}")
    print(f"Company Name: {account_info.company_name}")
    
    # Test lists
    contacts_api = sib_api_v3_sdk.ContactsApi(sib_api_v3_sdk.ApiClient(configuration))
    lists = contacts_api.get_lists()
    
    print(f"\n📋 Available Lists:")
    for contact_list in lists.lists:
        print(f"  - ID: {contact_list.id}, Name: {contact_list.name}")
        
    target_list_id = int(os.getenv('BREVO_LIST_ID', '5'))
    target_list = next((lst for lst in lists.lists if lst.id == target_list_id), None)
    
    if target_list:
        print(f"\n✅ Target list found: '{target_list.name}' (ID: {target_list.id})")
    else:
        print(f"\n❌ Target list ID {target_list_id} not found!")
    
except ApiException as e:
    print(f"\n❌ Brevo API Error: {e}")
    print(f"Status: {e.status}")
    print(f"Reason: {e.reason}")
    print(f"Body: {e.body}")
    
except Exception as e:
    print(f"\n❌ General Error: {e}")
    
print("\n" + "="*50)