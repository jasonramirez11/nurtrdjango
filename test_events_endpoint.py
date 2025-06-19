#!/usr/bin/env python3
"""
Simple test script for the existing RecommendedEventsAPIView endpoint

This script tests the /api/events/recommended/ endpoint that's already configured.
"""

import requests
import json
import time

# Configuration
BASE_URL = "http://localhost:8000"
ENDPOINT = "/api/events/recommended/"

def test_endpoint():
    """Test the recommended events endpoint."""
    
    print("🧪 Testing RecommendedEventsAPIView")
    print("=" * 50)
    
    # Test 1: GET without authentication (should fail)
    print("\n1. Testing GET without authentication...")
    try:
        response = requests.get(f"{BASE_URL}{ENDPOINT}")
        print(f"   Status: {response.status_code}")
        print(f"   Response: {response.text[:200]}...")
        
        if response.status_code == 401:
            print("   ✅ Correctly requires authentication")
        else:
            print("   ❓ Unexpected response for unauthenticated request")
    except requests.exceptions.RequestException as e:
        print(f"   ❌ Request failed: {e}")
        return
    
    # Test 2: GET with dummy token (should fail)
    print("\n2. Testing GET with invalid token...")
    headers = {"Authorization": "Token dummy_token_12345"}
    try:
        response = requests.get(f"{BASE_URL}{ENDPOINT}", headers=headers)
        print(f"   Status: {response.status_code}")
        print(f"   Response: {response.text[:200]}...")
        
        if response.status_code == 401:
            print("   ✅ Correctly rejects invalid token")
        else:
            print("   ❓ Unexpected response for invalid token")
    except requests.exceptions.RequestException as e:
        print(f"   ❌ Request failed: {e}")
    
    # Test 3: POST without authentication (should fail)
    print("\n3. Testing POST without authentication...")
    post_data = {"zip_code": "90210", "radius": 25}
    try:
        response = requests.post(f"{BASE_URL}{ENDPOINT}", json=post_data)
        print(f"   Status: {response.status_code}")
        print(f"   Response: {response.text[:200]}...")
        
        if response.status_code == 401:
            print("   ✅ POST correctly requires authentication")
        else:
            print("   ❓ Unexpected response for unauthenticated POST")
    except requests.exceptions.RequestException as e:
        print(f"   ❌ POST Request failed: {e}")
    
    # Test 4: POST with dummy token (should fail)
    print("\n4. Testing POST with invalid token...")
    try:
        response = requests.post(f"{BASE_URL}{ENDPOINT}", json=post_data, headers=headers)
        print(f"   Status: {response.status_code}")
        print(f"   Response: {response.text[:200]}...")
        
        if response.status_code == 401:
            print("   ✅ POST correctly rejects invalid token")
        else:
            print("   ❓ Unexpected response for invalid POST token")
    except requests.exceptions.RequestException as e:
        print(f"   ❌ POST Request failed: {e}")
    
    print("\n5. Authenticated testing:")
    print("   • Will test both GET and POST methods with real users")
    print("   • POST generates personalized event recommendations")
    print("   • GET retrieves cached recommendations")
    
    print("\n" + "=" * 50)
    print("Basic endpoint tests completed!")

def test_with_token(token):
    """Test the endpoint with a real authentication token."""
    
    print(f"\n🔐 Testing with authentication token...")
    
    headers = {"Authorization": f"Token {token}"}
    
    # Test GET method first
    print("\n📋 Testing GET /api/events/recommended/ (retrieve cached)...")
    try:
        response = requests.get(f"{BASE_URL}{ENDPOINT}", headers=headers)
        print(f"GET Status: {response.status_code}")
        
        if response.status_code == 200:
            try:
                data = response.json()
                print("✅ GET Success! Response structure:")
                print(f"   Keys: {list(data.keys())}")
                
                recommendations = data.get("recommendations", [])
                print(f"   Number of cached recommendations: {len(recommendations)}")
                
                if recommendations:
                    print("   Sample cached recommendation:")
                    sample = recommendations[0]
                    for key, value in list(sample.items())[:5]:  # Show first 5 fields
                        print(f"     {key}: {value}")
                else:
                    print("   No cached recommendations found")
                    
            except json.JSONDecodeError:
                print("   ❌ GET Response is not valid JSON")
                print(f"   Raw response: {response.text}")
        else:
            print(f"❌ GET Failed with status {response.status_code}")
            print(f"Response: {response.text}")
            
    except requests.exceptions.RequestException as e:
        print(f"❌ GET Request failed: {e}")
    
    # Test POST method
    print("\n🚀 Testing POST /api/events/recommended/ (generate new)...")
    post_data = {
        "zip_code": "90210",  # Beverly Hills
        "radius": 10000,         # 25 miles
        "date_range": "this_week",
        "event_types": ["family", "kids"],
        "force_refresh": True  # Force fresh generation
    }
    
    try:
        response = requests.post(f"{BASE_URL}{ENDPOINT}", json=post_data, headers=headers)
        print(f"POST Status: {response.status_code}")
        
        if response.status_code == 200:
            try:
                data = response.json()
                print("✅ POST Success! Response structure:")
                print(f"   Keys: {list(data.keys())}")
                
                results = data.get("results", [])
                print(f"   Number of new recommendations: {len(results)}")
                print(f"   Execution time: {data.get('execution_time_ms', 'N/A')}ms")
                print(f"   Cached: {data.get('cached', False)}")
                print(f"   Total results: {data.get('total_results', 0)}")
                
                if 'custom_queries' in data:
                    print(f"   Generated queries: {data['custom_queries']}")
                
                if results:
                    print("   Top 3 new recommendations:")
                    for i, event in enumerate(results[:3], 1):
                        title = event.get('title', 'Unknown')
                        score = event.get('personalization_score', 0)
                        event_type = event.get('event_type', 'Unknown')
                        venue = event.get('venue_name', 'Unknown venue')
                        print(f"     {i}. {title} (Score: {score}, Type: {event_type})")
                        print(f"        Venue: {venue}")
                    
            except json.JSONDecodeError:
                print("   ❌ POST Response is not valid JSON")
                print(f"   Raw response: {response.text}")
        else:
            print(f"❌ POST Failed with status {response.status_code}")
            print(f"Response: {response.text}")
            
    except requests.exceptions.RequestException as e:
        print(f"❌ POST Request failed: {e}")
    
    input("Press Enter to continue...")
    
    # Test POST with cache (should be faster)
    print("\n⚡ Testing POST /api/events/recommended/ (use cache)...")
    post_data_cached = post_data.copy()
    post_data_cached["force_refresh"] = False  # Use cache if available
    
    try:
        response = requests.post(f"{BASE_URL}{ENDPOINT}", json=post_data_cached, headers=headers)
        print(f"Cached POST Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"   Cached: {data.get('cached', False)}")
            print(f"   Execution time: {data.get('execution_time_ms', 'N/A')}ms")
            results = data.get("results", [])
            print(f"   Number of recommendations: {len(results)}")
                    
    except requests.exceptions.RequestException as e:
        print(f"❌ Cached POST Request failed: {e}")

def test_post_endpoint_detailed(token):
    """Dedicated detailed test for the POST endpoint functionality."""
    
    print("\n" + "="*60)
    print("🚀 DETAILED POST ENDPOINT TESTING")
    print("="*60)
    
    headers = {"Authorization": f"Token {token}"}
    
    # Test different POST scenarios
    test_scenarios = [
        {
            "name": "Beverly Hills Family Events",
            "data": {
                "zip_code": "90210",
                "radius": 20,
                "date_range": "this_month",
                "event_types": ["family", "kids"],
                "force_refresh": True
            }
        },
        {
            "name": "NYC Educational Events",
            "data": {
                "latitude": 40.7128,
                "longitude": -74.0060,
                "radius": 15,
                "date_range": "this_week",
                "event_types": ["educational", "arts"],
                "force_refresh": True
            }
        },
        {
            "name": "Cached Request Test",
            "data": {
                "zip_code": "90210",
                "radius": 20,
                "date_range": "this_month",
                "force_refresh": False  # Should use cache
            }
        }
    ]
    
    for i, scenario in enumerate(test_scenarios, 1):
        print(f"\n🧪 Scenario {i}: {scenario['name']}")
        print("-" * 40)
        
        try:
            start_time = time.time()
            response = requests.post(f"{BASE_URL}{ENDPOINT}", json=scenario['data'], headers=headers)
            end_time = time.time()
            
            print(f"Status: {response.status_code}")
            print(f"Request time: {(end_time - start_time)*1000:.1f}ms")
            
            if response.status_code == 200:
                data = response.json()
                
                # Basic response info
                print(f"✅ Success!")
                print(f"   Cached: {data.get('cached', False)}")
                print(f"   Server execution time: {data.get('execution_time_ms', 'N/A')}ms")
                print(f"   Total results found: {data.get('total_results', 0)}")
                
                # Results details
                results = data.get("results", [])
                print(f"   Returned results: {len(results)}")
                
                # Show generated queries (if any)
                if 'custom_queries' in data and data['custom_queries']:
                    print(f"   Generated queries: {data['custom_queries']}")
                
                # Show user profile (if any)
                if 'user_profile' in data:
                    profile = data['user_profile']
                    print(f"   User profile: {profile.get('total_children', 0)} children")
                    if profile.get('age_ranges'):
                        print(f"   Age ranges: {profile['age_ranges']}")
                    if profile.get('all_interests'):
                        print(f"   Interests: {profile['all_interests']}")
                
                # Show top recommendations
                if results:
                    print(f"   Top recommendations:")
                    for j, event in enumerate(results[:3], 1):
                        title = event.get('title', 'Unknown')
                        score = event.get('personalization_score', 0)
                        event_type = event.get('event_type', 'Unknown')
                        when = event.get('when', 'Unknown time')
                        print(f"     {j}. {title}")
                        print(f"        Score: {score}, Type: {event_type}")
                        print(f"        When: {when}")
                
            else:
                print(f"❌ Failed with status {response.status_code}")
                try:
                    error_data = response.json()
                    print(f"   Error: {error_data.get('error', 'Unknown error')}")
                except:
                    print(f"   Raw response: {response.text[:200]}...")
                    
        except requests.exceptions.RequestException as e:
            print(f"❌ Request failed: {e}")
        
        if i < len(test_scenarios):
            print("\n⏳ Waiting 2 seconds before next test...")
            time.sleep(2)
    
    print("\n" + "="*60)
    print("🏁 POST ENDPOINT TESTING COMPLETE")
    print("="*60)

def get_auth_token(email, password):
    """Get authentication token for a user."""
    try:
        auth_data = {"email": email, "password": password}
        response = requests.post(f"{BASE_URL}/api/users/authenticate/", json=auth_data)
        
        if response.status_code == 200:
            return response.json().get('token')
        else:
            print(f"   ❌ Authentication failed for {email}: {response.status_code}")
            return None
    except Exception as e:
        print(f"   ❌ Authentication error for {email}: {e}")
        return None

def test_with_existing_users():
    """Test with existing users from test_recommendations.py."""
    
    # Test users from test_recommendations.py
    test_users = [
        {"email": "jason@newtest.com", "password": "test123"},
        #{"email": "test@example.com", "password": "testpass123"}
    ]
    
    print("\n🔐 Testing with existing users...")
    
    for user_info in test_users:
        email = user_info["email"]
        password = user_info["password"]
        
        print(f"\n👤 Trying user: {email}")
        
        # Get token
        token = get_auth_token(email, password)
        
        if token:
            print(f"   ✅ Authentication successful!")
            print(f"   🎫 Token: {token[:20]}...")
            
            # Test the events endpoint
            test_with_token(token)
            
            # Run detailed POST endpoint testing
            test_post_endpoint_detailed(token)
            return True
        else:
            print(f"   ❌ Failed to authenticate {email}")
    
    print("\n❌ No existing users could be authenticated.")
    print("You may need to create a test user first.")
    return False

if __name__ == "__main__":
    print("RecommendedEventsAPIView Endpoint Test")
    
    # Check if server is running
    try:
        response = requests.get(f"{BASE_URL}/api/", timeout=5)
        print("✅ Server is running")
    except requests.exceptions.RequestException:
        print("❌ Server not running. Start with: python manage.py runserver")
        exit(1)
    
    # Run basic tests
    test_endpoint()
    
    # Test with existing users automatically
    print("\n" + "=" * 50)
    print("🔄 Trying to authenticate with existing users...")
    
    success = test_with_existing_users()
    
    if not success:
        # Fallback to manual token input
        print("\n" + "=" * 50)
        token_input = input("Enter an auth token to test manually (or press Enter to skip): ").strip()
        if token_input:
            test_with_token(token_input)
    
    print("\n🎉 Testing complete!") 