#!/usr/bin/env python3
"""
Simple test script for the RecommendedPlacesAPIView endpoint

This script tests the /api/places/recommended/ endpoint.
"""

import requests
import json
import time

# Configuration
BASE_URL = "http://localhost:8000"
ENDPOINT = "/api/places/recommended/"

def test_endpoint():
    """Test the recommended places endpoint."""
    
    print("🧪 Testing RecommendedPlacesAPIView")
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
    
    # Test 2: POST without authentication (should fail)
    print("\n2. Testing POST without authentication...")
    post_data = {"latitude": 34.0522, "longitude": -118.2437, "radius": 25}
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
    
    print("\n" + "=" * 50)
    print("Basic endpoint tests completed!")

def test_with_token(token):
    """Test the endpoint with a real authentication token."""
    
    print(f"\n🔐 Testing with authentication token...")
    
    headers = {"Authorization": f"Token {token}"}
    
    # Test GET method first
    print("\n📋 Testing GET /api/places/recommended/ (retrieve cached)...")
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
                    print(f"     Title: {sample.get('title', 'N/A')}")
                    print(f"     Score: {sample.get('personalization_score', 'N/A')}")
                    print(f"     Category: {sample.get('category', 'N/A')}")
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
    print("\n🚀 Testing POST /api/places/recommended/ (generate new)...")
    post_data = {
        #"latitude": 34.0522,   # Los Angeles
        #"longitude": -118.2437,
        "radius": 20000,          # 25 km radius
        "zip_code": "91001",      # New York
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
                print(f"   Total results: {data.get('total_results', 0)}")
                
                if 'custom_queries' in data:
                    print(f"   Generated queries: {data['custom_queries']}")
                
                if results:
                    print("   Top 3 new recommendations:")
                    for i, place in enumerate(results[:3], 1):
                        title = place.get('title', 'Unknown')
                        score = place.get('personalization_score', 0)
                        explanation = place.get('personalized_explanation', '')
                        category = place.get('category', 'Unknown')
                        rating = place.get('rating', 'N/A')
                        print(f"     {i}. {title} (Score: {score}, Rating: {rating})")
                        print(f"        Category: {category}")
                        if explanation:
                            print(f"        Explanation: {explanation[:100]}...")
                    
            except json.JSONDecodeError:
                print("   ❌ POST Response is not valid JSON")
                print(f"   Raw response: {response.text}")
        else:
            print(f"❌ POST Failed with status {response.status_code}")
            print(f"Response: {response.text}")
            
    except requests.exceptions.RequestException as e:
        print(f"❌ POST Request failed: {e}")

def test_post_endpoint_detailed(token):
    """Detailed test for the POST endpoint functionality."""
    
    print("\n" + "="*60)
    print("🚀 DETAILED POST ENDPOINT TESTING")
    print("="*60)
    
    headers = {"Authorization": f"Token {token}"}
    
    # Test different POST scenarios
    test_scenarios = [
        {
            "name": "Los Angeles Family Places",
            "data": {
                "latitude": 34.0522,
                "longitude": -118.2437,
                "radius": 20,
                "force_refresh": True
            }
        },
        {
            "name": "Beverly Hills Area",
            "data": {
                "latitude": 34.0736,
                "longitude": -118.4004,
                "radius": 15,
                "force_refresh": True
            }
        },
        {
            "name": "Cached Request Test",
            "data": {
                "latitude": 34.0522,
                "longitude": -118.2437,
                "radius": 20,
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
                    print(f"   User profile: {profile}")
                
                # Show top recommendations with scores and explanations
                if results:
                    print(f"   Top recommendations:")
                    for j, place in enumerate(results[:3], 1):
                        title = place.get('title', 'Unknown')
                        score = place.get('personalization_score', 0)
                        explanation = place.get('personalized_explanation', '')
                        category = place.get('category', 'Unknown')
                        rating = place.get('rating', 'N/A')
                        print(f"     {j}. {title}")
                        print(f"        Score: {score}, Rating: {rating}, Category: {category}")
                        if explanation:
                            print(f"        AI Explanation: {explanation}")
                
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
            print(f"   Response: {response.text}")
            return None
    except Exception as e:
        print(f"   ❌ Authentication error for {email}: {e}")
        return None

def test_with_existing_user():
    """Test with the specified user."""
    
    print("\n🔐 Testing with jason@newtest.com...")
    
    # Test user
    email = "jason@newtest.com"
    password = "test123"
    
    print(f"\n👤 Trying user: {email}")
    
    # Get token
    token = get_auth_token(email, password)
    
    if token:
        print(f"   ✅ Authentication successful!")
        print(f"   🎫 Token: {token[:20]}...")
        
        # Test the places endpoint
        test_with_token(token)
        
        # Run detailed POST endpoint testing
        test_post_endpoint_detailed(token)
        return True
    else:
        print(f"   ❌ Failed to authenticate {email}")
        return False

if __name__ == "__main__":
    print("RecommendedPlacesAPIView Endpoint Test")
    
    # Check if server is running
    try:
        response = requests.get(f"{BASE_URL}/api/", timeout=5)
        print("✅ Server is running")
    except requests.exceptions.RequestException:
        print("❌ Server not running. Start with: python manage.py runserver")
        exit(1)
    
    # Run basic tests
    test_endpoint()
    
    # Test with the specified user
    print("\n" + "=" * 50)
    print("🔄 Testing with jason@newtest.com...")
    
    success = test_with_existing_user()
    
    if not success:
        # Fallback to manual token input
        print("\n" + "=" * 50)
        token_input = input("Enter an auth token to test manually (or press Enter to skip): ").strip()
        if token_input:
            test_with_token(token_input)
            test_post_endpoint_detailed(token_input)
    
    print("\n🎉 Testing complete!")