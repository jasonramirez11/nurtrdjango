import json
import os
import uuid
import django.views.generic
from django.contrib.auth import get_user_model
from django.contrib.staticfiles import finders
from django.core.files.base import ContentFile
from django.core.mail import send_mail, EmailMessage
from django.db.models import Q
from django.http import HttpResponse, StreamingHttpResponse
from django.template.loader import get_template
from django.utils import timezone
from django.utils.html import strip_tags
from rest_framework import viewsets, permissions, status
from rest_framework.decorators import api_view, permission_classes, action
from rest_framework.parsers import MultiPartParser, FormParser, JSONParser
from rest_framework.permissions import AllowAny, IsAuthenticated
from rest_framework.response import Response
from rest_framework.authtoken.models import Token
from rest_framework.views import APIView
from rest_framework.generics import get_object_or_404
import users.serializers as app_serializers
import NurtrDjango.models as app_models
from NurtrDjango.utils import strip_non_model_fields
from rest_framework.decorators import action
from django.contrib.auth import get_user_model
import time  # Import time module for delay
import requests
from django.conf import settings
from rest_framework.status import HTTP_400_BAD_REQUEST, HTTP_200_OK
from concurrent.futures import ThreadPoolExecutor, as_completed
from django.http import JsonResponse
import asyncio


import asyncio
import aiohttp
import time
from rest_framework.response import Response
from rest_framework.status import HTTP_400_BAD_REQUEST, HTTP_200_OK

User = get_user_model()

class UserViewSet(viewsets.ModelViewSet):
    queryset = User.objects.order_by('-id')

    def get_serializer_class(self):
        return app_serializers.UserSerializer  # Always use the same serializer

    def get_permissions(self):
        if self.action in ['create', 'authenticate']:
            return [AllowAny()]
        return [IsAuthenticated()]

    @action(detail=False, methods=['post'], permission_classes=[AllowAny])
    def authenticate(self, request):
        """Authenticate user and return an auth token."""
        email = request.data.get('email')
        password = request.data.get('password')

        if not email or not password:
            return Response({'error': 'Email and password are required.'}, status=status.HTTP_400_BAD_REQUEST)

        user = get_object_or_404(User, email=email)
        

        if not user.is_active:
            user.is_active = True 
            user.save()
        if not user.check_password(password):
            return Response({'error': 'Invalid credentials.'}, status=status.HTTP_401_UNAUTHORIZED)

        token, created = Token.objects.get_or_create(user=user)
        user_data = app_serializers.UserSerializer(user, context={'request': request}).data
        user_data['token'] = token.key 

        return Response(user_data, status=status.HTTP_200_OK)

    def create(self, request, *args, **kwargs):
        """Custom signup method to return an auth token after successful registration."""
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        try:
            user = serializer.save() 
            if 'password' in request.data:
                user.set_password(request.data['password']) 
                print(user,"userprinting")
                user.save() # Create user
        except Exception as e:
            return Response({'error': str(e)}, status=status.HTTP_400_BAD_REQUEST)

        # Generate authentication token
        token, created = Token.objects.get_or_create(user=user)

        # Prepare response with user data and token
        user_data = app_serializers.UserSerializer(user, context={'request': request}).data
        user_data['token'] = token.key  # Include token in response

        return Response(user_data, status=status.HTTP_201_CREATED)

import asyncio
import aiohttp
import time
from rest_framework.response import Response
from rest_framework.status import HTTP_400_BAD_REQUEST, HTTP_200_OK
from rest_framework.views import APIView
from rest_framework.parsers import JSONParser
from rest_framework.permissions import AllowAny

class PlacesAPIView(APIView):
    permission_classes = [AllowAny]
    parser_classes = [JSONParser]

    #API_KEY = "AIzaSyDqGaMkkyppcmCU39xhPjSoi4K06BnsS98"
    PLACES_API_KEY = "AIzaSyDOaI_FZgX5flU5TC6JuxYSWOcCBRCGScY1"
    API_KEY = "AIzaSyBfW8nU2EoPK1Zg_bYOSREzqmRDwZfUgbM"
    #BASE_URL = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"
    BASE_URL = "https://places.googleapis.com/v1/places:searchNearby"
    DETAILS_URL = "https://places.googleapis.com/v1/places/"
    GEOCODE_URL = "https://maps.googleapis.com/maps/api/geocode/json"

    async def get_coordinates_from_zip(self, zip_code):
        """Fetch latitude and longitude from ZIP code using Geocoding API."""
        # Include USA in the address query
        url = f"{self.GEOCODE_URL}?address={zip_code}+USA&key={self.API_KEY}"

        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                data = await response.json()

        #print(f"coordinate data response {data}")

        if data.get("results"):
            location = data["results"][0]["geometry"]["location"]
            return location["lat"], location["lng"]

        return None, None

    def post(self, request):
        start_time = time.time()
        data = request.data
        zip_code = data.get("zip_code")  # Get ZIP code if provided
        latitude = data.get("latitude", 40.712776)
        longitude = data.get("longitude", -74.005974)
        radius = data.get("radius", 500)
        types = data.get("types", [])
        min_price = data.get("minPrice", 0)
        max_price = data.get("maxPrice", 500)
        filters = data.get("filters", [])
        page = int(data.get("page", 1))
        #items_per_page = 4

        print('places request ', request)
        if zip_code:
            #print(f"zip code: {zip_code} {type(zip_code)}")
            latitude, longitude = asyncio.run(self.get_coordinates_from_zip(zip_code))
            if not latitude or not longitude:
                return Response({"error": "Invalid ZIP code or location not found"}, status=HTTP_400_BAD_REQUEST)
        
        if not latitude or not longitude:
            return Response({"error": "Latitude and Longitude are required"}, status=HTTP_400_BAD_REQUEST)

        params = {
            "location": f"{latitude},{longitude}",
            "radius": radius,
            "key": self.API_KEY,
        }
        
        if types:
            params["types"] = types
            print(f"num types {len(types)}")

        print(f"Params: {params}") # OK HERE

        # Run asynchronous I/O using asyncio and aiohttp
        detailed_results = asyncio.run(self.async_main(params, filters, min_price, max_price))

        import math
        def haversine(lat1, lon1, lat2, lon2):
            """
            Calculate the great-circle distance between two points on the Earth using the Haversine formula.
            Returns distance in miles.
            """
            # Radius of the Earth in miles
            R = 3958.8

            # Convert latitude and longitude from degrees to radians
            lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])

            # Differences in coordinates
            dlat = lat2 - lat1
            dlon = lon2 - lon1

            # Haversine formula
            a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
            c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

            # Distance in miles
            distance = R * c
            return distance
        
        # Calculate and store distance for each result
        for result in detailed_results:
            if "location" in result and "latitude" in result["location"] and "longitude" in result["location"]:
                result_lat = result["location"]["latitude"]
                result_lon = result["location"]["longitude"]
                origin_lat = latitude
                origin_lon = longitude
                
                # Calculate distance using Haversine formula
                distance = haversine(origin_lat, origin_lon, result_lat, result_lon)
                
                # Store distance in the result
                result["distance"] = round(distance, 2)  # Round to 2 decimal places

        #print(f"Detailed Results: {detailed_results}")

        # Apply pagination AFTER filtering
        total_filtered_results = len(detailed_results)
        #start_idx = (page - 1) * items_per_page
        #paginated_results = detailed_results[start_idx : start_idx + items_per_page]
        paginated_results = detailed_results

        execution_time = round((time.time() - start_time) * 1000, 2)
        return Response({
            "results": paginated_results,
            "page": page,
            "total_results": total_filtered_results,
            "execution_time_ms": execution_time
        }, status=HTTP_200_OK)

    async def async_main(self, params, filters, min_price, max_price):
        async with aiohttp.ClientSession() as session:
            all_results = await self.async_fetch_all_places(params, session)
            
            print(f"num results before filter {len(all_results)}")
            all_results = [r for r in all_results if self.apply_filters(r, filters, min_price, max_price)]
            print(f"num results after filter {len(all_results)}")

            all_results = await self.fetch_place_images(all_results)
            all_results = [p for p in all_results if p.get("imagePlaces", [])]
            #detailed_results = await self.async_fetch_all_place_details(all_results, filters, min_price, max_price, session)
        
        #print(f'all plcace results: {len(all_results)}')
        return all_results

    async def async_fetch_place_details(self, place_id, session):
        """Fetch individual place details asynchronously."""
        if not place_id:
            print(f"Place ID not found")
            return None

        url = f"{self.DETAILS_URL}{place_id}"
        headers = {
            "Content-Type": "application/json",
            "X-Goog-Api-Key": self.API_KEY,
            "X-Goog-FieldMask": "*",  # Fetch all fields
        }

        async with session.get(url, headers=headers) as response:
            details_data = await response.json()
        
        photos = details_data.get("photos", [])[:3]  # Limit to first 5 photos
        image_places = []

        for photo in photos:
            name = photo.get("name")
            if name:
                image_url = f"https://places.googleapis.com/v1/{name}/media?key={self.API_KEY}&maxHeightPx=600"
                #proxied_image_url = f"/api/serve-image?url={google_image_url}"  # Proxy through your backend
                image_places.append(image_url)

        details_data["images"] = image_places  # Attach new key

        return details_data

    def get(self, request, place_id=None):
        """Endpoint to fetch place details by ID."""
        if not place_id:
            return Response({"error": "Place ID is required"}, status=HTTP_400_BAD_REQUEST)

        async def fetch_details():
            async with aiohttp.ClientSession() as session:
                return await self.async_fetch_place_details(place_id, session)

        place_details = asyncio.run(fetch_details())
        
        print('Number of reviews for place', len(place_details.get('reviews', [])))
        #print('place details', place_details)

        if not place_details:
            return Response({"error": "Place not found"}, status=HTTP_404_NOT_FOUND)

        return Response(place_details, status=HTTP_200_OK)

    async def async_fetch_all_places(self, params, session):
        """Fetches all pages asynchronously using the new Places API v1."""
        headers = {
            "Content-Type": "application/json",
            "X-Goog-Api-Key": self.API_KEY,
            "X-Goog-FieldMask": "places.id,places.displayName,places.location,places.businessStatus,places.photos,places.formattedAddress,places.priceRange,places.types,places.userRatingCount,places.rating",  # Include places.id
        }

        # Get the list of types from params, default to ["restaurant"]
        types = params.get("types", ["restaurant"])
        
        # Split types into 3 groups using modulo operator
        group_0 = [types[i] for i in range(len(types)) if i % 3 == 0]
        group_1 = [types[i] for i in range(len(types)) if i % 3 == 1]
        group_2 = [types[i] for i in range(len(types)) if i % 3 == 2]

        # Function to fetch results for a group of types
        async def fetch_group(group_types):
            if not group_types:
                return []

            print(f"Fetching group: {group_types}") 

            request_body = {
                "locationRestriction": {
                    "circle": {
                        "center": {
                            "latitude": float(params["location"].split(",")[0]),
                            "longitude": float(params["location"].split(",")[1]),
                        },
                        "radius": params["radius"],
                    }
                },
                "includedTypes": group_types,
                "maxResultCount": 17,
            }

            all_results = []
            next_page_token = None

            while True:
                if next_page_token:
                    request_body["pageToken"] = next_page_token
                    await asyncio.sleep(2)  # Google recommends a short delay between paginated requests

                async with session.post(self.BASE_URL, json=request_body, headers=headers) as response:
                    data = await response.json()
                
                all_results.extend(data.get("places", []))

                next_page_token = data.get("nextPageToken")
                if not next_page_token or len(all_results) >= 20:
                    break
            
            return all_results

        # Fetch results for all 3 groups concurrently
        results = await asyncio.gather(
            fetch_group(group_0),
            fetch_group(group_1),
            fetch_group(group_2)
        )

        # Combine results from all groups
        all_results = [item for sublist in results for item in sublist]

        print(f"results 0: {len(results[0])}")
        print(f"results 1: {len(results[1])}")
        print(f"results 2: {len(results[2])}")

        print(f"num results for all 3 groups {len(all_results)}")

        # Remove duplicates based on place ID
        unique_results = []
        seen_ids = set()
        for result in all_results:
            if result["id"] not in seen_ids:
                unique_results.append(result)
                seen_ids.add(result["id"])

        return unique_results

    async def async_fetch_all_place_details(self, places, filters, min_price, max_price, session):
        """Fetch all place details concurrently using asyncio.gather."""
        tasks = [
            asyncio.create_task(self.async_fetch_place_details(place["id"], session))
            for place in places
        ]
        results = await asyncio.gather(*tasks)

        # Debugging: Log the results
        #print(f"Place details results: {results}")

        filtered_places = [res for res in results if res]
        filtered_places = await self.fetch_place_images(filtered_places, session)

        return filtered_places

    async def fetch_place_images(self, places):
        """Fetch up to 5 photos for each place and add imagePlaces key."""
        for place in places:
            photos = place.get("photos", [])[:3]  # Limit to first 5 photos
            image_places = []

            print(f"num distinct photos: {len(set([p['name'] for p in photos]))}")
            for photo in photos:
                name = photo.get("name")
                if name:
                    image_url = f"https://places.googleapis.com/v1/{name}/media?key={self.API_KEY}&maxHeightPx=600"
                    #proxied_image_url = f"/api/serve-image?url={google_image_url}"  # Proxy through your backend
                    image_places.append(image_url)

            #print('image place 1', image_places[0])
            #print('\nimage place 2', image_places[1])
            #print('\nimage place 3', image_places[2])

            print(f"num distinct photos: {len(set(image_places))}")

            place["imagePlaces"] = image_places  # Attach new key

        return places

    def apply_filters(self, place, filters, min_price, max_price):
        """Applies filters to the place data."""
        if not filters:
            return True

        price_range = place.get("priceRange", {})
        start_price = int(price_range.get("startPrice", {}).get("units", 0))
        end_price = int(price_range.get("endPrice", {}).get("units", 0))

        if not (min_price <= start_price <= max_price or min_price <= end_price <= max_price):
            return False

        if "reservable" in filters and not place.get("reservable", False):
            return False

        for key in ["parkingOptions", "accessibilityOptions", "restroom"]:
            if key in filters and not place.get(key):
                return False

        return True

def serve_image(request, image_url):
    """Proxy image from Google's server."""
    response = requests.get(image_url, stream=True)
    return HttpResponse(response.content, content_type=response.headers['Content-Type'])
