import random
import uuid
import math
from django.http import HttpResponse, StreamingHttpResponse
from rest_framework import viewsets, permissions, status
from rest_framework.decorators import api_view, permission_classes, action
from rest_framework.parsers import MultiPartParser, FormParser, JSONParser
from rest_framework.permissions import AllowAny, IsAuthenticated
from rest_framework.response import Response
from rest_framework.authtoken.models import Token
from rest_framework.views import APIView
from rest_framework.generics import get_object_or_404
import users.serializers as app_serializers
from rest_framework.decorators import action
from django.contrib.auth import get_user_model
import time  # Import time module for delay
import requests
from django.conf import settings
from rest_framework.status import HTTP_404_NOT_FOUND
from concurrent.futures import ThreadPoolExecutor, as_completed
from django.http import JsonResponse
import asyncio
from django.core.cache import cache # Add cache import
from google.cloud.storage import Client, transfer_manager
from google.cloud import exceptions
from typing import List
import asyncio
import time
from rest_framework.response import Response
from rest_framework.status import HTTP_400_BAD_REQUEST, HTTP_200_OK, HTTP_429_TOO_MANY_REQUESTS, HTTP_500_INTERNAL_SERVER_ERROR
from rest_framework.views import APIView
from rest_framework.parsers import JSONParser
from rest_framework.permissions import AllowAny
import aiohttp
from dotenv import load_dotenv
import os
import traceback
import serpapi
from .models import Place, Event, UserEventRecommendation, UserRecommendationEvent # Import the models
from django.core.exceptions import ObjectDoesNotExist
from asgiref.sync import sync_to_async
import concurrent.futures
import logging
import json
from datetime import datetime, timedelta
from google import genai  # Add this import for Gemini LLM integration
from users.models import Child  # Add this import
from .models import UserRecommendation, UserRecommendationPlace  # Add this import
import hashlib
from pydantic import BaseModel
from typing import List

load_dotenv()

# Pydantic models for Gemini structured output
class PlaceRecommendation(BaseModel):
    index: int
    score: int
    explanation: str

class PlaceRecommendations(BaseModel):
    recommendations: List[PlaceRecommendation]

IP_INFO_API_TOKEN = os.getenv("IP_INFO_API_TOKEN")
IMAGE_DOWNLOAD_URL = os.getenv("IMAGE_DOWNLOAD_URL")
TESTING = os.getenv("TESTING", "False").lower() == "true"
SERP_API_KEY = os.getenv("SERP_API_KEY")

print(f"TESTING {os.getenv('TESTING')}")

storage_client = Client()

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

        print("user_data", user_data)
        return Response(user_data, status=status.HTTP_200_OK)

    @action(detail=False, methods=['get'], permission_classes=[IsAuthenticated])
    def me(self, request):
        """Get current authenticated user's profile."""
        user_data = app_serializers.UserSerializer(request.user, context={'request': request}).data
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

# Define constants for rate limiting
MAX_UNAUTHENTICATED_REQUESTS_PER_IP = 5
CACHE_TIMEOUT_SECONDS = None


def set_cors_for_get(bucket_name):
    """Sets a bucket's CORS policies to allow GET requests from any origin."""
    # bucket_name = "your-bucket-name"

    storage_client = Client()
    bucket = storage_client.get_bucket(bucket_name) # Use get_bucket for existence check

    # Define the CORS rule for GET requests
    cors_rule = {
        "origin": ["*"], # Allow any origin
        "method": ["GET"], # Allow only GET method
        "responseHeader": ["Content-Type"], # Allow browser access to Content-Type header
        "maxAgeSeconds": 3600 # Cache preflight response for 1 hour
    }

    # Assign the list containing the rule to the bucket's cors property
    bucket.cors = [cors_rule]

    # Patch the bucket to apply the change
    bucket.patch()

    print(f"Set CORS policies for bucket {bucket.name}: {bucket.cors}")
    return bucket

def upload_single_image_threaded(blob, local_path):
    """Helper synchronous function to be run in a thread."""
    max_attempts = 5

    for _ in range(max_attempts):
        try:
            blob.upload_from_filename(local_path)
            #print(f"Successfully uploaded {local_path} to {blob.name}")
            # Make the blob publicly viewable (optional, adjust as needed)
            # blob.make_public()
            return blob.public_url
        except Exception as e:
            print(f"Error uploading {local_path} to {blob.name}: {e}")
            time.sleep(random.randint(1,5))

    return None # Indicate failure

def upload_place_images_to_bucket(
    bucket_name, place_id, image_paths
):
    """Upload local image files for a specific place into a place-specific folder
    within a bucket, concurrently.

    Args:
        bucket_name (str): The ID of your GCS bucket.
        place_id (str): The unique identifier for the place (e.g., Google Place ID).
        image_paths (list[str]): A list of local file paths to the images for this place.
        workers (int): The maximum number of processes/threads to use.
    """
    print(f"Uploading images for Place ID '{place_id}' to bucket '{bucket_name}'...")

    bucket = storage_client.bucket(bucket_name)

    # Define the prefix for the place ID folder
    prefix = f"places/{place_id}/"

    # List and delete existing blobs in the place ID folder
    blobs_to_delete = bucket.list_blobs(prefix=prefix)
    for blob in blobs_to_delete:
        print(f"Deleting existing blob: {blob.name}")
        blob.delete()

    upload_tasks = []
    gcs_links = []
    for local_file_path in image_paths:
        original_filename = os.path.basename(local_file_path)
        destination_blob_name = f"places/{place_id}/{original_filename}"
        blob = bucket.blob(destination_blob_name)
        #uploads.append((local_file_path, blob))
        # Create a coroutine to run the synchronous upload in a thread
        #task = asyncio.to_thread(upload_single_image_threaded, blob, local_file_path)
        upload_tasks.append((blob, local_file_path))

    if not upload_tasks:
        print(f"No valid image paths found to upload for Place ID '{place_id}'.")
        return []

    start_time = time.time()
    print(f"Starting concurrent upload of {len(upload_tasks)} images for Place ID '{place_id}'...")

    # Use ThreadPoolExecutor for synchronous concurrent uploads
    from concurrent.futures import ThreadPoolExecutor, as_completed
    with ThreadPoolExecutor(max_workers=min(len(upload_tasks), 10)) as executor:
        futures = [executor.submit(upload_single_image_threaded, blob, local_path) for blob, local_path in upload_tasks]
        results = [future.result() for future in as_completed(futures)]

    # Process results
    for result in results:
        if result:
            gcs_links.append(result)

    end_time = time.time()
    print(f"Finished uploading for Place ID '{place_id}'.")
    print(f"Total time for {len(gcs_links)} successful uploads: {end_time - start_time:.2f} seconds")
    return gcs_links


def download_image(image_url):
    local_file_path = f"temp_images/{uuid.uuid4()}.jpg"
    for _ in range(3):
        try:
            response = requests.get(image_url, timeout=10)  # Set a timeout for the request
            if response.status_code == 200:
                with open(local_file_path, 'wb') as f:
                    f.write(response.content)
                return local_file_path
            else:
                raise Exception(f"Failed to download {image_url}: Status {response.status_code}")
        except Exception as e:
            print(f"Error downloading {image_url}: {str(e)}")
            time.sleep(random.randint(1,5))
    
    return None

def check_existing_images(bucket_name: str, place_id: str, max_retries: int = 3) -> List[str]:
    """
    Check if images for the given place_id exist in the specified GCS bucket with retry logic.
    """
    print(f"Checking for existing images in bucket '{bucket_name}' for Place ID '{place_id}'...")
    start_time = time.time()
    
    for attempt in range(max_retries + 1):
        try:
            storage_client = Client()
            bucket = storage_client.bucket(bucket_name)
            prefix = f"places/{place_id}/"
            
            blobs = bucket.list_blobs(prefix=prefix)
            existing_images = [blob.public_url for blob in blobs]
            
            end_time = time.time()
            print(f"Finished checking for existing images for Place ID '{place_id}'.")
            print(f"Total time for checking {len(existing_images)} existing images: {end_time - start_time:.2f} seconds")
            return existing_images
            
        except Exception as e:
            if attempt == max_retries:
                # Last attempt failed, log and return empty list
                logging.error(f"Failed to check existing images for place {place_id} after {max_retries} retries: {str(e)}")
                print(f"Error checking images for Place ID '{place_id}' after {max_retries} retries: {str(e)}")
                return []
            else:
                # Wait before retrying with exponential backoff
                wait_time = (2 ** attempt) + 1  # 1, 3, 5 seconds
                print(f"Attempt {attempt + 1} failed for Place ID '{place_id}', retrying in {wait_time}s...")
                time.sleep(wait_time)
    
    return []


def get_images_from_serp(place_image_link, max_images=1):
    """
    Extract image URLs from the SERP API response.

    Args:
        place_image_link (str): The link to the place's images.

    Returns:
        list: List of image URLs.
    """
    print(f"Fetching images from SERP API for link: {place_image_link}")
    try:
        api_key = os.environ.get('SERP_API_KEY')

        params = {'api_key': api_key}

        response = requests.get(place_image_link, params=params)

        if response.status_code == 200:
            data = response.json()
            image_urls = [image_data['image'] for image_data in data.get('photos', []) if image_data.get('image')][:max_images]
            print(f"Fetched {len(image_urls)} images from SERP API.")
            return image_urls
        else:
            print(f"Failed to fetch images from SERP API: Status {response.status_code}, Response: {response.text}")
            return []
    except Exception as e:
        print(f"Error fetching images from SERP API: {str(e)}")
        return []

def get_reviews_from_serp(place_id, max_reviews=5):
    """
    Extract review URLs from the SERP API response.

    Args:
        review_link (str): The link to the place's reviews.

    Returns:
        list: List of review URLs.
    """
    print(f"Fetching reviews from SERP API for place ID: {place_id}")
    try:

        params = {
            "engine": "google_maps_reviews",
            "hl": "en",
            "api_key": SERP_API_KEY,
            "place_id": place_id
        }

        start_time = time.time()
        results = serpapi.search(params).get('reviews', [])[:max_reviews]
        end_time = time.time()

        print(f"Total time for fetching {len(results)} reviews: {end_time - start_time:.2f} seconds")

        print(f"Fetched {len(results)} reviews from SERP API.")
        return results
    except Exception as e:
        print(f"Error fetching reviews from SERP API: {str(e)}")
        return []

def process_images_for_place(place, min_images=1):
    """
    Process and download images for a place (synchronous).
    This entire function runs in a worker thread.
    
    Args:
        place_id (str): The place ID.
        
    Returns:
        list: Processed image URLs from GCS.
    """
        
    try:
        print(f"Processing images for place: {place.get('place_id', place.get('id'))}")

        if min_images == 1:
            max_images = 1
        else:
            max_images = 5

        image_urls = get_images_from_serp(place["photos_link"], max_images=max_images)

        if not image_urls:
            print(f"No image URLs found for place ID {place.get('place_id', place.get('id'))}")
            return []

        place_id = place.get('place_id', place.get('id'))

        # Check for existing images first
        existing_image_links = check_existing_images(
            bucket_name='nurtr-places', 
            place_id=place_id
        )
        
        if existing_image_links and len(existing_image_links) >= min_images:
            print(f"Using existing images for place ID {place_id}")
            return existing_image_links
            
        # Create temp directory if needed
        os.makedirs("temp_images", exist_ok=True)
        
        # Download images sequentially (since the entire function is already in a thread)
        local_image_paths = []
        with ThreadPoolExecutor(max_workers=min(10, len(image_urls))) as inner_executor:
            # Map the download_image function over all image URLs concurrently
            download_futures = {inner_executor.submit(download_image, url): url for url in image_urls}
            
            # Process results as they complete
            for future in as_completed(download_futures):
                path = future.result()
                if path:
                    local_image_paths.append(path)
        
        print(f"Successfully downloaded {len(local_image_paths)} images locally for place ID {place_id}")

        # Upload to GCS if we have local images
        if local_image_paths:
            gcs_image_urls = upload_place_images_to_bucket(
                bucket_name='nurtr-places',
                place_id=place_id,
                image_paths=local_image_paths
            )
            return gcs_image_urls
        else:
            print(f"No images were successfully downloaded for place ID {place_id}")
            return []
            
    except Exception as e:
        print(f"Error processing images for place ID {place_id}: {e}")
        import traceback
        traceback.print_exc()
        return []

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

    def get_client_ip(self, request):
        """Utility function to get client IP address."""
        x_forwarded_for = request.META.get('HTTP_X_FORWARDED_FOR')
        if x_forwarded_for:
            ip = x_forwarded_for.split(',')[0]
        else:
            ip = request.META.get('REMOTE_ADDR')
        return ip

    async def get_coordinates_from_zip(self, zip_code):
        """Fetch latitude and longitude from ZIP code using Geocoding API."""
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

        #set_cors_for_get('nurtr-places')

        #print('places request ', request.data)
        print('user is authenticated', request.data.get('is_authenticated', False))

        # Rate limiting for unauthenticated users
        if not request.data.get('is_authenticated', False):
            rate_limit_response = self.check_rate_limit(
                request,
                MAX_UNAUTHENTICATED_REQUESTS_PER_IP,
                CACHE_TIMEOUT_SECONDS
            )
            if rate_limit_response:
                return rate_limit_response

        data = request.data
        zip_code = data.get("zip_code")  # Get ZIP code if provided
        latitude = data.get("latitude", 40.712776)
        longitude = data.get("longitude", -74.005974)
        radius = data.get("radius", 500)
        types = data.get("types", {})
        min_price = data.get("minPrice", 0)
        max_price = data.get("maxPrice", 500)
        filters = data.get("filters", [])
        page = int(data.get("page", 1))

        if data.get('load_more', False):
            page += 1
        
        print('page', page)
        print('load more', data.get('load_more', False))
        
        if zip_code:
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
            print(f"num types {len(types.keys())}")

        print(f"Params: {params}") # OK HERE

        # Run asynchronous I/O to get place details from external API
        api_results = asyncio.run(self.async_main(params, filters, min_price, max_price))

        processed_results = [] # List to store final results (from DB or API)

        # Haversine function for distance calculation (moved outside the loop)
        def haversine(lat1, lon1, lat2, lon2):
            if None in [lat1, lon1, lat2, lon2]: # Handle potential None values
                 return None
            R = 3958.8 # Radius of the Earth in miles
            lat1, lon1, lat2, lon2 = map(math.radians, [float(lat1), float(lon1), float(lat2), float(lon2)]) # Ensure floats before radians
            dlat = lat2 - lat1
            dlon = lon2 - lon1
            a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
            c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
            distance = R * c
            return round(distance, 2)

        seen_place_ids = set()

        # Process each result from the API
        for api_result in api_results:
            place_id = api_result.get("place_id")
            if not place_id or place_id in seen_place_ids:
                continue # Skip if no place_id

            seen_place_ids.add(place_id)

            try:
                # Check if place exists in DB - use a direct function call with sync_to_async
                # instead of await
                get_place = sync_to_async(Place.objects.get)
                try:
                    place = asyncio.run(get_place(place_id=place_id))
                    print(f"Found place in DB: {place_id}")

                    # Use DB data, construct dictionary matching API structure (or desired output)
                    place_data = {
                        "place_id": place.place_id,
                        "title": place.title,
                        "description": place.description,
                        "data_id": place.data_id,
                        "reviews_link": place.reviews_link,
                        "photos_link": place.photos_link,
                        "location": { # Reconstruct location structure
                            "latitude": float(place.latitude) if place.latitude is not None else None,
                            "longitude": float(place.longitude) if place.longitude is not None else None,
                        },
                        "gps_coordinates": { # Add gps_coordinates if needed by frontend
                            "latitude": float(place.latitude) if place.latitude is not None else None,
                            "longitude": float(place.longitude) if place.longitude is not None else None,
                        },
                        "type": place.type,
                        "types": place.types,
                        "address": place.address,
                        "extensions": place.extensions,
                        "displayName": {"text": place.display_name} if place.display_name else None, # Reconstruct display name
                        "formattedAddress": place.formatted_address,
                        "rating": place.rating,
                        "reviews": place.reviews, # This is review count
                        "operating_hours": place.hours, # Use stored hours
                        "place_images": place.place_images,
                        "reviews_list": place.reviews_list, # This is the list of reviews
                        "popular_times": place.popular_times,
                        "category": place.category,  # Include category in response
                        # Add other fields from the model as needed
                    }
                    # Calculate distance using DB coordinates and request location
                    db_lat = float(place.latitude) if place.latitude is not None else None
                    db_lon = float(place.longitude) if place.longitude is not None else None
                    place_data["distance"] = haversine(float(latitude), float(longitude), db_lat, db_lon)

                    # Add to processed results
                    processed_results.append(place_data)

                except ObjectDoesNotExist:
                    # Place not in DB, process from API and save
                    place_data = api_result # Start with API result

                    # Calculate distance using API coordinates
                    api_lat = api_result.get("location", {}).get("latitude")
                    api_lon = api_result.get("location", {}).get("longitude")
                    place_data["distance"] = haversine(float(latitude), float(longitude), api_lat, api_lon)

                    # Save to DB for future requests - use a wrapper function with sync_to_async
                    def save_place_to_db():
                        try:
                            # Create new Place instance
                            new_place = Place(
                                title=api_result.get("title"),
                                description=api_result.get("description"),
                                place_id=place_id,
                                data_id=api_result.get("data_id"),
                                reviews_link=api_result.get("reviews_link"),
                                photos_link=api_result.get("photos_link"),
                                latitude=api_lat,
                                longitude=api_lon,
                                type=api_result.get("type"),
                                types=api_result.get("types", []),
                                address=api_result.get("address"),
                                extensions=api_result.get("extensions", {}),
                                display_name=api_result.get("displayName", {}).get("text") or api_result.get("title"),
                                formatted_address=api_result.get("formattedAddress"),
                                rating=api_result.get("rating"),
                                reviews=api_result.get("reviews") or api_result.get("userRatingCount"),
                                hours=api_result.get("operating_hours", {}),
                                place_images=api_result.get("imagePlaces", []), # Save image URLs
                                reviews_list=api_result.get("reviews_list", []) # Save reviews list if present in api_result
                            )
                            new_place.save()
                            return True
                        except Exception as e:
                            print(f"Error saving place to DB: {e}")
                            return False

                    # Run the save function in a synchronous context
                    save_place_to_db()

                    # Calculate distance for the newly saved place
                    # Use the coordinates directly from the api_result for distance calculation
                    place_data["distance"] = haversine(float(latitude), float(longitude), api_lat, api_lon)

                    # Add to processed results
                    processed_results.append(place_data)

            except Exception as e:
                print(f"Error processing place {place_id}: {e}")
                # Still add API result to processed results
                place_data = api_result
                # Calculate distance
                api_lat = api_result.get("location", {}).get("latitude")
                api_lon = api_result.get("location", {}).get("longitude")
                place_data["distance"] = haversine(float(latitude), float(longitude), api_lat, api_lon)
                processed_results.append(place_data)

        # Sort by distance if calculated
        #processed_results.sort(key=lambda x: x.get("distance", float("inf")))

        total_results = len(processed_results) # Use count from processed list
        paginated_results = processed_results # For now, returning all processed results

        execution_time = round((time.time() - start_time) * 1000, 2)
        return Response({
            "results": paginated_results, # Use the processed & paginated list
            "page": page,
            "total_results": total_results,
            "execution_time_ms": execution_time
        }, status=HTTP_200_OK)

    async def async_main(self, params, filters, min_price, max_price):
        async with aiohttp.ClientSession() as session:
            # 1. Fetch initial place data from API
            print("Fetching initial place data from API...")
            api_results = await self.async_fetch_all_places(params, session)
            if not api_results:
                return []

            # 2. Extract place IDs for database lookup
            place_ids = [p.get("place_id") for p in api_results if p.get("place_id")]
            print(f"Found {len(place_ids)} place IDs from initial API fetch")

            # 3. Create lookup map for API results to use later
            api_results_map = {p.get("place_id"): p for p in api_results if p.get("place_id")}

            # 4. Check database for existing places with these IDs
            db_places = {}
            places_needing_images = []
            processed_results = []

            try:
                # Query DB for existing places with these IDs - use sync_to_async
                existing_places = await sync_to_async(list)(Place.objects.filter(place_id__in=place_ids))
                print(f"Found {len(existing_places)} places in database")

                start_time = time.time()
                for place in existing_places:
                    # Convert DB model to dict format expected by frontend
                    place_dict = {
                        "place_id": place.place_id,
                        "title": place.title,
                        "description": place.description,
                        "data_id": place.data_id,
                        "reviews_link": place.reviews_link,
                        "photos_link": place.photos_link,
                        "location": {
                            "latitude": float(place.latitude) if place.latitude is not None else None,
                            "longitude": float(place.longitude) if place.longitude is not None else None
                        },
                        "gps_coordinates": {
                            "latitude": float(place.latitude) if place.latitude is not None else None,
                            "longitude": float(place.longitude) if place.longitude is not None else None
                        },
                        "type": place.type,
                        "types": place.types,
                        "address": place.address,
                        "extensions": place.extensions,
                        "displayName": {"text": place.display_name} if place.display_name else None,
                        "formattedAddress": place.formatted_address,
                        "rating": place.rating,
                        "reviews": place.reviews,
                        "operating_hours": place.hours,
                        "imagePlaces": place.place_images or [],
                        "reviews_list": place.reviews_list or [],
                        "popular_times": place.popular_times,
                        "category": place.category,  # Include category in response
                        "source": "database"
                    }

                    # Store in our dictionary
                    db_places[place.place_id] = place_dict

                    # Check if this place has enough images
                    min_images = 1  # Minimum number of images required
                    if not place.place_images or len(place.place_images) < min_images:
                        # If from database but needs images, add to processing list
                        places_needing_images.append(place_dict)
                    else:
                        # Has enough images, add directly to final results
                        processed_results.append(place_dict)
                        print(f"Using cached place with images: {place.place_id}")

            except Exception as e:
                print(f"Error querying database for places: {e}")
                # Continue execution even if DB query fails

            execution_time = round((time.time() - start_time) * 1000, 2)
            print(f"Total time for querying database: {execution_time} ms")

            # 5. Identify places not in database - need full processing
            missing_place_ids = [pid for pid in place_ids if pid not in db_places]
            print(f"{len(missing_place_ids)} places not found in database")

            # 6. Add API results for places not in database to the processing list
            for place_id in missing_place_ids:
                if place_id in api_results_map:
                    api_result = api_results_map[place_id]
                    api_result["source"] = "api"
                    places_needing_images.append(api_result)

            # 7. Process images only for places that need them
            print(f"Processing images for {len(places_needing_images)} places...")
            if places_needing_images:
                start_time = time.time()
                with ThreadPoolExecutor(max_workers=20) as executor:
                    # Map place to future for easy lookup after completion
                    future_to_place = {
                        executor.submit(process_images_for_place, place): place
                        for place in places_needing_images
                    }

                    # Process completed tasks as they finish
                    for future in as_completed(future_to_place):
                        place = future_to_place[future]
                        place_id = place.get('place_id')

                        try:
                            # Get processed image URLs
                            processed_urls = future.result()

                            # Update the place dict with new images
                            if processed_urls:
                                place["imagePlaces"] = processed_urls
                                print(f"Processed {len(processed_urls)} images for place {place_id}")

                                # Add to our results
                                processed_results.append(place)

                                # If it came from API (not DB), save to database
                                if place.get("source") == "api":
                                    try:
                                        # Get coordinates
                                        lat_val = place.get("location", {}).get("latitude")
                                        lon_val = place.get("location", {}).get("longitude")
                                        
                                        async def save_to_db():
                                            new_place = Place(
                                                title=place.get("title"),
                                                description=place.get("description"),
                                                place_id=place_id,
                                                data_id=place.get("data_id"),
                                                reviews_link=place.get("reviews_link"),
                                                photos_link=place.get("photos_link"),
                                                latitude=lat_val,
                                                longitude=lon_val,
                                                type=place.get("type"),
                                                types=place.get("types", []),
                                                address=place.get("address"),
                                                extensions=place.get("extensions", {}),
                                                display_name=place.get("displayName", {}).get("text") or place.get("title"),
                                                formatted_address=place.get("formattedAddress"),
                                                rating=place.get("rating"),
                                                reviews=place.get("reviews") or place.get("userRatingCount"),
                                                hours=place.get("operating_hours", {}),
                                                place_images=processed_urls,  # Store processed images
                                                reviews_list=place.get("reviews_list", []),
                                                popular_times=place.get("popular_times", []),
                                                category=place.get("category", "")
                                            )
                                            await sync_to_async(new_place.save)()
                                            return True

                                        await save_to_db()
                                        print(f"Saved new place to database: {place_id}")
                                    except Exception as e:
                                        print(f"Error saving place {place_id} to database: {e}")

                                # If it came from DB, update the images
                                elif place.get("source") == "database":
                                    try:
                                        async def update_db_images():
                                            db_place = await sync_to_async(Place.objects.get)(place_id=place_id)
                                            db_place.place_images = processed_urls
                                            await sync_to_async(db_place.save)(update_fields=['place_images'])
                                            return True

                                        await update_db_images()
                                        print(f"Updated images for existing place in database: {place_id}")
                                    except Exception as e:
                                        print(f"Error updating images for place {place_id}: {e}")
                            else:
                                # No images processed, but still add if from API
                                if place.get("source") == "api":
                                    processed_results.append(place)
                                print(f"No images processed for place {place_id}")

                        except Exception as e:
                            print(f"Error processing place {place_id}: {e}")
                            # Still add API result to results
                            if place.get("source") == "api":
                                place["imagePlaces"] = []
                                processed_results.append(place)

                end_time = time.time()
                execution_time = round((end_time - start_time) * 1000, 2)
                print(f"Finished processing images. Execution time: {execution_time} ms")

            # 8. Filter results to places with images
            final_results = [p for p in processed_results if p.get("imagePlaces", [])]
            print(f"Returning {len(final_results)} places with images")

            return final_results

    async def async_fetch_place_details(self, place_id, session):
        """Fetch individual place details asynchronously. Checks DB first."""
        if not place_id:
            print(f"Error: Place ID is required for fetch_place_details")
            return None

        min_images = 5  # Define minimum required images
        db_place = None

        place_data = {}

        async def save_or_update_db(data, existing_db_place):
            try:
                lat_val = data.get("gps_coordinates", {}).get("latitude")  # Prefer gps_coordinates if available
                lon_val = data.get("gps_coordinates", {}).get("longitude")
                if lat_val is None or lon_val is None:  # Fallback to location if gps_coordinates missing
                    lat_val = data.get("location", {}).get("latitude")
                    lon_val = data.get("location", {}).get("longitude")

                save_func = sync_to_async(lambda p: p.save())
                update_fields = []

                if existing_db_place:
                    # Update existing record
                    print(f"Updating existing database record for {place_id}")
                    place_to_save = existing_db_place
                    # Update fields selectively if they are better/more complete from API
                    if data.get("title") and not place_to_save.title: place_to_save.title = data.get("title"); update_fields.append('title')
                    if data.get("description") and not place_to_save.description: place_to_save.description = data.get("description"); update_fields.append('description')
                    if data.get("address") and not place_to_save.address: place_to_save.address = data.get("address"); update_fields.append('address')
                    if data.get("rating") and (place_to_save.rating is None or data.get("rating") > place_to_save.rating): place_to_save.rating = data.get("rating"); update_fields.append('rating')  # Example: update rating if API's is higher
                    # Add category field if it exists in data
                    if data.get("category") and not place_to_save.category: place_to_save.category = data.get("category"); update_fields.append('category')
                    # Add more fields as needed
                    place_to_save.place_images = data.get("place_images", [])
                    update_fields.append('place_images')

                    # Update popular_times if present in API data and missing in DB
                    if data.get("popular_times") and not place_to_save.popular_times:
                        place_to_save.popular_times = data.get("popular_times")
                        update_fields.append('popular_times')

                    await save_func(place_to_save)
                    # Optionally use update_fields with save: await sync_to_async(place_to_save.save)(update_fields=update_fields)
                else:
                    # Create new record
                    print(f"Creating new database record for {place_id}")
                    place_to_save = Place(
                        title=data.get("title"),
                        description=data.get("description", ""),
                        place_id=place_id,
                        data_id=data.get("data_id"),
                        reviews_link=data.get("reviews_link"),
                        photos_link=data.get("photos_link"),
                        latitude=lat_val,
                        longitude=lon_val,
                        type=data.get("type"),
                        types=data.get("types", []),
                        hours=data.get("hours", data.get("operating_hours", {})),
                        address=data.get("address"),
                        extensions=data.get("extensions", {}),
                        display_name=(data.get("displayName", {}).get("text") or data.get("title")),
                        formatted_address=data.get("formattedAddress"),
                        rating=data.get("rating"),
                        reviews=data.get("reviews") or data.get("userRatingCount"),
                        place_images=data.get("place_images", []),
                        reviews_list=data.get("reviews_list", []),
                        popular_times=data.get("popular_times", []),
                        category=data.get("category", "")
                    )
                    await save_func(place_to_save)
                    print(f"Saved new place details to database: {place_id}")

            except Exception as e:
                print(f"Error saving/updating place details {place_id} to database: {e}")

        # 1. Check Database First
        try:
            get_place_from_db = sync_to_async(Place.objects.get)
            db_place = await get_place_from_db(place_id=place_id)
            print(f"Found place details in database for ID: {place_id}")

            place_data = {
                "place_id": db_place.place_id,
                "title": db_place.title,
                "description": db_place.description,
                "data_id": db_place.data_id,
                "reviews_link": db_place.reviews_link,
                "photos_link": db_place.photos_link,
                "location": {
                    "latitude": float(db_place.latitude) if db_place.latitude is not None else None,
                    "longitude": float(db_place.longitude) if db_place.longitude is not None else None
                },
                "gps_coordinates": {
                    "latitude": float(db_place.latitude) if db_place.latitude is not None else None,
                    "longitude": float(db_place.longitude) if db_place.longitude is not None else None
                },
                "hours": db_place.hours,
                "type": db_place.type,
                "types": db_place.types,
                "address": db_place.address,
                "extensions": db_place.extensions,
                "displayName": {"text": db_place.display_name} if db_place.display_name else None,
                "formattedAddress": db_place.formatted_address,
                "rating": db_place.rating,
                "reviews": db_place.reviews,
                "operating_hours": db_place.hours,
                "place_images": db_place.place_images or [],  # Return existing images
                "reviews_list": db_place.reviews_list or [],
                "popular_times": db_place.popular_times,
                "category": db_place.category,  # Include category in response
                "source": "database"
            }

            # Check if images meet requirements
            if db_place.place_images and len(db_place.place_images) >= min_images and db_place.reviews_list and db_place.popular_times and db_place.description:
                print(f"Database entry for {place_id} has sufficient images. Returning cached data.")
                # Convert DB model to dict format expected by frontend
                return place_data
            else:
                print(f"Database entry for {place_id} found but needs images.")
                import concurrent.futures
                # Initialize place_data with default values
                place_data['place_images'] = place_data.get('place_images', [])
                place_data['reviews_list'] = place_data.get('reviews_list', [])
                place_data['popular_times'] = place_data.get('popular_times', [])

                # Define functions to run conditionally based on what we need
                def fetch_images():
                    if not db_place.place_images or not len(db_place.place_images) >= min_images:
                        print(f"Fetching images for {place_id}")
                        return process_images_for_place(place_data, min_images=min_images)
                    return place_data.get('place_images', [])

                def fetch_reviews():
                    if not db_place.reviews_list:
                        print(f"Fetching reviews for {place_id}")
                        return get_reviews_from_serp(place_data.get('place_id'))
                    return place_data.get('reviews_list', [])

                def fetch_api_data():
                    # Only fetch from API if we need description or popular_times
                    if not place_data.get('description') or not place_data.get('popular_times'):
                        print(f"Fetching additional details from SerpAPI for place ID: {place_id}")
                        params = {
                            "engine": "google_maps",
                            "api_key": SERP_API_KEY,
                            "place_id": place_id,
                        }
                        try:
                            results = serpapi.search(params).get('place_results', {})
                            return {
                                'description': results.get('description'),
                                'popular_times': results.get('popular_times')
                            }
                        except Exception as e:
                            print(f"Error in API fetch: {e}")
                            return {}
                    return {}

                # Execute the tasks in parallel
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    # Submit tasks that need to be executed
                    futures = []

                    # Only submit tasks if their conditions are met
                    futures.append(executor.submit(fetch_images))
                    futures.append(executor.submit(fetch_reviews))
                    futures.append(executor.submit(fetch_api_data))

                    # Process results as they complete
                    for future in concurrent.futures.as_completed(futures):
                        try:
                            result = future.result()
                            if isinstance(result, list) and result and isinstance(result[0], str) and '://storage.googleapis.com/' in result[0]:
                                # This is likely the images result
                                place_data['place_images'] = result
                            elif isinstance(result, list) and result and isinstance(result[0], dict) and 'rating' in result[0]:
                                # This is likely the reviews result
                                place_data['reviews_list'] = result
                            elif isinstance(result, dict):

                                # This is likely the API result
                                if result.get('description'):
                                    place_data['description'] = result.get('description')
                                else:
                                    place_data['description'] = 'No description available'

                                if result.get('popular_times'):
                                    place_data['popular_times'] = result.get('popular_times')
                        except Exception as e:
                            print(f"Error processing parallel task result: {e}")

                #if place_data.get('description') and place_data.get('popular_times'):

                # Save to database synchronously to ensure images are persisted
                await save_or_update_db(place_data, db_place)
                return place_data

        except ObjectDoesNotExist:
            print(f"Place details not found in database for ID: {place_id}. Fetching from API.")
        except Exception as e:
            print(f"Error checking database for place details {place_id}: {e}. Proceeding to API call.")

        # 2. Fetch from SerpAPI if not in DB or needs images
        print(f"Fetching details from SerpAPI for place ID: {place_id}")
        params = {
            "engine": "google_maps",
            "api_key": SERP_API_KEY,
            "place_id": place_id,
        }

        try:
            # Note: serpapi.search is synchronous, consider running in executor if it blocks the event loop
            # For now, assuming it's acceptable or handled within the library
            api_results = serpapi.search(params).get('place_results', {})
            if not api_results:
                print(f"No results found from SerpAPI for place ID: {place_id}")
                #return None

            print(f"Successfully fetched details from SerpAPI for place ID {place_id}")

            # 4. Save/Update Database Asynchronously
            # If we have place_data from DB, update it with API results
            if place_data:
                # Update description if missing
                if not place_data.get('description') and api_results.get('description'):
                    place_data['description'] = api_results.get('description')

                # Update popular_times if missing
                if not place_data.get('popular_times') and api_results.get('popular_times'):
                    place_data['popular_times'] = api_results.get('popular_times')

                # Keep existing images and reviews in API results for database update
                api_results["place_images"] = place_data.get("place_images", [])
                api_results['reviews_list'] = place_data.get('reviews_list', [])

                # Save/update DB with combined data synchronously to ensure images are persisted
                await save_or_update_db(place_data, db_place)

                # Return the updated place_data
                place_data['source'] = "database_updated"
                return place_data
            else:
                # If no existing place_data, use API results directly
                api_results["source"] = "api"

                # Save/update DB with API results synchronously to ensure images are persisted
                await save_or_update_db(api_results, db_place)

                return api_results

        except Exception as e:
            print(f"Error during SerpAPI fetch or processing for {place_id}: {e}")
            # Decide what to return on API error. Maybe DB data if available but lacks images?
            if db_place: # If we had DB data but API failed
                 print(f"API failed for {place_id}, returning stale DB data.")
                 # Convert DB model to dict and return (images might be missing/incomplete)
                 place_data = { # Duplicating conversion logic, consider a helper function
                     "place_id": db_place.place_id,
                     "title": db_place.title,
                     "description": db_place.description, # etc. ... fill all fields
                     "place_images": db_place.place_images or [],
                     "reviews_list": db_place.reviews_list or [],
                     "popular_times": db_place.popular_times,
                     "category": db_place.category,  # Include category in response
                     "source": "database_stale"
                 }
                 return place_data
            return None # Return None if no data source succeeded

    def get(self, request, place_id=None, is_authenticated=False):
        """Endpoint to fetch place details by ID."""
        if not place_id:
            return Response({"error": "Place ID is required"}, status=HTTP_400_BAD_REQUEST)

        print('request query params', request.query_params)

        async def fetch_details():
            # We don't need a full session here if async_fetch_place_details doesn't use it anymore
            # However, keeping session creation for consistency if other methods need it
            async with aiohttp.ClientSession() as session:
                return await self.async_fetch_place_details(place_id, session)

        place_details = asyncio.run(fetch_details())

        if not place_details:
            return Response({"error": "Place not found"}, status=HTTP_404_NOT_FOUND)

        print(f"Returning place details for {place_id} from GET endpoint. Source: {place_details.get('source', 'unknown')}")

        return Response(place_details, status=HTTP_200_OK)

    async def async_fetch_all_places(self, params, session, max_results=60, queries=None):
        """Fetches all places asynchronously using the new Places API v2."""

        radius = params["radius"]
        location = params["location"]

        print(f"Radius: {radius}, Location: {location}")

        # Convert radius to zoom level
        # Significantly adjusted conversion formula for tighter results
        if radius <= 100:
            zoom_level = 21  # Extremely detailed view
        elif radius <= 250:
            zoom_level = 21
        elif radius <= 500:
            zoom_level = 21
        elif radius <= 1000:
            zoom_level = 21
        elif radius <= 2000:
            zoom_level = 21
        elif radius <= 5000:
            zoom_level = 21
        elif radius <= 8000:  # ~5 miles
            zoom_level = 21
        elif radius <= 16000:  # ~10 miles
            zoom_level = 20
        else:
            zoom_level = 19

        print(f"Zoom level: {zoom_level}, Radius: {radius}, Location: {location}")

        # Base parameters for the search
        base_params = {
            "api_key": SERP_API_KEY,
            "engine": "google_maps",
            "type": "search",
            "google_domain": "google.com",
            "ll": f"@{location},{zoom_level}z",
            "hl": "en"
        }

        start_val = 1 if params.get("load_more", False) else 0

        def fetch_page_sync(query_str, category_val=None, start_val=start_val):
            page_params_sync = base_params.copy()
            if start_val > 0:
                page_params_sync["start"] = str(start_val)

            print(f"Fetching page with query: {query_str}")
            page_params_sync["q"] = query_str

            try:
                # Direct blocking call to serpapi
                search_result_sync = serpapi.search(page_params_sync)
                results_sync = search_result_sync.get('local_results', [])

                if category_val:
                    for res_item in results_sync:
                        res_item['category'] = category_val

                return results_sync
            except Exception as e_sync:
                print(f"Error fetching results with params {page_params_sync}: {e_sync}")
                return []

        print(f"Queries: {queries}")
        
        
        query_to_category = {}
        if not queries:
            types = params["types"]
            if len(types.keys()) == 1:
                type_list = list(types.values())[0]
                # Ensure type_group_size is at least 1 to avoid issues with small lists
                type_group_size = len(type_list) // 3 if len(type_list) // 3 > 0 else 1
                types_groups = [type_list[i:i+type_group_size] for i in range(0, len(type_list), type_group_size)]
                queries = [f"{' OR '.join(group)}" for group in types_groups if group] # Ensure group is not empty
                for query_item in queries:
                    query_to_category[query_item] = list(types.keys())[0]
            else:
                queries = []
                max_queries_per_type = 10 #This can be adjusted
                for category_item, curr_types in types.items():
                    # Ensure curr_types is not empty before join
                    if curr_types:
                        new_query = f"{' OR '.join(curr_types[:max_queries_per_type])}"
                        query_to_category[new_query] = category_item
                        queries.append(new_query)

        print(f"{len(queries)} queries: {queries}")

        start_time = time.time()
        all_api_results = []

        if queries:
            num_workers = min(len(queries), 5)

            with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
                
                if query_to_category:
                    futures = {
                        executor.submit(fetch_page_sync, q, category_val=query_to_category[q]): q 
                        for q in queries
                    }
                else:
                    futures = {
                        executor.submit(fetch_page_sync, q): q 
                        for q in queries
                    }
                
                # Process results as they complete
                for future in concurrent.futures.as_completed(futures):
                    query = futures[future]
                    try:
                        # Get the result of this specific future
                        result_list = future.result()
                        if result_list:  # Ensure it's not None or empty from an error
                            all_api_results.extend(result_list)
                            print(f"Received {len(result_list)} results for query: {query[:30]}...")
                    except Exception as exc:
                        print(f"Query {query[:30]}... generated an exception: {exc}")
        
        end_time = time.time()
        print(f"Time taken to fetch all pages: {end_time - start_time} seconds")
        
        unique_results = {}
        for result in all_api_results:
            place_id = result.get("place_id")
            if place_id and place_id not in unique_results:
                unique_results[place_id] = result

        # Remove duplicates based on place ID
        unique_results = list(unique_results.values())
        
        print(f"Total unique results after pagination: {len(unique_results)}")

        if TESTING:
            unique_results = unique_results[:15]

        def distance_between(lat1, lon1, lat2_lon2, lon2=None):
            """Calculate distance between two coordinates using haversine formula"""
            if lon2 is None:
                # If lat2_lon2 is a string like "lat,lon"
                if isinstance(lat2_lon2, str) and ',' in lat2_lon2:
                    lat2, lon2 = map(float, lat2_lon2.split(','))
                else:
                    return 0  # Invalid input
            else:
                lat2 = lat2_lon2  # If all parameters provided individually

            # Haversine formula
            R = 3958.8  # Radius of the Earth in miles
            lat1, lon1, lat2, lon2 = map(math.radians, [float(lat1), float(lon1), float(lat2), float(lon2)])
            dlat = lat2 - lat1
            dlon = lon2 - lon1
            a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
            c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
            distance = R * c
            return round(distance, 2)

        # List to store filtered results
        filtered_results = []

        num_results_filtered_out = 0
        print(f"Unique results: {len(unique_results)}")
        # Format results to match prev structure for now
        for result in unique_results:
            try:
                result['location'] = result['gps_coordinates']

                # Check if result is within radius
                lat, lon = result['gps_coordinates']['latitude'], result['gps_coordinates']['longitude']
                distance_miles = distance_between(lat, lon, location)
                result['distance'] = distance_miles  # Add distance to result for reference
                
                # Convert radius from meters to miles for comparison
                radius_miles = radius / 1609.34  # 1609.34 meters in a mile
                
                if distance_miles > radius_miles:
                    print(f"Result {result['title']} is outside radius {radius_miles} miles")
                    num_results_filtered_out += 1
                    continue
                    
                result['displayName'] = {
                    'text': result['title'],
                }
                result['formattedAddress'] = result.get('address', 'no address found')
                result['userRatingCount'] = result.get('reviews', 0)
                filtered_results.append(result)
            except Exception as e:
                print(f"Error processing result {result}: {e}")
            
        print(f"Num results filtered out: {num_results_filtered_out}")
                
        return filtered_results[:max_results]

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
        filtered_places = await self.fetch_place_images(filtered_places)

        for place in filtered_places:
            image_places = place.get("imagePlaces", [])

            if not image_places:
                place["imagePlaces"] = []

            data = {'image_urls': image_places, 'place_id': place["id"]}

            try:
                response = requests.post(IMAGE_DOWNLOAD_URL, json=data)
                if response.status_code == 200:
                    print(f"Image download successful for place ID {place['id']}")
                    place["imagePlaces"] = response.json().get("gcs_image_urls", [])
                else:
                    print(f"Image download failed for place ID {place['id']}: {response.status_code}")
                    place["imagePlacesGCS"] = []
            except Exception as e:
                print(f"Error during image download for place ID {place['id']}: {e}")
                place["imagePlaces"] = []

        return filtered_places

    async def fetch_place_images(self, places, max_images=3):
        """Fetch up to 5 photos for each place and add imagePlaces key."""
        for place in places:
            photos = place.get("photos", [])[:max_images]  # Limit to first 3 photos
            image_places = []

            print(f"num distinct photos: {len(set([p['name'] for p in photos]))}")
            for photo in photos:
                name = photo.get("name")
                if name:
                    image_url = f"https://places.googleapis.com/v1/{name}/media?key={self.API_KEY}&maxHeightPx=600"
                    #proxied_image_url = f"/api/serve-image?url={google_image_url}"  # Proxy through your backend
                    image_places.append(image_url)

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

    def check_rate_limit(self, request, max_requests, cache_timeout=None):
        """
        Enforces rate limiting for unauthenticated requests.
        
        Args:
            request: The Django request object.
            max_requests: The maximum number of allowed requests.
            cache_timeout: The timeout for the cache key (in seconds). If None, the key will not expire.
        
        Returns:
            Response: A 429 response if the rate limit is exceeded, otherwise None.
        """
        ip_address = self.get_client_ip(request)

        country_cache_key = f"ip_country_{ip_address}"
        country = cache.get(country_cache_key)
        if not country:
            try:
                url = f"https://ipinfo.io/json?token={IP_INFO_API_TOKEN}"
                response = requests.get(url)
                print(f"IP info response: {response.json()}")
                if response.status_code == 200:
                    data = response.json()
                    country = data.get("country", "Unknown")
                    print(f"Country for IP {ip_address}: {country}")
                    cache.set(country_cache_key, country, timeout=cache_timeout)
                else:
                    # If API call fails, assume allowed to avoid blocking legitimate users
                    country = 'US'
            except Exception as e:
                print(f"Error checking IP country for {ip_address}: {e}")
                country = 'US'
        else:
            print(f"Cached country for IP {ip_address}: {country}")

        # Block requests not from the US
        if country != 'US':
            return Response(
                {"error": "Access restricted to US-based users only."},
                status=status.HTTP_403_FORBIDDEN
            )

        cache_key = f"places_api_requests_{ip_address}"
        request_count = cache.get(cache_key, 0)

        if request_count > max_requests:
            return Response(
                {"error": "Request limit reached for unauthenticated users. Please sign in to continue."},
                status=HTTP_429_TOO_MANY_REQUESTS
            )
        
        # Increment count and set cache with the specified timeout
        cache.set(cache_key, request_count + 1, timeout=cache_timeout)
        print(f"Unauthenticated request from IP {ip_address}. Count: {request_count + 1}")
        return None

class ImageDownloadAPIView(APIView):
    permission_classes = [AllowAny]
    parser_classes = [JSONParser]

    def post(self, request):
        """Endpoint to download images."""
        image_urls = request.data.get("image_urls")
        place_id = request.data.get("place_id")
        if not image_urls:
            response = Response({'error': 'No image URLs provided'}, status=HTTP_400_BAD_REQUEST)
            response['Access-Control-Allow-Origin'] = '*'
            response['Access-Control-Allow-Methods'] = 'POST, OPTIONS'
            response['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
            return response

        local_image_paths = []
        os.makedirs("temp_images", exist_ok=True)

        try:
            # Check for existing images
            existing_image_links = check_existing_images(bucket_name='nurtr-places', place_id=place_id)
            if existing_image_links:
                print(f"Returning existing images for place ID {place_id}")
                response = Response({'gcs_image_urls': existing_image_links}, status=HTTP_200_OK)
            else:
                print(f"Downloading images for place ID {place_id}...")
                local_image_paths = []
                with ThreadPoolExecutor(max_workers=5) as executor:  # Adjust max_workers as needed
                    results = executor.map(download_image, image_urls)
                    local_image_paths.extend(filter(None, results))  # Filter out None values from failed downloads

                gcs_place_image_links = upload_place_images_to_bucket(
                    bucket_name='nurtr-places',
                    place_id=place_id,
                    image_paths=local_image_paths
                )

                response = Response({'gcs_image_urls': gcs_place_image_links}, status=HTTP_200_OK)

        except Exception as e:
            print(f"Error downloading images for place ID {place_id}: {e}")
            traceback.print_exc()
            response = Response({'error': str(e)}, status=HTTP_500_INTERNAL_SERVER_ERROR)

        # Add CORS headers to all responses
        response['Access-Control-Allow-Origin'] = '*'
        response['Access-Control-Allow-Methods'] = 'POST, OPTIONS'
        response['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
        return response

def serve_image(request, image_url):
    """Proxy image from Google's server."""
    response = requests.get(image_url, stream=True)
    return HttpResponse(response.content, content_type=response.headers['Content-Type'])

class EventsAPIView(APIView):
    permission_classes = [AllowAny]
    parser_classes = [JSONParser]

    def get_client_ip(self, request):
        """Utility function to get client IP address."""
        x_forwarded_for = request.META.get('HTTP_X_FORWARDED_FOR')
        if x_forwarded_for:
            ip = x_forwarded_for.split(',')[0]
        else:
            ip = request.META.get('REMOTE_ADDR')
        return ip

    async def get_coordinates_from_zip(self, zip_code):
        """Fetch latitude and longitude from ZIP code using Geocoding API."""
        url = f"https://maps.googleapis.com/maps/api/geocode/json?address={zip_code}+USA&key=AIzaSyBfW8nU2EoPK1Zg_bYOSREzqmRDwZfUgbM"

        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                data = await response.json()

        if data.get("results"):
            location = data["results"][0]["geometry"]["location"]
            return location["lat"], location["lng"]

        return None, None

    def get_city_from_coordinates(self, latitude, longitude):
        """Get city name from coordinates using reverse geocoding."""
        try:
            url = f"https://maps.googleapis.com/maps/api/geocode/json?latlng={latitude},{longitude}&key=AIzaSyBfW8nU2EoPK1Zg_bYOSREzqmRDwZfUgbM"
            response = requests.get(url)
            data = response.json()
            
            if data.get("results"):
                for component in data["results"][0]["address_components"]:
                    if "locality" in component["types"]:
                        return component["long_name"]
                    elif "administrative_area_level_1" in component["types"]:
                        return component["long_name"]
            return "Unknown Location"
        except Exception as e:
            print(f"Error getting city from coordinates: {e}")
            return "Unknown Location"
    
    async def get_city_state_and_coordinates_from_zip(self, zip_code):
        """Fetch city, state, and coordinates from ZIP code using Geocoding API."""
        url = f"https://maps.googleapis.com/maps/api/geocode/json?address={zip_code}+USA&key=AIzaSyBfW8nU2EoPK1Zg_bYOSREzqmRDwZfUgbM"

        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                data = await response.json()

        if data.get("results") and len(data["results"]) > 0:
            result = data["results"][0]
            
            # Extract coordinates from geometry
            location = result.get("geometry", {}).get("location", {})
            latitude = location.get("lat")
            longitude = location.get("lng")
            
            # Extract city and state from address_components
            city = None
            state = None
            
            address_components = result.get("address_components", [])
            for component in address_components:
                types = component.get("types", [])
                
                # Get city from locality
                if "locality" in types:
                    city = component.get("long_name")
                
                # Get state abbreviation from administrative_area_level_1
                elif "administrative_area_level_1" in types:
                    state = component.get("long_name")
            
            return city, state, latitude, longitude

        return None, None, None, None

    def calculate_distance(self, lat1, lon1, lat2, lon2):
        """Calculate distance between two coordinates using haversine formula."""
        if None in [lat1, lon1, lat2, lon2]:
            return None
        
        R = 3958.8  # Radius of the Earth in miles
        lat1, lon1, lat2, lon2 = map(math.radians, [float(lat1), float(lon1), float(lat2), float(lon2)])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        distance = R * c
        return round(distance, 2)

    def extract_coordinates_from_address(self, address_list):
        """Extract coordinates from address using geocoding."""
        if not address_list:
            return None, None
        
        # Join address components
        full_address = ", ".join(address_list)
        
        try:
            url = f"https://maps.googleapis.com/maps/api/geocode/json?address={full_address}&key=AIzaSyBfW8nU2EoPK1Zg_bYOSREzqmRDwZfUgbM"
            response = requests.get(url)
            data = response.json()
            
            if data.get("results"):
                location = data["results"][0]["geometry"]["location"]
                return location["lat"], location["lng"]
        except Exception as e:
            print(f"Error geocoding address {full_address}: {e}")
        
        return None, None

    def post(self, request):
        start_time = time.time()
        
        print('Events request:', request.data)
        print('User is authenticated:', request.data.get('is_authenticated', False))

        # Rate limiting for unauthenticated users
        if False and not request.data.get('is_authenticated', False): # NOT ENABLED FOR NOW
            rate_limit_response = self.check_rate_limit(
                request,
                MAX_UNAUTHENTICATED_REQUESTS_PER_IP,
                CACHE_TIMEOUT_SECONDS
            )
            if rate_limit_response:
                return rate_limit_response

        data = request.data

        print(f"request data: {data}")
        zip_code = data.get("zip_code")
        
        city_and_state = data.get("city_and_state")
        latitude = data.get("latitude", 40.712776)
        longitude = data.get("longitude", -74.005974)

        # Convert radius to float to ensure proper comparison
        try:
            radius = float(data.get("radius", 25))  # Default radius in miles
        except (ValueError, TypeError):
            radius = 25.0  # Fallback to default if conversion fails
        
        event_types = data.get("event_types", [])  # Array of event types to filter
        print(f"event_types: {event_types}")
        date_range = data.get("date_range", "this_month")  # this_week, this_month, next_month
        
        try:
            page = int(data.get("page", 1))
        except (ValueError, TypeError):
            page = 1

        # Get coordinates from zip code if provided
        city_and_state = ""
        if zip_code:
            city, state, latitude, longitude = asyncio.run(self.get_city_state_and_coordinates_from_zip(zip_code))
            if city and state:
                city_and_state = f"{city}, {state}"
        
        # Run asynchronous search for events
        events_results = asyncio.run(self.async_search_events(city_and_state, event_types, date_range, page))

        # Process and filter events by distance
        processed_events = []
        for event in events_results:
            # Extract coordinates from event address
            event_lat, event_lon = self.extract_coordinates_from_address(event.get("address", []))
            
            if event_lat and event_lon:
                # Calculate distance from search location
                distance = self.calculate_distance(latitude, longitude, event_lat, event_lon)
                
                # Filter by radius (now both are guaranteed to be numbers)
                if distance and distance <= radius:
                    event["distance"] = distance
                    event["event_latitude"] = event_lat
                    event["event_longitude"] = event_lon
                    processed_events.append(event)
                    print(f"Event {event.get('title', 'Unknown')} is within the radius of {radius} miles")
                else:
                    print(f"Event {event.get('title', 'Unknown')} is outside the radius of {radius} miles")
                    print(f"Distance: {distance}")
                    print(f"Latitude: {event_lat}")
                    print(f"Longitude: {event_lon}")
                    print(f"Radius: {radius}")
            else:
                # If we can't get coordinates, include the event but mark distance as unknown
                event["distance"] = None
                event["event_latitude"] = None
                event["event_longitude"] = None
                processed_events.append(event)

        # Sort by distance (events with unknown distance go to the end)
        processed_events.sort(key=lambda x: x.get("distance") if x.get("distance") is not None else float('inf'))

        # Limit results for pagination
        results_per_page = 20
        start_index = (page - 1) * results_per_page
        end_index = start_index + results_per_page
        paginated_events = processed_events[start_index:end_index]

        execution_time = round((time.time() - start_time) * 1000, 2)
        
        return Response({
            "results": paginated_events,
            "page": page,
            "total_results": len(processed_events),
            "execution_time_ms": execution_time,
            "search_location": {
                "city_and_state": city_and_state,
                "latitude": latitude,
                "longitude": longitude
            }
        }, status=HTTP_200_OK)

    async def async_search_events_2(self, city_and_state, event_types, date_range, page):
        """Search for events using SERP API with parallel execution."""
        print(f"Searching for events in {city_and_state}")
        
        try:
            # Build optimized search queries
            search_queries = self.build_kid_friendly_search_queries(event_types, city_and_state)[:3]
            
            print(f"Executing {len(search_queries)} searches in parallel...")
            
            # Create a function to execute a single search query
            def execute_single_search(query):
                """Execute a single search query - runs in thread pool."""
                try:
                    print(f"Executing search query: {query}")
                    
                    params = {
                        "engine": "google_events",
                        "q": query,
                        "api_key": SERP_API_KEY,
                        "hl": "en",
                        "gl": "us"
                    }

                    if city_and_state:
                        params["location"] = f"{city_and_state}, United States"

                    # Add date filtering
                    if date_range and isinstance(date_range, list):
                        date_filters = [f"date:{date_item}" for date_item in date_range]
                        params["htichips"] = ",".join(date_filters)

                    print(f"SERP API params for '{query}': {params}")

                    start_time = time.time()
                    search_results = serpapi.search(params)
                    end_time = time.time()

                    print(f"SERP API call for '{query}' took {end_time - start_time:.2f} seconds")

                    # Extract events from results
                    events = search_results.get('events_results', [])
                    print(f"Found {len(events)} events for query: {query}")

                    # Process events for this query
                    processed_events = []
                    for event in events:
                        processed_event = self.process_event_data(event)
                        if processed_event:
                            processed_events.append(processed_event)

                    return {
                        'query': query,
                        'events': processed_events,
                        'success': True,
                        'execution_time': end_time - start_time
                    }
                    
                except Exception as e:
                    print(f"Error with query '{query}': {e}")
                    return {
                        'query': query,
                        'events': [],
                        'success': False,
                        'error': str(e)
                    }

            # Execute all searches in parallel
            overall_start_time = time.time()
            all_events = []
            
            # Use ThreadPoolExecutor to run searches concurrently
            with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(search_queries), 5)) as executor:
                # Submit all search tasks
                future_to_query = {
                    executor.submit(execute_single_search, query): query 
                    for query in search_queries
                }
                
                # Process results as they complete
                for future in concurrent.futures.as_completed(future_to_query):
                    query = future_to_query[future]
                    try:
                        result = future.result()
                        
                        if result['success']:
                            all_events.extend(result['events'])
                            print(f"Successfully processed {len(result['events'])} events from query: '{query}' "
                                  f"(took {result['execution_time']:.2f}s)")
                        else:
                            print(f"Failed to process query '{query}': {result.get('error', 'Unknown error')}")
                            
                    except Exception as exc:
                        print(f"Query '{query}' generated an exception: {exc}")

            overall_end_time = time.time()
            total_execution_time = overall_end_time - overall_start_time
            
            print(f"Parallel execution completed in {total_execution_time:.2f} seconds")
            print(f"Total events collected from all queries: {len(all_events)}")
            
            #if all_events:
             #   print(f"Sample event: {all_events[0]}")
            print(f"search_queries: {search_queries}")
            
            # Remove duplicates based on event ID
            unique_events = []
            seen_events = set()
            
            for event in all_events:
                # Create a unique identifier based on event ID
                event_key = event.get('id', '')
                
                if event_key and event_key not in seen_events:
                    seen_events.add(event_key)
                    unique_events.append(event)

            print(f"Total unique events found: {len(unique_events)} (from {len(all_events)} total)")
            
            return unique_events

        except Exception as e:
            print(f"Error in parallel search execution: {e}")
            traceback.print_exc()
            return []

    def process_event_data(self, event):
        """Process and standardize event data from SERP API."""

        # Create a more stable unique identifier
        title = event.get('title', '').strip()
        venue_name = event.get('venue', {}).get('name', '').strip()
        start_date = event.get('date', {}).get('start_date', '').strip()
        when_time = event.get('date', {}).get('when', '').strip()
        
        # Use address as fallback if venue name is missing or inconsistent
        address_list = event.get('address', [])
        venue_identifier = venue_name
        if not venue_identifier and address_list:
            # Use first address component as venue identifier
            venue_identifier = address_list[0].strip()
        
        # Normalize the when field to remove timezone variations and extra whitespace
        normalized_when = ' '.join(when_time.split()) if when_time else ''
        # Remove timezone abbreviations for consistency (PDT, PST, EDT, EST, etc.)
        normalized_when = normalized_when.replace(' PDT', '').replace(' PST', '').replace(' EDT', '').replace(' EST', '')
        
        # Create a more stable identifier using venue + date + normalized title
        # This helps avoid duplicates from the same event at the same venue/time
        id_components = [
            title.lower().replace(' ', '_'),
            venue_identifier.lower().replace(' ', '_'),
            start_date.replace(' ', '_'),
            normalized_when.lower().replace(' ', '_').replace(',', '')
        ]
        
        # Filter out empty components and create a stable hash
        stable_components = [comp for comp in id_components if comp]
        stable_string = '_'.join(stable_components)
        
        # Use a more deterministic approach - you could also use a UUID based on content
        import hashlib
        event_id = f"event_{hashlib.md5(stable_string.encode()).hexdigest()[:16]}"
        
        try:
            # Extract and standardize event information
            processed_event = {
                "id": event_id,
                "title": event.get("title", ""),
                "description": event.get("description", ""),
                "date": event.get("date", {}),
                "address": event.get("address", []),
                "link": event.get("link", ""),
                "venue": event.get("venue", {}),
                "ticket_info": event.get("ticket_info", []),
                "thumbnail": event.get("thumbnail", ""),
                "image": event.get("image", ""),
                "event_location_map": event.get("event_location_map", {}),
                
                # Additional processed fields
                "formatted_address": ", ".join(event.get("address", [])),
                "venue_name": event.get("venue", {}).get("name", ""),
                "venue_rating": event.get("venue", {}).get("rating"),
                "venue_reviews": event.get("venue", {}).get("reviews"),
                
                # Date processing
                "start_date": event.get("date", {}).get("start_date", ""),
                "when": event.get("date", {}).get("when", ""),
                
                # Ticket availability
                "has_tickets": len(event.get("ticket_info", [])) > 0,
                "ticket_sources": [ticket.get("source") for ticket in event.get("ticket_info", [])],
                
                # Event type classification (can be enhanced)
                "event_type": self.classify_event_type(event.get("title", "") + " " + event.get("description", "")),
                
                # Source information
                "source": "serp_api",
                "last_updated": datetime.now().isoformat()
            }

            return processed_event

        except Exception as e:
            print(f"Error processing event data: {e}")
            return None

    def classify_event_type(self, text):
        """Classify event type based on title and description."""
        text_lower = text.lower()
        
        # Define event type keywords
        event_types = {
            "music": ["concert", "music", "band", "festival", "live music", "performance", "show"],
            "sports": ["game", "match", "tournament", "sports", "football", "basketball", "baseball", "soccer"],
            "family": ["family", "kids", "children", "playground", "zoo", "museum"],
            "food": ["food", "restaurant", "dining", "taste", "culinary", "cooking"],
            "arts": ["art", "gallery", "exhibition", "theater", "dance", "cultural"],
            "education": ["workshop", "class", "seminar", "conference", "learning", "training"],
            "outdoor": ["outdoor", "park", "hiking", "nature", "camping", "adventure"],
            "entertainment": ["comedy", "movie", "film", "entertainment", "fun"]
        }
        
        # Check which category the event belongs to
        for event_type, keywords in event_types.items():
            if any(keyword in text_lower for keyword in keywords):
                return event_type
        
        return "general"

    def check_rate_limit(self, request, max_requests, cache_timeout=None):
        """
        Enforces rate limiting for unauthenticated requests.
        Uses the same logic as PlacesAPIView.
        """
        ip_address = self.get_client_ip(request)

        country_cache_key = f"ip_country_{ip_address}"
        country = cache.get(country_cache_key)
        if not country:
            try:
                url = f"https://ipinfo.io/json?token={IP_INFO_API_TOKEN}"
                response = requests.get(url)
                if response.status_code == 200:
                    data = response.json()
                    country = data.get("country", "Unknown")
                    cache.set(country_cache_key, country, timeout=cache_timeout)
                else:
                    country = 'US'
            except Exception as e:
                print(f"Error checking IP country for {ip_address}: {e}")
                country = 'US'

        # Block requests not from the US
        if country != 'US':
            return Response(
                {"error": "Access restricted to US-based users only."},
                status=status.HTTP_403_FORBIDDEN
            )

        cache_key = f"events_api_requests_{ip_address}"
        request_count = cache.get(cache_key, 0)

        if request_count > max_requests:
            return Response(
                {"error": "Request limit reached for unauthenticated users. Please sign in to continue."},
                status=HTTP_429_TOO_MANY_REQUESTS
            )
        
        cache.set(cache_key, request_count + 1, timeout=cache_timeout)
        print(f"Unauthenticated events request from IP {ip_address}. Count: {request_count + 1}")
        return None


    def build_kid_friendly_search_queries(self, event_types, city_and_state):
        """
        Build optimized search queries for kid-friendly events based on selected categories.
        Uses broader, more general terms to get more results.
        """
        
        # Define category-specific keywords - simplified and more general
        category_keywords = {
            "music_classes": [
                "kids music",
                "family music",
                "children music classes"
            ],
            "art_classes": [
                "kids art",
                "family art", 
                "children art classes"
            ],
            "outdoor": [
                "kids outdoor",
                "family outdoor activities",
                "children nature"
            ],
            "indoor": [
                "kids indoor",
                "family fun center",
                "children indoor activities"
            ],
            "seasonal_play": [
                "kids seasonal",
                "family holiday events",
                "children activities"
            ],
            "museums": [
                "kids museum",
                "family museum",
                "children museum"
            ],
            "aquariums": [
                "kids aquarium",
                "family aquarium", 
                "children aquarium"
            ],
            "water_parks": [
                "kids water park",
                "family water activities",
                "children swimming"
            ]
        }
        
        queries = []
        
        # If no specific categories are selected, create general kid-friendly queries
        if not event_types:
            base_queries = [
                f"kids activities near {city_and_state}",
                f"family events near {city_and_state}",
                f"children activities near {city_and_state}"
            ]
            return base_queries
        
        # Convert event_types to lowercase and replace spaces with underscores for mapping
        normalized_types = [t.lower().replace(" ", "_") for t in event_types]
        
        # For each category, create 3 search variations
        for event_type in normalized_types:
            if event_type in category_keywords:
                # Use all 3 variations for each category
                for keyword_phrase in category_keywords[event_type]:
                    queries.append(f"{keyword_phrase} near {city_and_state}")
        
        # If we don't have enough queries, add some general fallbacks
        if len(queries) < 3:
            fallback_queries = [
                f"family events near {city_and_state}",
                f"kids activities near {city_and_state}",
                f"children programs near {city_and_state}"
            ]
            
            # Add fallbacks to reach at least 3 queries
            for fallback in fallback_queries:
                if fallback not in queries and len(queries) < 6:
                    queries.append(fallback)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_queries = []
        for query in queries:
            if query not in seen:
                seen.add(query)
                unique_queries.append(query)
        
        # Limit to maximum 6 queries and randomize
        final_queries = unique_queries[:6]
        random.shuffle(final_queries)
        
        print(f"Generated {len(final_queries)} search queries for event types {event_types}:")
        for i, query in enumerate(final_queries, 1):
            print(f"  {i}. {query}")
        
        return final_queries

    # Add a new method to filter kid-friendly events
    def is_kid_friendly_event(self, event):
        """
        Check if an event is kid-friendly based on title and description.
        
        Args:
            event (dict): Event data
        
        Returns:
            bool: True if event appears to be kid-friendly
        """
        
        # Kid-friendly keywords to look for
        kid_friendly_keywords = [
            'kids', 'children', 'child', 'family', 'toddler', 'infant', 
            'preschool', 'kid-friendly', 'family-friendly', 'child-friendly',
            'little ones', 'young children', 'babies', 'youth', 'junior',
            'all ages', 'ages', 'elementary', 'kindergarten'
        ]
        
        # Adult-only or inappropriate keywords to exclude
        exclude_keywords = [
            'adults only', 'adult', '21+', '18+', 'mature', 'wine', 'beer', 
            'cocktail', 'bar', 'nightclub', 'dating', 'singles', 'romantic',
            'bachelor', 'bachelorette', 'casino', 'gambling'
        ]
        
        # Get text to analyze
        title = event.get('title', '').lower()
        description = event.get('description', '').lower()
        venue_name = event.get('venue_name', '').lower()
        
        # Combine all text
        full_text = f"{title} {description} {venue_name}"
        
        # Check for exclude keywords first
        for exclude_word in exclude_keywords:
            if exclude_word in full_text:
                return False
        
        # Check for kid-friendly keywords
        for keyword in kid_friendly_keywords:
            if keyword in full_text:
                return True
        
        # Special cases - venues that are typically kid-friendly
        kid_friendly_venues = [
            'museum', 'aquarium', 'zoo', 'park', 'library', 'community center',
            'recreation center', 'ymca', 'playground', 'nature center'
        ]
        
        for venue_type in kid_friendly_venues:
            if venue_type in full_text:
                return True
        
        #print(f"event: {event}")
        #print(f"full_text: {full_text}")
        # If no explicit kid-friendly indicators found, be conservative
        return False

    # Update the async_search_events method to include filtering
    async def async_search_events(self, city_and_state, event_types, date_range, page):
        """Search for events using SERP API with parallel execution and kid-friendly filtering."""
        print(f"Searching for events in {city_and_state}")
        
        try:
            # Build optimized search queries
            search_queries = self.build_kid_friendly_search_queries(event_types, city_and_state)[:3]
            
            print(f"Executing {len(search_queries)} searches in parallel...")
            
            # Create a function to execute a single search query
            def execute_single_search(query):
                """Execute a single search query - runs in thread pool."""
                try:
                    #print(f"Executing search query: {query}")
                    
                    params = {
                        "engine": "google_events",
                        "q": query,
                        "api_key": SERP_API_KEY,
                        "hl": "en",
                        "gl": "us"
                    }
                    
                    #print(f"params: {params}")
                    #if city_and_state:
                     #   params["location"] = f"{city_and_state}, United States"

                    # Add date filtering
                    if date_range and isinstance(date_range, list):
                        date_filters = [f"date:{date_item}" for date_item in date_range]
                        params["htichips"] = ",".join(date_filters)

                    start_time = time.time()
                    search_results = serpapi.search(params)
                    end_time = time.time()

                    print(f"SERP API call for '{query}' took {end_time - start_time:.2f} seconds")

                    # Extract events from results
                    events = search_results.get('events_results', [])
                    #print(f"Found {len(events)} events for query: {query}")

                    # Process events for this query
                    processed_events = []
                    kid_friendly_events = []
                    
                    for event in events:
                        processed_event = self.process_event_data(event)
                        if processed_event:
                            processed_events.append(processed_event)
                            
                            # Filter for kid-friendly events
                            if self.is_kid_friendly_event(processed_event):
                                kid_friendly_events.append(processed_event)

                    print(f"Filtered to {len(kid_friendly_events)} kid-friendly events from {len(processed_events)} total")

                    return {
                        'query': query,
                        'events': kid_friendly_events,  # Return only kid-friendly events
                        'total_found': len(processed_events),
                        'kid_friendly_found': len(kid_friendly_events),
                        'success': True,
                        'execution_time': end_time - start_time
                    }
                    
                except Exception as e:
                    print(f"Error with query '{query}': {e}")
                    return {
                        'query': query,
                        'events': [],
                        'total_found': 0,
                        'kid_friendly_found': 0,
                        'success': False,
                        'error': str(e)
                    }

            # Execute all searches in parallel
            overall_start_time = time.time()
            all_events = []
            total_found = 0
            total_kid_friendly = 0
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(search_queries), 5)) as executor:
                # Submit all search tasks
                future_to_query = {
                    executor.submit(execute_single_search, query): query 
                    for query in search_queries
                }
                
                # Process results as they complete
                for future in concurrent.futures.as_completed(future_to_query):
                    query = future_to_query[future]
                    try:
                        result = future.result()
                        
                        if result['success']:
                            all_events.extend(result['events'])
                            total_found += result['total_found']
                            total_kid_friendly += result['kid_friendly_found']
                            
                            print(f"Query '{query}': {result['kid_friendly_found']}/{result['total_found']} "
                                  f"kid-friendly events (took {result['execution_time']:.2f}s)")
                        else:
                            print(f"Failed to process query '{query}': {result.get('error', 'Unknown error')}")
                            
                    except Exception as exc:
                        print(f"Query '{query}' generated an exception: {exc}")

            overall_end_time = time.time()
            total_execution_time = overall_end_time - overall_start_time
            
            print(f"Parallel execution completed in {total_execution_time:.2f} seconds")
            print(f"Total events found: {total_found}, Kid-friendly: {total_kid_friendly}")
            
            # Remove duplicates based on event ID
            unique_events = []
            seen_events = set()
            
            for event in all_events:
                event_key = event.get('id', '')
                
                if event_key and event_key not in seen_events:
                    seen_events.add(event_key)
                    unique_events.append(event)

            print(f"Total unique kid-friendly events: {len(unique_events)} (from {len(all_events)} total)")
            
            return unique_events

        except Exception as e:
            print(f"Error in parallel search execution: {e}")
            traceback.print_exc()
            return []

class RecommendedPlacesAPIView(APIView):
    """
    Personalized place recommendations endpoint for authenticated users.
    
    - GET: Retrieve user's latest recommended places (for frontend display)
    - POST: Generate/refresh personalized recommendations for a user
    """
    permission_classes = [IsAuthenticated]  # Simplified - always require authentication
    parser_classes = [JSONParser]

    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

    def get(self, request):
        """
        Get the latest recommended places for the authenticated user.
        This endpoint is designed for frontend consumption - returns full place data
        for the user's most recent daily recommendations.
        """
        print(f"🔍 GET /places/recommended/ called by user: {request.user.email} (ID: {request.user.id})")
        
        if not request.user.is_authenticated:
            print(f"❌ User not authenticated")
            return Response({"error": "Authentication required"}, status=401)
        
        return self._get_user_latest_recommendations(request.user)
    
    def _get_user_latest_recommendations(self, user):
        """Get the latest recommended places for an authenticated user."""
        start_time = time.time()
        print(f"🔍 _get_user_latest_recommendations called for user: {user.email} (ID: {user.id})")
        
        try:
            # Check if user has any children first (recommendations are based on children)
            from users.models import Child
            children = Child.objects.filter(user=user)
            print(f"👶 User has {children.count()} children: {[child.name for child in children]}")
            
            # Get total count of UserRecommendation objects in database
            total_recommendations = UserRecommendation.objects.count()
            print(f"📊 Total UserRecommendation objects in database: {total_recommendations}")
            
            # Get user's latest recommendation
            user_recommendations = UserRecommendation.objects.filter(user=user)
            print(f"📊 UserRecommendation objects for this user: {user_recommendations.count()}")
            
            latest_recommendation = user_recommendations.order_by('-date_generated').first()
            print(f"📅 Latest recommendation for user: {latest_recommendation}")
            
            if not latest_recommendation:
                print(f"❌ No recommendations found for user {user.email}")
                return Response({
                    "message": "No recommendations found. Please make a POST request to generate recommendations.",
                    "results": [],
                    "total_results": 0
                }, status=HTTP_200_OK)
            
            # Get all recommended places with full data
            from .models import UserRecommendationPlace
            recommended_places = []
            recommendation_places = UserRecommendationPlace.objects.filter(
                user_recommendation=latest_recommendation
            ).select_related('place').order_by('recommendation_rank')
            
            print(f"✅ Found recommendation from {latest_recommendation.date_generated} with {recommendation_places.count()} places")
            
            for rec_place in recommendation_places:
                place = rec_place.place
                
                # Build comprehensive place data
                place_data = {
                    # Core place information
                    "place_id": place.place_id,
                    "title": place.title,
                    "name": place.title,  # For compatibility
                    "description": place.description or "",
                    "category": place.category or "",
                    
                    # Location data
                    "location": {
                        "latitude": float(place.latitude) if place.latitude else None,
                        "longitude": float(place.longitude) if place.longitude else None
                    },
                    "gps_coordinates": {
                        "latitude": float(place.latitude) if place.latitude else None,
                        "longitude": float(place.longitude) if place.longitude else None
                    },
                    "address": place.address or "",
                    "formattedAddress": place.formatted_address or "",
                    "vicinity": place.formatted_address or "",  # For compatibility
                    
                    # Rating and review data
                    "rating": place.rating,
                    "reviews": place.reviews,
                    "userRatingCount": place.reviews,  # For compatibility
                    
                    # Place metadata
                    "types": place.types or [],
                    "type": place.type or "",
                    "extensions": place.extensions or {},
                    "displayName": {"text": place.title} if place.title else None,
                    
                    # Visual content
                    "imagePlaces": place.place_images or [],
                    "place_images": place.place_images or [],  # For compatibility
                    
                    # Operating information
                    "operating_hours": place.hours or {},
                    "hours": place.hours or {},  # For compatibility
                    "popular_times": place.popular_times or [],
                    
                    # Reviews data
                    "reviews_list": place.reviews_list or [],
                    "reviews_link": place.reviews_link or "",
                    "photos_link": place.photos_link or "",
                    
                    # Recommendation-specific data
                    "personalization_score": rec_place.personalization_score,
                    "personalized_explanation": rec_place.personalized_explanation,
                    "searched_location": rec_place.searched_location,
                    "source_query": rec_place.source_query,
                    "recommendation_rank": rec_place.recommendation_rank,
                    
                    # Metadata
                    "data_id": place.data_id or "",
                    "source": "database_recommendation"
                }
                recommended_places.append(place_data)
            
            execution_time = round((time.time() - start_time) * 1000, 2)
            
            return Response({
                "results": recommended_places,
                "total_results": len(recommended_places),
                "recommendation_metadata": {
                    "date_generated": latest_recommendation.date_generated.isoformat(),
                    "custom_queries": latest_recommendation.custom_queries,
                    "locations_searched": latest_recommendation.locations_searched,
                    "user_profile_hash": latest_recommendation.user_profile_hash,
                    "created_at": latest_recommendation.created_at.isoformat(),
                    "updated_at": latest_recommendation.updated_at.isoformat()
                },
                "execution_time_ms": execution_time,
                "cached": True,
                "message": f"Latest recommendations from {latest_recommendation.date_generated}"
            }, status=HTTP_200_OK)
            
        except Exception as e:
            print(f"Error getting latest recommendations for {user.email}: {e}")
            import traceback
            traceback.print_exc()
            return Response({"error": str(e)}, status=HTTP_500_INTERNAL_SERVER_ERROR)

    def post(self, request):
        """
        Get personalized recommendations for the authenticated user.
        Returns cached results if available and fresh, otherwise generates new ones.
        """
        start_time = time.time()
        
        try:
            user = request.user
            print(f"Getting recommendations for user: {user.email}")
            
            # Get user's children and their data
            children = Child.objects.filter(user=user)

            print(f"Children: {children}")
            
            # Get location data from request
            data = request.data
            zip_code = data.get("zip_code")
            latitude = data.get("latitude", 40.712776)
            longitude = data.get("longitude", -74.005974)
            radius = data.get("radius", 500)
            force_refresh = data.get("force_refresh", False)
            is_homebase = data.get("is_homebase", False)  # Flag to indicate this is a homebase update
            
            # Get coordinates from zip if provided
            if zip_code:
                latitude, longitude = asyncio.run(self.get_coordinates_from_zip(zip_code))
                if not latitude or not longitude:
                    return Response({"error": "Invalid ZIP code"}, status=HTTP_400_BAD_REQUEST)
            
            # Update user's homebase if this is a homebase update
            if is_homebase and zip_code:
                try:
                    user.homebaseZipCode = zip_code
                    user.save()
                    print(f"✅ Updated homebase for {user.email} to {zip_code}")
                except Exception as e:
                    print(f"⚠️ Failed to update homebase for {user.email}: {e}")
                    # Don't fail the entire request if homebase update fails
            
            # Add this location to user's search history for future recommendations
            location_key = zip_code or f"{latitude},{longitude}"
            self.add_location_to_search_history(user, location_key)
            
            print(f"Zip code: {zip_code}")
            print(f"Latitude: {latitude}")
            print(f"Longitude: {longitude}")
            print(f"Radius: {radius}")
            print(f"Force refresh: {force_refresh}")
            
            # Check for today's cached recommendations first (unless force_refresh is True)
            if not force_refresh:
                cached_recommendation = self.get_cached_recommendations(user)
                if cached_recommendation:
                    execution_time = round((time.time() - start_time) * 1000, 2)
                    print(f"✅ Returning today's cached recommendations for {user.email}")
                    
                    # Convert the related places back to the expected JSON format
                    from .models import UserRecommendationPlace
                    recommended_places = []
                    recommendation_places = UserRecommendationPlace.objects.filter(
                        user_recommendation=cached_recommendation
                    ).select_related('place').order_by('recommendation_rank')
                    
                    for rec_place in recommendation_places:
                        place = rec_place.place
                        place_data = {
                            "place_id": place.place_id,
                            "title": place.title,
                            "name": place.title,  # For compatibility
                            "description": place.description,
                            "rating": place.rating,
                            "reviews": place.reviews,
                            "formattedAddress": place.formatted_address,
                            "vicinity": place.formatted_address,  # For compatibility
                            "types": place.types,
                            "imagePlaces": place.place_images,
                            "category": place.category,
                            "location": {
                                "latitude": float(place.latitude) if place.latitude else None,
                                "longitude": float(place.longitude) if place.longitude else None
                            },
                            "gps_coordinates": {
                                "latitude": float(place.latitude) if place.latitude else None,
                                "longitude": float(place.longitude) if place.longitude else None
                            },
                            # Recommendation-specific data
                            "personalization_score": rec_place.personalization_score,
                            "personalized_explanation": rec_place.personalized_explanation,
                            "searched_location": rec_place.searched_location,
                            "source_query": rec_place.source_query,
                            "recommendation_rank": rec_place.recommendation_rank
                        }
                        recommended_places.append(place_data)
                    
                    return Response({
                        "results": recommended_places,
                        "custom_queries": cached_recommendation.custom_queries,
                        "locations_searched": cached_recommendation.locations_searched,
                        "total_results": cached_recommendation.total_results,
                        "execution_time_ms": execution_time,
                        "cached": True,
                        "date_generated": cached_recommendation.date_generated.isoformat(),
                        "cache_updated": cached_recommendation.updated_at.isoformat()
                    }, status=HTTP_200_OK)
            
            # Generate fresh daily recommendations
            recommendation_data = self.generate_and_cache_daily_recommendations(user)
            
            execution_time = round((time.time() - start_time) * 1000, 2)
            recommendation_data["execution_time_ms"] = execution_time
            recommendation_data["cached"] = False
            
            return Response(recommendation_data, status=HTTP_200_OK)
            
        except Exception as e:
            print(f"Error in recommendations: {e}")
            import traceback
            traceback.print_exc()
            return Response({"error": str(e)}, status=HTTP_500_INTERNAL_SERVER_ERROR)

    def build_user_profile(self, user, children):
        """Build a comprehensive user profile for LLM context."""
        profile = {
            "user_name": user.name or "Parent",
            "email": user.email,
            "favorites": json.loads(user.favorites) if user.favorites else {},
            "children": []
        }
        
        for child in children:
            child_data = {
                "name": child.name,
                "age": child.age,
                "interests": child.interests or [],
                "age_group": self.get_age_group(child.age)
            }
            profile["children"].append(child_data)
        
        # Add derived insights
        profile["total_children"] = len(children)
        profile["age_ranges"] = list(set([self.get_age_group(child.age) for child in children]))
        profile["all_interests"] = list(set([interest for child in children for interest in (child.interests or [])]))
        
        return profile

    def get_age_group(self, age):
        """Categorize children by age group for better recommendations."""
        if age <= 2:
            return "toddler"
        elif age <= 5:
            return "preschooler"
        elif age <= 8:
            return "early_elementary"
        elif age <= 12:
            return "late_elementary"
        else:
            return "preteen"

    def generate_llm_queries(self, user_profile, location):
        """Use Gemini to generate personalized search queries based on user profile."""
        try:
            client = genai.Client(api_key=self.GEMINI_API_KEY)
            
            prompt = f"""
            You are a family activity recommendation expert. Based on the user profile below, generate 5-7 specific search queries to use for Google Maps search to find the best family places and activities.

            User Profile:
            - Parent: {user_profile['user_name']}
            - Number of children: {user_profile['total_children']}
            - Children details: {json.dumps(user_profile['children'], indent=2)}
            - Age groups represented: {', '.join(user_profile['age_ranges'])}
            - Combined interests: {', '.join(user_profile['all_interests'])}

            Guidelines:
            1. Create queries that match the children's ages and interests
            2. Consider safety and age-appropriateness
            3. Include both indoor and outdoor options
            4. Mix popular activities with unique local experiences
            5. Each query should be 2-5 words
            6. Focus on kid-friendly venues that welcome families

            Return only a JSON array of search query strings, like:
            ["family indoor play center", "toddler-friendly museums", "outdoor playgrounds"]
            """

            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
            )

            #print(f"Gemini response: {response}")

            # Parse Gemini response
            llm_output = response.text.strip()
            #print(f"Gemini raw output: {llm_output}")
            
            # Try to extract JSON array from response
            try:
                # Sometimes Gemini includes markdown formatting, so let's clean it
                if "```json" in llm_output:
                    llm_output = llm_output.split("```json")[1].split("```")[0].strip()
                elif "```" in llm_output:
                    llm_output = llm_output.split("```")[1].strip()
                
                custom_queries = json.loads(llm_output)
                if isinstance(custom_queries, list):
                    return custom_queries[:7]  # Limit to 7 queries
            except json.JSONDecodeError:
                print("Failed to parse Gemini JSON, using fallback")
            
        except Exception as e:
            print(f"Gemini generation failed: {e}")
        
        # Fallback queries if LLM fails
        return self.generate_fallback_queries(user_profile)

    def generate_fallback_queries(self, user_profile):
        """Generate fallback queries based on user profile without LLM."""
        queries = []
        interests = user_profile['all_interests']
        age_groups = user_profile['age_ranges']
        
        # Base queries for different interests
        interest_mapping = {
            "Parks": "family parks playgrounds",
            "Nature": "nature centers outdoor kids",
            "Animals": "petting zoos aquariums kids",
            "Creative": "kids art classes creative",
            "Sports": "family sports activities kids",
            "STEM": "science museums kids STEM",
            "Food": "family restaurants kids menu",
            "Play Centers": "indoor play centers kids",
            "Playgrounds": "playgrounds family outdoor"
        }
        
        # Add interest-based queries
        for interest in interests[:4]:  # Limit to top 4 interests
            if interest in interest_mapping:
                queries.append(interest_mapping[interest])
        
        # Add age-appropriate queries
        if "toddler" in age_groups:
            queries.append("toddler friendly activities")
        if "preschooler" in age_groups:
            queries.append("preschool kids activities")
        if any(age in age_groups for age in ["early_elementary", "late_elementary"]):
            queries.append("elementary kids activities")
        
        return queries[:6] if queries else ["family activities kids", "playgrounds parks", "kids museums"]

    async def get_recommended_places(self, custom_queries, latitude, longitude, radius, max_queries=7):
        """Use existing places search infrastructure with custom queries."""
        all_places = []
        
        print(f"Getting recommended places for {custom_queries} with radius {radius} at {latitude},{longitude}")

        # Convert custom queries into the format expected by existing search
            
        search_params = {
            "location": f"{latitude},{longitude}",
            "radius": radius,
            "key": PlacesAPIView.API_KEY
        }

        # Limit to 3 queries
        #curr_query = ' OR '.join(custom_queries[:3])
            
        try:
            places_api = PlacesAPIView()
            async with aiohttp.ClientSession() as session:
                results = await places_api.async_fetch_all_places(
                    search_params,
                    session,
                    max_results=15,
                    queries=custom_queries[:max_queries]
                )
                
            # Process images for each place using the existing pipeline
            if results:
                print(f"Processing images for {len(results)} places from recommendations...")
                from concurrent.futures import ThreadPoolExecutor, as_completed
                
                with ThreadPoolExecutor(max_workers=20) as executor:
                    future_to_place = {
                        executor.submit(process_images_for_place, place): place
                        for place in results
                    }
                    
                    for future in as_completed(future_to_place):
                        place = future_to_place[future]
                        try:
                            processed_urls = future.result()
                            if processed_urls:
                                place["imagePlaces"] = processed_urls
                                print(f"Processed {len(processed_urls)} images for place {place.get('place_id', 'unknown')}")
                            else:
                                place["imagePlaces"] = []
                        except Exception as e:
                            print(f"Error processing images for place {place.get('place_id', 'unknown')}: {e}")
                            place["imagePlaces"] = []
                
            for place in results:
                #place["source_query"] = curr_query
                all_places.append(place)

        except Exception as e:
            print(f"Error searching with queries {custom_queries}: {e}")
        
        # Remove duplicates based on place_id
        unique_places = []
        seen_ids = set()
        for place in all_places:
            place_id = place.get("place_id")
            if place_id and place_id not in seen_ids:
                seen_ids.add(place_id)
                unique_places.append(place)
        
        return unique_places

    def score_places_for_user(self, places, user_profile):
        """Add personalization scores and explanations using Gemini AI based on how well places match user profile."""
        if not places:
            return places

        try:
            print(f"Scoring {len(places)} places for user profile: {user_profile}")
            gemini_api_key = os.environ.get('GEMINI_API_KEY')
            if not gemini_api_key:
                raise Exception("Missing GEMINI_API_KEY environment variable")

            client = genai.Client(api_key=gemini_api_key)

            # Process places in batches to avoid token limits
            batch_size = 50
            for i in range(0, len(places), batch_size):
                batch = places[i:i + batch_size]

                # Prepare batch data for Gemini
                places_data = []
                for idx, place in enumerate(batch):
                    place_info = {
                        "index": idx + 1,
                        "name": place.get('title', 'Unknown'),
                        "description": place.get('description', ''),
                        "category": place.get('category', ''),
                        "types": place.get('types', []),
                        "rating": place.get('rating', 0),
                        "reviews": place.get('reviews', 0) or place.get('userRatingCount', 0),
                        "address": place.get('formattedAddress', place.get('address', ''))
                    }
                    places_data.append(place_info)

                # Create the prompt with individual child profiles
                child_count = user_profile.get('total_children', 1)
                child_plural = "children" if child_count > 1 else "child"
                
                # Format individual child profiles (without names)
                child_profiles = []
                for i, child in enumerate(user_profile.get('children', []), 1):
                    child_profile = f"""
                Child #{i}:
                - Age: {child.get('age', 'Unknown')} years old ({child.get('age_group', 'Unknown')} stage)
                - Primary Interests: {', '.join(child.get('interests', []))}
                - Developmental Stage: {child.get('age_group', 'Unknown')}"""
                    child_profiles.append(child_profile)
                
                children_section = '\n'.join(child_profiles) if child_profiles else "No children data available"
                
                prompt = f"""
                You are a family activity recommendation expert. Analyze these places and provide personalization scores (70-100) and engaging explanations for why each place would be perfect for this parent and their children. IMPORTANT: Only return recommendations for the TOP 5 highest-scoring places.
            
                FAMILY PROFILE:
                - Parent: {user_profile.get('user_name', 'Parent')}
                - Total Children: {child_count}
                
                INDIVIDUAL CHILDREN PROFILES:
                {children_section}
                
                COMBINED FAMILY DATA:
                - All Age Groups: {', '.join(user_profile.get('age_ranges', []))}
                - All Interests Combined: {', '.join(user_profile.get('all_interests', []))}
            
                PLACES TO ANALYZE:
                {json.dumps(places_data, indent=2)}
            
                SCORING CRITERIA (70-100 scale):
                
                **BASELINE SCORE: 70** (Every place starts at 70 - assumes basic family-friendliness)
                
                **PRIMARY SCORING FACTORS (+30 points available):**
                1. **Interest Match (0-20 points)**: How well does this place align with the children's specific interests?
                   - Perfect match (multiple interests): +15-20 points
                   - Good match (1-2 interests): +10-14 points
                   - Moderate match (related interests): +5-9 points
                   - Poor match (no clear alignment): +0-4 points
                
                2. **Age Appropriateness (0-10 points)**: How suitable is this place for the children's ages and developmental stages?
                   - Perfect for all children's ages: +8-10 points
                   - Good for most children: +5-7 points
                   - Suitable but not ideal: +2-4 points
                   - Age concerns or limitations: +0-1 points
                
                **EXAMPLE SCORING:**
                - STEM child + Science Museum = 70 baseline + 18 interest match + 9 age appropriate = 97
                - Sports child + Indoor Playground = 70 baseline + 12 interest match + 8 age appropriate = 90
                - Creative child + Art Studio = 70 baseline + 17 interest match + 10 age appropriate = 97
                - Any child + Generic restaurant = 70 baseline + 2 interest match + 5 age appropriate = 77
                
                **SELECTION PROCESS:**
                1. Score ALL places using the criteria above
                2. Rank them by score (highest to lowest)
                3. ONLY return the TOP 5 highest-scoring places in your response
                4. If there are fewer than 5 places, return all of them IN ORDER BY RANKING
            
                EXPLANATION GUIDELINES:
                - Write directly to the parent using "you", "your little ones", "your tiny explorers"
                - Use fun, endearing terms: "little adventurers", "tiny explorers", "little scientists", "mini athletes", "creative spirits", "curious minds"
                - DO NOT mention children's actual names - only use descriptive terms
                - Focus on what the children will experience, enjoy, and benefit from
                - Keep tone enthusiastic, playful, and child-focused
                - Use exciting action words: "zoom", "splash", "discover", "create", "explore", "adventure"
                
                EXAMPLE EXPLANATIONS (keep these as strong examples):
                - Indoor Playground: "Your little adventurers will absolutely love this vibrant indoor playground! With climbing structures perfectly sized for early elementary explorers, they'll spend hours zooming through tunnels, conquering obstacle courses, and making new friends in this safe, magical play wonderland."
                - Science Museum: "This hands-on science museum is perfect for your curious little scientists! Your tiny explorers will be amazed by interactive exhibits where they can conduct real experiments, touch actual fossils, and discover how the world works through exciting play-based adventures."
                - Adventure Park: "Your active little athletes will have an absolute blast at this outdoor adventure park! These mini daredevils will challenge themselves on age-appropriate zip lines, navigate thrilling obstacle courses, and build confidence while soaking up sunshine and fresh air fun."
            
                Respond in this exact JSON format with ONLY the top 5 places:
                {{
                  "recommendations": [
                    {{
                      "index": 2,
                      "score": 87,
                      "explanation": "Your little adventurers will absolutely love this vibrant indoor playground! With climbing structures perfectly sized for early elementary explorers, they'll spend hours zooming through tunnels and making new friends in this magical play wonderland."
                    }}
                  ]
                }}
                EACH AND EVERY PLACE IN THE TOP 5 MUST HAVE A PERSONALIZED EXPLANATION AND A PERSONALIZATION SCORE. The index of each place in the top 5 MUST BE THE EXACT INDEX OF THE PLACE IN THE ORIGINAL LIST OF PLACES.
                """


                try:
                    response = client.models.generate_content(
                        model="gemini-2.5-flash",
                        contents=prompt,
                        config={
                            "response_mime_type": "application/json",
                            "response_schema": PlaceRecommendations,
                        },
                    )

                    print(f"Gemini response: {response}")

                    # Use structured output - response.parsed contains the validated data
                    result = response.parsed
                    
                    # Apply scores and explanations to places
                    # Note: Gemini returns only top 5 places, so we need to map indices correctly
                    for recommendation in result.recommendations:
                        place_index = recommendation.index
                        actual_index = place_index - 1  # Adjust for batch offset

                        if 0 <= actual_index < len(batch):
                            batch[actual_index]["personalization_score"] = recommendation.score
                            batch[actual_index]["personalized_explanation"] = recommendation.explanation
                        else:
                            print(f"Warning: Recommendation index {place_index} (actual: {actual_index}) out of bounds for batch size {len(batch)} at offset {i}")

                except Exception as e:
                    print(f"Error with Gemini API for batch {i // batch_size + 1}: {e}")
                    # Fallback to simple scoring for this batch
                    for place in batch:
                        fallback_score = self._calculate_fallback_score(place, user_profile)
                        place["personalization_score"] = fallback_score
                        place["personalized_explanation"] = self._generate_fallback_explanation(place, user_profile)

                # Small delay between batches to respect rate limits
                if i + batch_size < len(places):
                    time.sleep(0.5)

        except Exception as e:
            print(f"Error initializing Gemini for place scoring: {e}")
            # Fallback to simple scoring for all places
            for place in places:
                fallback_score = self._calculate_fallback_score(place, user_profile)
                place["personalization_score"] = fallback_score
                place["personalized_explanation"] = self._generate_fallback_explanation(place, user_profile)

        # Sort by personalization score and return top 5
        places.sort(key=lambda x: x.get("personalization_score", 0), reverse=True)
        return places[:5]

    def _calculate_fallback_score(self, place, user_profile):
        """Fallback scoring method when Gemini is unavailable."""
        score = 0

        # Score based on children's interests
        place_text = f"{place.get('title', '')} {place.get('description', '')} {place.get('category', '')}".lower()

        for interest in user_profile['all_interests']:
            if interest.lower() in place_text:
                score += 10

        # Score based on age appropriateness
        age_keywords = {
            "toddler": ["toddler", "baby", "infant", "crawl"],
            "preschooler": ["preschool", "4 year", "5 year"],
            "early_elementary": ["elementary", "school age"],
            "late_elementary": ["kids", "children", "youth"]
        }

        for age_group in user_profile['age_ranges']:
            if age_group in age_keywords:
                for keyword in age_keywords[age_group]:
                    if keyword in place_text:
                        score += 5

        # Boost score for highly rated places
        rating = place.get('rating', 0)
        if rating >= 4.5:
            score += 15
        elif rating >= 4.0:
            score += 10
        elif rating >= 3.5:
            score += 5

        # Boost score for places with reviews (more established)
        review_count = place.get('reviews', 0) or place.get('userRatingCount', 0)
        if review_count > 100:
            score += 10
        elif review_count > 50:
            score += 5

        return min(score, 100)  # Cap at 100

    def _generate_fallback_explanation(self, place, user_profile):
        """Generate a simple explanation when Gemini is unavailable."""
        place_name = place.get('title', 'This place')
        interests = user_profile.get('all_interests', [])
        age_ranges = user_profile.get('age_ranges', [])

        if 'Parks' in interests or 'Nature' in interests:
            return f"{place_name} offers wonderful outdoor experiences perfect for your family's love of nature and active play!"
        elif 'Art' in interests or 'Music' in interests:
            return f"{place_name} provides creative and cultural experiences that will inspire your family's artistic interests!"
        elif 'STEM' in interests:
            return f"{place_name} offers educational and hands-on learning opportunities perfect for curious young minds!"
        else:
            return f"{place_name} is a great family-friendly destination that offers fun activities for children of all ages!"

    def get_user_locations(self, user):
        """Get up to 5 locations for recommendations (preferences first, then search history)."""
        all_locations = []
        
        # First, add location preferences (priority)
        if user.location_preferences:
            all_locations.extend(user.location_preferences)
        
        # Then add search history locations
        if user.search_history_locations:
            for location in user.search_history_locations:
                if location not in all_locations:  # Avoid duplicates
                    all_locations.append(location)
        
        # Limit to 5 total locations (only using current homebase for now)
        locations_to_use = all_locations[:1]
        
        # If user has no locations, use defaults
        #if not locations_to_use:
         #   locations_to_use = ["90210", "10001", "94102"]  # Default cities
        
        # Convert to the expected format with coordinates
        location_data = {}
        default_coords = {}
        
        for location in locations_to_use:
            if location in default_coords:
                location_data[location] = default_coords[location]
            else:
                # For unknown zip codes, try to geocode them
                try:
                    lat, lng = asyncio.run(self.get_coordinates_from_zip(location))
                    if lat and lng:
                        location_data[location] = {"latitude": lat, "longitude": lng}
                except Exception as e:
                    print(f"Error geocoding {location}: {e}")
                    continue
        
        print(f"Using {len(location_data)} locations for {user.email}: {list(location_data.keys())}")
        return location_data

    def get_cached_recommendations(self, user):
        """Get today's cached recommendations if they exist and are fresh."""
        try:
            today = datetime.now().date()
            
            cached = UserRecommendation.objects.filter(
                user=user,
                date_generated=today
            ).first()
            
            if cached:
                # Verify the user profile hasn't changed by comparing hash
                children = Child.objects.filter(user=user)
                current_profile = self.build_user_profile(user, children)
                current_hash = self.get_profile_hash(current_profile)
                
                if cached.user_profile_hash == current_hash:
                    return cached
                else:
                    print(f"User profile changed for {user.email}, invalidating today's cache")
                    cached.delete()
            
            return None
            
        except Exception as e:
            print(f"Error checking cache for {user.email}: {e}")
            return None

    def get_profile_hash(self, user_profile):
        """Generate a hash of the user profile for cache invalidation."""
        # Create a stable string representation of the profile
        profile_string = json.dumps(user_profile, sort_keys=True)
        return hashlib.md5(profile_string.encode()).hexdigest()

    def generate_and_cache_daily_recommendations(self, user):
        """Generate daily aggregated recommendations from user's preferred and search history locations."""
        try:
            # Get user's children and their data
            children = Child.objects.filter(user=user)
            
            # Build user profile for LLM
            user_profile = self.build_user_profile(user, children)
            print(f"User profile for {user.email}: {user_profile}")
            
            # Get locations to search (preferences + history, max 5)
            locations_to_process = self.get_user_locations(user)
            
            all_places = []
            all_custom_queries = []
            locations_searched = list(locations_to_process.keys())
            
            print(f"Locations to process: {locations_to_process}")
            

            # Search across all user's locations
            for location_key, location_data in locations_to_process.items():
                try:
                    latitude = location_data['latitude']
                    longitude = location_data['longitude']
                    
                    # Generate custom search queries using LLM for this location
                    location_queries = self.generate_llm_queries(user_profile, location_key)
                    print(f"Location queries: {location_queries}")

                    all_custom_queries.extend(location_queries)
                    
                    # Get places for this location
                    recommended_places = asyncio.run(self.get_recommended_places(
                        location_queries, latitude, longitude, 20000
                    ))
                    
                    # Add location info to each place
                    for place in recommended_places:
                        place['searched_location'] = location_key
                    
                    all_places.extend(recommended_places)
                    print(f"Found {len(recommended_places)} places for {location_key}")
                    
                except Exception as e:
                    print(f"Error searching location {location_key} for {user.email}: {e}")
                    continue
            
            # Remove duplicates based on place_id
            unique_places = []
            seen_ids = set()
            for place in all_places:
                place_id = place.get("place_id")
                if place_id and place_id not in seen_ids:
                    seen_ids.add(place_id)
                    unique_places.append(place)

            print(f"Unique places: {len(unique_places)}")
            

            start_time = time.time()

            # Add personalization scores
            scored_places = self.score_places_for_user(unique_places, user_profile)

            end_time = time.time()
            print(f"Personalization took {end_time - start_time} seconds")
            
            # Sort by personalization score
            scored_places.sort(key=lambda x: x.get("personalization_score", 0), reverse=True)
            
            # Limit to top 15 for daily recommendations
            daily_recommendations = scored_places[:15]
            
            # Cache the results for today
            today = datetime.now().date()
            profile_hash = self.get_profile_hash(user_profile)
            
            print(f"Updating or creating today's recommendation for {user.email}")
            
            # Update or create today's recommendation
            cached_recommendation, created = UserRecommendation.objects.update_or_create(
                user=user,
                date_generated=today,
                defaults={
                    'user_profile_hash': profile_hash,
                    'custom_queries': list(set(all_custom_queries)),  # Remove duplicates
                    'locations_searched': locations_searched,
                    'total_results': len(scored_places)
                }
            )
            
            # Clear existing places for this recommendation (in case of update)
            if not created:
                cached_recommendation.recommended_places.clear()
            
            # Create Place objects and UserRecommendationPlace relationships
            from .models import UserRecommendationPlace
            for rank, place_data in enumerate(daily_recommendations, 1):
                try:
                    print(f"Place title and location: {place_data.get('title', place_data.get('name', ''))} {place_data.get('location', {}).get('latitude') or place_data.get('gps_coordinates', {}).get('latitude')} {place_data.get('location', {}).get('longitude') or place_data.get('gps_coordinates', {}).get('longitude')}")

                    print(f"place score explanation: {place_data.get('personalized_explanation', '')}")

                    # Get or create the Place object
                    place, place_created = Place.objects.get_or_create(
                        place_id=place_data.get('place_id'),
                        defaults={
                            'title': place_data.get('title', place_data.get('name', '')),
                            'description': place_data.get('description', ''),
                            'latitude': place_data.get('location', {}).get('latitude') or place_data.get('gps_coordinates', {}).get('latitude'),
                            'longitude': place_data.get('location', {}).get('longitude') or place_data.get('gps_coordinates', {}).get('longitude'),
                            'rating': place_data.get('rating'),
                            'reviews': place_data.get('reviews') or place_data.get('userRatingCount'),
                            'formatted_address': place_data.get('formattedAddress', place_data.get('vicinity', '')),
                            'types': place_data.get('types', []),
                            'place_images': place_data.get('imagePlaces', []),
                            'category': place_data.get('category', ''),
                        }
                    )
                    
                    # Create the recommendation relationship
                    UserRecommendationPlace.objects.create(
                        user_recommendation=cached_recommendation,
                        place=place,
                        personalization_score=place_data.get('personalization_score', 0),
                        personalized_explanation=place_data.get('personalized_explanation', ''),
                        searched_location=place_data.get('searched_location', ''),
                        source_query=place_data.get('source_query', ''),
                        recommendation_rank=rank
                    )

                except Exception as e:
                    print(f"Error saving place {place_data.get('place_id', 'unknown')}: {e}")
                    continue
            
            print(f"{'Created' if created else 'Updated'} daily recommendations for {user.email}")
            print(f"Searched {len(locations_searched)} locations, generated {len(daily_recommendations)} top recommendations")
            
            return {
                "results": daily_recommendations,
                "user_profile": user_profile,
                "custom_queries": list(set(all_custom_queries)),
                "locations_searched": locations_searched,
                "total_results": len(scored_places)
            }
            
        except Exception as e:
            print(f"Error generating daily recommendations for {user.email}: {e}")
            import traceback
            traceback.print_exc()
            raise e

    def add_location_to_search_history(self, user, location_key):
        """Add a location to user's search history (max 10 recent locations)."""
        try:
            # Get current search history
            history = user.search_history_locations if user.search_history_locations else []
            
            # Remove location if it already exists (to move it to front)
            if location_key in history:
                history.remove(location_key)
            
            # Add to front of list
            history.insert(0, location_key)
            
            # Keep only last 10 searches
            history = history[:10]
            
            # Update user
            user.search_history_locations = history
            user.save(update_fields=['search_history_locations'])
            
            print(f"Added {location_key} to search history for {user.email}")
            
        except Exception as e:
            print(f"Error updating search history for {user.email}: {e}")

    async def get_coordinates_from_zip(self, zip_code):
        """Reuse existing geocoding method."""
        places_api = PlacesAPIView()
        return await places_api.get_coordinates_from_zip(zip_code)

class BatchRecommendationProcessingAPIView(APIView):
    """
    Separate endpoint for Cloud Run Scheduler to batch process daily recommendations.
    This is completely separate from the user-facing recommendation endpoint.
    """
    permission_classes = [AllowAny]  # Uses custom token authentication
    parser_classes = [JSONParser]

    def get(self, request):
        """
        Batch process recommendations for all users (Cloud Run Scheduler only).
        Requires SCHEDULER_AUTH_TOKEN for authentication.
        """
        # Validate scheduler authentication
        auth_header = request.META.get('HTTP_AUTHORIZATION', '')
        scheduler_token = os.getenv('SCHEDULER_AUTH_TOKEN', 'default-scheduler-token')
        
        if not auth_header.endswith(scheduler_token):
            return Response({"error": "Unauthorized scheduler request"}, status=401)
        
        return self._batch_process_all_users()
    
    def _batch_process_all_users(self):
        """
        Batch process recommendations for all users (for Cloud Run Scheduler).
        This endpoint processes all users and caches their recommendations.
        """
        start_time = time.time()
        processed_users = 0
        failed_users = 0
        
        try:
            # Get all active users with children
            users_with_children = User.objects.filter(
                is_active=True,
                children__isnull=False
            ).distinct()
            
            print(f"Starting batch processing for {users_with_children.count()} users with children")
            
            for user in users_with_children:
                try:
                    # Generate daily aggregated recommendations for this user
                    # Create a temporary instance to access the methods
                    recommendation_api = RecommendedPlacesAPIView()
                    recommendation_api.generate_and_cache_daily_recommendations(user)
                    
                    processed_users += 1
                    print(f"✅ Processed daily recommendations for {user.email}")
                    
                except Exception as e:
                    failed_users += 1
                    print(f"❌ Failed to process {user.email}: {e}")
                    continue
            
            execution_time = round((time.time() - start_time) * 1000, 2)
            
            return Response({
                "message": "Batch recommendation processing completed",
                "processed_users": processed_users,
                "failed_users": failed_users,
                "total_users": users_with_children.count(),
                "execution_time_ms": execution_time
            }, status=HTTP_200_OK)
            
        except Exception as e:
            print(f"Error in batch processing: {e}")
            import traceback
            traceback.print_exc()
            return Response({"error": str(e)}, status=HTTP_500_INTERNAL_SERVER_ERROR)

    def post(self, request):
        """
        Get personalized recommendations for the authenticated user.
        Returns cached results if available and fresh, otherwise generates new ones.
        """
        start_time = time.time()
        
        try:
            user = request.user
            print(f"Getting recommendations for user: {user.email}")
            
            # Get user's children and their data
            children = Child.objects.filter(user=user)

            print(f"Children: {children}")
            
            # Get location data from request
            data = request.data
            zip_code = data.get("zip_code")
            latitude = data.get("latitude", 40.712776)
            longitude = data.get("longitude", -74.005974)
            radius = data.get("radius", 500)
            force_refresh = data.get("force_refresh", False)
            
            # Get coordinates from zip if provided
            if zip_code:
                latitude, longitude = asyncio.run(self.get_coordinates_from_zip(zip_code))
                if not latitude or not longitude:
                    return Response({"error": "Invalid ZIP code"}, status=HTTP_400_BAD_REQUEST)
            
            # Add this location to user's search history for future recommendations
            location_key = zip_code or f"{latitude},{longitude}"
            self.add_location_to_search_history(user, location_key)
            
            print(f"Zip code: {zip_code}")
            print(f"Latitude: {latitude}")
            print(f"Longitude: {longitude}")
            print(f"Radius: {radius}")
            print(f"Force refresh: {force_refresh}")
            
            # Check for today's cached recommendations first (unless force_refresh is True)
            if not force_refresh:
                cached_recommendation = self.get_cached_recommendations(user)
                if cached_recommendation:
                    execution_time = round((time.time() - start_time) * 1000, 2)
                    print(f"✅ Returning today's cached recommendations for {user.email}")
                    
                    # Convert the related places back to the expected JSON format
                    from .models import UserRecommendationPlace
                    recommended_places = []
                    recommendation_places = UserRecommendationPlace.objects.filter(
                        user_recommendation=cached_recommendation
                    ).select_related('place').order_by('recommendation_rank')
                    
                    for rec_place in recommendation_places:
                        place = rec_place.place
                        place_data = {
                            "place_id": place.place_id,
                            "title": place.title,
                            "name": place.title,  # For compatibility
                            "description": place.description,
                            "rating": place.rating,
                            "reviews": place.reviews,
                            "formattedAddress": place.formatted_address,
                            "vicinity": place.formatted_address,  # For compatibility
                            "types": place.types,
                            "imagePlaces": place.place_images,
                            "category": place.category,
                            "location": {
                                "latitude": float(place.latitude) if place.latitude else None,
                                "longitude": float(place.longitude) if place.longitude else None
                            },
                            "gps_coordinates": {
                                "latitude": float(place.latitude) if place.latitude else None,
                                "longitude": float(place.longitude) if place.longitude else None
                            },
                            # Recommendation-specific data
                            "personalization_score": rec_place.personalization_score,
                            "searched_location": rec_place.searched_location,
                            "source_query": rec_place.source_query,
                            "recommendation_rank": rec_place.recommendation_rank
                        }
                        recommended_places.append(place_data)
                    
                    return Response({
                        "results": recommended_places,
                        "custom_queries": cached_recommendation.custom_queries,
                        "locations_searched": cached_recommendation.locations_searched,
                        "total_results": cached_recommendation.total_results,
                        "execution_time_ms": execution_time,
                        "cached": True,
                        "date_generated": cached_recommendation.date_generated.isoformat(),
                        "cache_updated": cached_recommendation.updated_at.isoformat()
                    }, status=HTTP_200_OK)
            
            # Generate fresh daily recommendations
            recommendation_data = self.generate_and_cache_daily_recommendations(user)
            
            execution_time = round((time.time() - start_time) * 1000, 2)
            recommendation_data["execution_time_ms"] = execution_time
            recommendation_data["cached"] = False
            
            return Response(recommendation_data, status=HTTP_200_OK)
            
        except Exception as e:
            print(f"Error in recommendations: {e}")
            import traceback
            traceback.print_exc()
            return Response({"error": str(e)}, status=HTTP_500_INTERNAL_SERVER_ERROR)

    def build_user_profile(self, user, children):
        """Build a comprehensive user profile for LLM context."""
        profile = {
            "user_name": user.name or "Parent",
            "email": user.email,
            "favorites": json.loads(user.favorites) if user.favorites else {},
            "children": []
        }
        
        for child in children:
            child_data = {
                "name": child.name,
                "age": child.age,
                "interests": child.interests or [],
                "age_group": self.get_age_group(child.age)
            }
            profile["children"].append(child_data)
        
        # Add derived insights
        profile["total_children"] = len(children)
        profile["age_ranges"] = list(set([self.get_age_group(child.age) for child in children]))
        profile["all_interests"] = list(set([interest for child in children for interest in (child.interests or [])]))
        
        return profile

    def get_age_group(self, age):
        """Categorize children by age group for better recommendations."""
        if age <= 2:
            return "toddler"
        elif age <= 5:
            return "preschooler"
        elif age <= 8:
            return "early_elementary"
        elif age <= 12:
            return "late_elementary"
        else:
            return "preteen"

    def generate_llm_queries(self, user_profile, location):
        """Use Gemini to generate personalized search queries based on user profile."""
        try:
            client = genai.Client(api_key=self.GEMINI_API_KEY)
            
            prompt = f"""
            You are a family activity recommendation expert. Based on the user profile below, generate 5-7 specific search queries to use for Google Maps search to find the best family places and activities.

            User Profile:
            - Parent: {user_profile['user_name']}
            - Number of children: {user_profile['total_children']}
            - Children details: {json.dumps(user_profile['children'], indent=2)}
            - Age groups represented: {', '.join(user_profile['age_ranges'])}
            - Combined interests: {', '.join(user_profile['all_interests'])}

            Guidelines:
            1. Create queries that match the children's ages and interests
            2. Consider safety and age-appropriateness
            3. Include both indoor and outdoor options
            4. Mix popular activities with unique local experiences
            5. Each query should be 3-7 words
            6. Focus on kid-friendly venues that welcome families

            Return only a JSON array of search query strings, like:
            ["family indoor play center", "toddler-friendly museums", "outdoor playgrounds"]
            """

            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
            )

            #print(f"Gemini response: {response}")

            # Parse Gemini response
            llm_output = response.text.strip()
            #print(f"Gemini raw output: {llm_output}")
            
            # Try to extract JSON array from response
            try:
                # Sometimes Gemini includes markdown formatting, so let's clean it
                if "```json" in llm_output:
                    llm_output = llm_output.split("```json")[1].split("```")[0].strip()
                elif "```" in llm_output:
                    llm_output = llm_output.split("```")[1].strip()
                
                custom_queries = json.loads(llm_output)
                if isinstance(custom_queries, list):
                    return custom_queries[:7]  # Limit to 7 queries
            except json.JSONDecodeError:
                print("Failed to parse Gemini JSON, using fallback")
            
        except Exception as e:
            print(f"Gemini generation failed: {e}")
        
        # Fallback queries if LLM fails
        return self.generate_fallback_queries(user_profile)

    def generate_fallback_queries(self, user_profile):
        """Generate fallback queries based on user profile without LLM."""
        queries = []
        interests = user_profile['all_interests']
        age_groups = user_profile['age_ranges']
        
        # Base queries for different interests
        interest_mapping = {
            "Parks": "family parks playgrounds",
            "Nature": "nature centers outdoor kids",
            "Animals": "petting zoos aquariums kids",
            "Creative": "kids art classes creative",
            "Sports": "family sports activities kids",
            "STEM": "science museums kids STEM",
            "Food": "family restaurants kids menu",
            "Play Centers": "indoor play centers kids",
            "Playgrounds": "playgrounds family outdoor"
        }
        
        # Add interest-based queries
        for interest in interests[:4]:  # Limit to top 4 interests
            if interest in interest_mapping:
                queries.append(interest_mapping[interest])
        
        # Add age-appropriate queries
        if "toddler" in age_groups:
            queries.append("toddler friendly activities")
        if "preschooler" in age_groups:
            queries.append("preschool kids activities")
        if any(age in age_groups for age in ["early_elementary", "late_elementary"]):
            queries.append("elementary kids activities")
        
        return queries[:6] if queries else ["family activities kids", "playgrounds parks", "kids museums"]

    async def get_recommended_places(self, custom_queries, latitude, longitude, radius):
        """Use existing places search infrastructure with custom queries."""
        all_places = []
        
        print(f"Getting recommended places for {custom_queries} with radius {radius} at {latitude},{longitude}")

        # TODO: split queries into groups using 'OR' and search each group separately

        # Convert custom queries into the format expected by existing search
            
        search_params = {
            "location": f"{latitude},{longitude}",
            "radius": radius,
            "key": PlacesAPIView.API_KEY
        }

        curr_query = ' OR '.join(custom_queries)
            
        try:
            places_api = PlacesAPIView()
            async with aiohttp.ClientSession() as session:
                results = await places_api.async_fetch_all_places(search_params, session, max_results=15, queries=custom_queries[:max_queries])
                
            # Process images for each place using the existing pipeline
            if results:
                print(f"Processing images for {len(results)} places from recommendations...")
                from concurrent.futures import ThreadPoolExecutor, as_completed
                
                with ThreadPoolExecutor(max_workers=10) as executor:
                    future_to_place = {
                        executor.submit(process_images_for_place, place): place
                        for place in results
                    }
                    
                    for future in as_completed(future_to_place):
                        place = future_to_place[future]
                        try:
                            processed_urls = future.result()
                            if processed_urls:
                                place["imagePlaces"] = processed_urls
                                print(f"Processed {len(processed_urls)} images for place {place.get('place_id', 'unknown')}")
                            else:
                                place["imagePlaces"] = []
                        except Exception as e:
                            print(f"Error processing images for place {place.get('place_id', 'unknown')}: {e}")
                            place["imagePlaces"] = []
            else:
                print(f"No results found for query '{curr_query}'")

            for place in results:
                place["source_query"] = curr_query
                all_places.append(place)

        except Exception as e:
            print(f"Error searching with query '{curr_query}': {e}")
        
        # Remove duplicates based on place_id
        unique_places = []
        seen_ids = set()
        for place in all_places:
            place_id = place.get("place_id")
            if place_id and place_id not in seen_ids:
                seen_ids.add(place_id)
                unique_places.append(place)

        print(f"Found {len(unique_places)} unique places for {custom_queries}")
        return unique_places

    def score_places_for_user(self, places, user_profile):
        """Add personalization scores based on how well places match user profile."""
        for place in places:
            score = 0
            
            # Score based on children's interests
            place_text = f"{place.get('title', '')} {place.get('description', '')} {place.get('category', '')}".lower()
            
            for interest in user_profile['all_interests']:
                if interest.lower() in place_text:
                    score += 10
            
            # Score based on age appropriateness
            age_keywords = {
                "toddler": ["toddler", "baby", "infant", "crawl"],
                "preschooler": ["preschool", "4 year", "5 year"],
                "early_elementary": ["elementary", "school age"],
                "late_elementary": ["kids", "children", "youth"]
            }
            
            for age_group in user_profile['age_ranges']:
                if age_group in age_keywords:
                    for keyword in age_keywords[age_group]:
                        if keyword in place_text:
                            score += 5
            
            # Boost score for highly rated places
            rating = place.get('rating', 0)
            if rating >= 4.5:
                score += 15
            elif rating >= 4.0:
                score += 10
            elif rating >= 3.5:
                score += 5
            
            # Boost score for places with reviews (more established)
            review_count = place.get('reviews', 0) or place.get('userRatingCount', 0)
            if review_count > 100:
                score += 10
            elif review_count > 50:
                score += 5
            
            place["personalization_score"] = score
            
        return places

    def get_user_locations(self, user):
        """Get up to 5 locations for recommendations (preferences first, then search history)."""
        all_locations = []
        
        # First, add location preferences (priority)
        if user.location_preferences:
            all_locations.extend(user.location_preferences)
        
        # Then add search history locations
        if user.search_history_locations:
            for location in user.search_history_locations:
                if location not in all_locations:  # Avoid duplicates
                    all_locations.append(location)
        
        # Limit to 5 total locations
        locations_to_use = all_locations[:5]
        
        # If user has no locations, use defaults
        if not locations_to_use:
            locations_to_use = ["90210", "10001", "94102"]  # Default cities
        
        # Convert to the expected format with coordinates
        location_data = {}
        default_coords = {
            "90210": {"latitude": 34.0901, "longitude": -118.4065},  # Beverly Hills
            "10001": {"latitude": 40.7505, "longitude": -73.9934},  # NYC
            "94102": {"latitude": 37.7849, "longitude": -122.4094}, # San Francisco
            "02101": {"latitude": 42.3601, "longitude": -71.0589}, # Boston
            "30309": {"latitude": 33.7490, "longitude": -84.3880}, # Atlanta
        }
        
        for location in locations_to_use:
            if location in default_coords:
                location_data[location] = default_coords[location]
            else:
                # For unknown zip codes, try to geocode them
                try:
                    lat, lng = asyncio.run(self.get_coordinates_from_zip(location))
                    if lat and lng:
                        location_data[location] = {"latitude": lat, "longitude": lng}
                except Exception as e:
                    print(f"Error geocoding {location}: {e}")
                    continue
        
        print(f"Using {len(location_data)} locations for {user.email}: {list(location_data.keys())}")
        return location_data

    def get_cached_recommendations(self, user):
        """Get today's cached recommendations if they exist and are fresh."""
        try:
            today = datetime.now().date()
            
            cached = UserRecommendation.objects.filter(
                user=user,
                date_generated=today
            ).first()
            
            if cached:
                # Verify the user profile hasn't changed by comparing hash
                children = Child.objects.filter(user=user)
                current_profile = self.build_user_profile(user, children)
                current_hash = self.get_profile_hash(current_profile)
                
                if cached.user_profile_hash == current_hash:
                    return cached
                else:
                    print(f"User profile changed for {user.email}, invalidating today's cache")
                    cached.delete()
            
            return None
            
        except Exception as e:
            print(f"Error checking cache for {user.email}: {e}")
            return None

    def get_profile_hash(self, user_profile):
        """Generate a hash of the user profile for cache invalidation."""
        # Create a stable string representation of the profile
        profile_string = json.dumps(user_profile, sort_keys=True)
        return hashlib.md5(profile_string.encode()).hexdigest()

    def generate_and_cache_daily_recommendations(self, user):
        """Generate daily aggregated recommendations from user's preferred and search history locations."""
        try:
            # Get user's children and their data
            children = Child.objects.filter(user=user)
            
            # Build user profile for LLM
            user_profile = self.build_user_profile(user, children)
            print(f"User profile for {user.email}: {user_profile}")
            
            # Get locations to search (preferences + history, max 5)
            locations_to_process = self.get_user_locations(user)
            
            all_places = []
            all_custom_queries = []
            locations_searched = list(locations_to_process.keys())
            
            print(f"Locations to process: {locations_to_process}")

            # Search across all user's locations
            for location_key, location_data in locations_to_process.items():
                try:
                    latitude = location_data['latitude']
                    longitude = location_data['longitude']
                    
                    # Generate custom search queries using LLM for this location
                    location_queries = self.generate_llm_queries(user_profile, location_key)
                    print(f"Location queries: {location_queries}")
                    
                    all_custom_queries.extend(location_queries)
                    
                    # Get places for this location
                    recommended_places = asyncio.run(self.get_recommended_places(
                        location_queries, latitude, longitude, 20000
                    ))
                    
                    # Add location info to each place
                    for place in recommended_places:
                        place['searched_location'] = location_key
                    
                    all_places.extend(recommended_places)
                    print(f"Found {len(recommended_places)} places for {location_key}")
                    
                except Exception as e:
                    print(f"Error searching location {location_key} for {user.email}: {e}")
                    continue
            
            # Remove duplicates based on place_id
            unique_places = []
            seen_ids = set()
            for place in all_places:
                place_id = place.get("place_id")
                if place_id and place_id not in seen_ids:
                    seen_ids.add(place_id)
                    unique_places.append(place)

            print(f"Unique places: {len(unique_places)}")
            
            
            # Add personalization scores
            scored_places = self.score_places_for_user(unique_places, user_profile)
            
            # Sort by personalization score
            scored_places.sort(key=lambda x: x.get("personalization_score", 0), reverse=True)
            
            # Limit to top 15 for daily recommendations
            daily_recommendations = scored_places[:15]
            
            # Cache the results for today
            today = datetime.now().date()
            profile_hash = self.get_profile_hash(user_profile)
            
            print(f"Updating or creating today's recommendation for {user.email}")
            
            
            # Update or create today's recommendation
            cached_recommendation, created = UserRecommendation.objects.update_or_create(
                user=user,
                date_generated=today,
                defaults={
                    'user_profile_hash': profile_hash,
                    'custom_queries': list(set(all_custom_queries)),  # Remove duplicates
                    'locations_searched': locations_searched,
                    'total_results': len(scored_places)
                }
            )
            
            # Clear existing places for this recommendation (in case of update)
            if not created:
                cached_recommendation.recommended_places.clear()
            
            # Create Place objects and UserRecommendationPlace relationships
            from .models import UserRecommendationPlace
            for rank, place_data in enumerate(daily_recommendations, 1):
                try:
                    # Get or create the Place object
                    place, place_created = Place.objects.get_or_create(
                        place_id=place_data.get('place_id'),
                        defaults={
                            'title': place_data.get('title', place_data.get('name', '')),
                            'description': place_data.get('description', ''),
                            'latitude': place_data.get('location', {}).get('latitude') or place_data.get('gps_coordinates', {}).get('latitude'),
                            'longitude': place_data.get('location', {}).get('longitude') or place_data.get('gps_coordinates', {}).get('longitude'),
                            'rating': place_data.get('rating'),
                            'reviews': place_data.get('reviews') or place_data.get('userRatingCount'),
                            'formatted_address': place_data.get('formattedAddress', place_data.get('vicinity', '')),
                            'types': place_data.get('types', []),
                            'place_images': place_data.get('imagePlaces', []),
                            'category': place_data.get('category', ''),
                        }
                    )
                    
                    # Create the recommendation relationship
                    UserRecommendationPlace.objects.create(
                        user_recommendation=cached_recommendation,
                        place=place,
                        personalization_score=place_data.get('personalization_score', 0),
                        searched_location=place_data.get('searched_location', ''),
                        source_query=place_data.get('source_query', ''),
                        recommendation_rank=rank
                    )
                    
                except Exception as e:
                    print(f"Error saving place {place_data.get('place_id', 'unknown')}: {e}")
                    continue
            
            print(f"{'Created' if created else 'Updated'} daily recommendations for {user.email}")
            print(f"Searched {len(locations_searched)} locations, generated {len(daily_recommendations)} top recommendations")
            
            return {
                "results": daily_recommendations,
                "user_profile": user_profile,
                "custom_queries": list(set(all_custom_queries)),
                "locations_searched": locations_searched,
                "total_results": len(scored_places)
            }
            
        except Exception as e:
            print(f"Error generating daily recommendations for {user.email}: {e}")
            import traceback
            traceback.print_exc()
            raise e

    def add_location_to_search_history(self, user, location_key):
        """Add a location to user's search history (max 10 recent locations)."""
        try:
            # Get current search history
            history = user.search_history_locations if user.search_history_locations else []
            
            # Remove location if it already exists (to move it to front)
            if location_key in history:
                history.remove(location_key)
            
            # Add to front of list
            history.insert(0, location_key)
            
            # Keep only last 10 searches
            history = history[:10]
            
            # Update user
            user.search_history_locations = history
            user.save(update_fields=['search_history_locations'])
            
            print(f"Added {location_key} to search history for {user.email}")
            
        except Exception as e:
            print(f"Error updating search history for {user.email}: {e}")

    async def get_coordinates_from_zip(self, zip_code):
        """Reuse existing geocoding method."""
        places_api = PlacesAPIView()
        return await places_api.get_coordinates_from_zip(zip_code)


class RecommendedEventsAPIView(APIView):
    """
    Personalized event recommendations endpoint for authenticated users.
    
    - GET: Retrieve user's latest recommended events (for frontend display)
    - POST: Generate/refresh personalized event recommendations for a user
    """
    permission_classes = [IsAuthenticated]
    parser_classes = [JSONParser]
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

    def get(self, request):
        """Get the latest recommended events for the authenticated user."""
        start_time = time.time()
        user = request.user
        
        print(f"🎯 GET /api/events/recommended/ - User: {user.email} (ID: {user.id})")
        print(f"📊 Request headers: {dict(request.headers)}")
        
        try:
            print(f"🔍 Fetching latest event recommendations for user: {user.email}")
            recommendations = self._get_user_latest_event_recommendations(user)
            
            execution_time = round((time.time() - start_time) * 1000, 2)
            print(f"✅ GET Success - Found {len(recommendations)} cached event recommendations")
            print(f"⏱️ GET Execution time: {execution_time}ms")
            
            return Response({
                "results": recommendations,
                "status": "success",
                "execution_time_ms": execution_time,
                "total_results": len(recommendations)
            }, status=HTTP_200_OK)
            
        except Exception as e:
            execution_time = round((time.time() - start_time) * 1000, 2)
            print(f"❌ GET Error for user {user.email}: {e}")
            print(f"⏱️ GET Failed execution time: {execution_time}ms")
            import traceback
            traceback.print_exc()
            
            return Response({
                "error": "Failed to get event recommendations",
                "status": "error",
                "execution_time_ms": execution_time
            }, status=HTTP_500_INTERNAL_SERVER_ERROR)

    def post(self, request):
        """
        Generate personalized event recommendations for the authenticated user.
        Returns cached results if available and fresh, otherwise generates new ones.
        """
        start_time = time.time()
        user = request.user
        
        print(f"🚀 POST /api/events/recommended/ - User: {user.email} (ID: {user.id})")
        print(f"📊 Request data keys: {list(request.data.keys())}")
        print(f"📊 Request headers: {dict(request.headers)}")
        
        try:
            print(f"🔍 Starting event recommendation generation for user: {user.email}")
            
            # Get user's children and their data
            children = Child.objects.filter(user=user)
            print(f"👶 User has {children.count()} children: {[f'{child.name} (age {child.age})' for child in children]}")
            
            # Get location data from request
            data = request.data
            zip_code = data.get("zip_code")
            
            #latitude = data.get("latitude", 40.712776)
            #longitude = data.get("longitude", -74.005974)
            radius = data.get("radius", 20000)  # miles for events
            
            event_types = data.get("event_types", ["kid-friendly", "family-friendly"])
            date_range = data.get("date_range", "this_week")
            force_refresh = data.get("force_refresh", True)
            is_homebase = data.get("is_homebase", False)  # Flag to indicate this is a homebase update
            
            print(f"📍 Location parameters:")
            print(f"   Zip code: {zip_code}")
            print(f"   Radius: {radius} miles")
            print(f"🎪 Event parameters:")
            print(f"   Event types: {event_types}")
            print(f"   Date range: {date_range}")
            print(f"   Force refresh: {force_refresh}")
            
            

            # Get coordinates from zip if provided
            if zip_code:
                print(f"🗺️ Geocoding ZIP code: {zip_code}")
                geocode_start = time.time()
                latitude, longitude = asyncio.run(self.get_coordinates_from_zip(zip_code))
                geocode_time = round((time.time() - geocode_start) * 1000, 2)
                print(f"🗺️ Geocoding completed in {geocode_time}ms: {latitude}, {longitude}")
                
                if not latitude or not longitude:
                    print(f"❌ Invalid ZIP code: {zip_code}")
                    return Response({"error": "Invalid ZIP code"}, status=HTTP_400_BAD_REQUEST)
            
            # Update user's homebase if this is a homebase update
            if is_homebase and zip_code:
                try:
                    user.homebaseZipCode = zip_code
                    user.save()
                    print(f"✅ Updated homebase for {user.email} to {zip_code}")
                except Exception as e:
                    print(f"⚠️ Failed to update homebase for {user.email}: {e}")
                    # Don't fail the entire request if homebase update fails
            
            # Add this location to user's search history for future recommendations
            location_key = zip_code or f"{latitude},{longitude}"
            print(f"📝 Adding location to search history: {location_key}")
            self.add_location_to_search_history(user, location_key)
            
            # Check for today's cached event recommendations first (unless force_refresh is True)
            if not force_refresh:
                print(f"🔍 Checking for cached event recommendations...")
                cache_check_start = time.time()
                cached_recommendation = self.get_cached_event_recommendations(user)
                cache_check_time = round((time.time() - cache_check_start) * 1000, 2)
                
                if cached_recommendation:
                    print(f"✅ Found cached event recommendations from {cached_recommendation.created_at}")
                    print(f"🔍 Cache check completed in {cache_check_time}ms")
                    
                    # Convert the related events back to the expected JSON format
                    conversion_start = time.time()
                    recommended_events = []
                    recommendation_events = UserRecommendationEvent.objects.filter(
                        user_recommendation=cached_recommendation
                    ).select_related('event').order_by('recommendation_rank')
                    
                    print(f"📊 Converting {recommendation_events.count()} cached events to response format")
                    
                    for rec_event in recommendation_events:
                        event = rec_event.event
                        event_data = {
                            "id": event.event_id,
                            "title": event.title,
                            "description": event.description,
                            "start_date": event.start_date,
                            "when": event.when,
                            "address": event.address,
                            "formatted_address": event.formatted_address,
                            "event_latitude": float(event.latitude) if event.latitude else None,
                            "event_longitude": float(event.longitude) if event.longitude else None,
                            "venue_name": event.venue_name,
                            "venue_rating": event.venue_rating,
                            "venue_reviews": event.venue_reviews,
                            "link": event.link,
                            "thumbnail": event.thumbnail,
                            "image": event.image,
                            "event_type": event.event_type,
                            "has_tickets": event.has_tickets,
                            "ticket_info": event.ticket_info,
                            "ticket_sources": event.ticket_sources,
                            "source": event.source,
                            # Recommendation-specific data
                            "personalization_score": rec_event.personalization_score,
                            "recommendation_rank": rec_event.recommendation_rank
                        }
                        recommended_events.append(event_data)
                    
                    conversion_time = round((time.time() - conversion_start) * 1000, 2)
                    execution_time = round((time.time() - start_time) * 1000, 2)
                    
                    print(f"📊 Event conversion completed in {conversion_time}ms")
                    print(f"✅ POST Success (CACHED) - Returning {len(recommended_events)} cached event recommendations")
                    print(f"⏱️ Total POST execution time: {execution_time}ms")
                    
                    return Response({
                        "results": recommended_events,
                        "custom_queries": cached_recommendation.custom_queries,
                        "locations_searched": cached_recommendation.locations_searched,
                        "total_results": cached_recommendation.total_results,
                        "execution_time_ms": execution_time,
                        "cached": True,
                        "date_generated": cached_recommendation.created_at.isoformat()
                    }, status=HTTP_200_OK)
                else:
                    print(f"❌ No cached event recommendations found")
                    print(f"🔍 Cache check completed in {cache_check_time}ms")
            
            # Generate fresh event recommendations
            print(f"🔄 Generating fresh event recommendations...")
            generation_start = time.time()
            
            recommendation_data = self.generate_and_cache_event_recommendations(
                user, latitude, longitude, radius, event_types, date_range
            )
            
            generation_time = round((time.time() - generation_start) * 1000, 2)
            execution_time = round((time.time() - start_time) * 1000, 2)
            
            print(f"🔄 Event generation completed in {generation_time}ms")
            print(f"✅ POST Success (FRESH) - Generated {len(recommendation_data.get('results', []))} new event recommendations")
            print(f"⏱️ Total POST execution time: {execution_time}ms")

            
            
            recommendation_data["execution_time_ms"] = execution_time
            recommendation_data["cached"] = False
            
            return Response(recommendation_data, status=HTTP_200_OK)
            
        except Exception as e:
            execution_time = round((time.time() - start_time) * 1000, 2)
            print(f"❌ POST Error for user {user.email}: {e}")
            print(f"⏱️ POST Failed execution time: {execution_time}ms")
            import traceback
            traceback.print_exc()
            return Response({
                "error": str(e),
                "execution_time_ms": execution_time
            }, status=HTTP_500_INTERNAL_SERVER_ERROR)

    def _get_user_latest_event_recommendations(self, user):
        """Get the latest event recommendations for a user from cache/database."""
        try:
            print(f"🔍 _get_user_latest_event_recommendations for user: {user.email}")
            
            # Get the latest recommendation entry for this user
            latest_recommendation = UserEventRecommendation.objects.filter(
                user=user
            ).order_by('-created_at').first()
            
            if not latest_recommendation:
                print(f"❌ No event recommendations found for user: {user.email}")
                return []
            
            print(f"✅ Found latest event recommendation from: {latest_recommendation.created_at}")
            print(f"📊 Recommendation metadata: {latest_recommendation.total_results} total results, {len(latest_recommendation.custom_queries)} queries")
            
            # Get the recommended events for this recommendation
            recommended_events = UserRecommendationEvent.objects.filter(
                user_recommendation=latest_recommendation
            ).order_by('recommendation_rank')
            
            print(f"📊 Converting {recommended_events.count()} recommendation events to response format")
            
            # Convert to list of event dictionaries
            events_list = []
            for rec_event in recommended_events:
                event_data = {
                    "id": rec_event.event.event_id,
                    "title": rec_event.event.title,
                    "description": rec_event.event.description,
                    "start_date": rec_event.event.start_date,
                    "when": rec_event.event.when,
                    "address": rec_event.event.address,
                    "formatted_address": rec_event.event.formatted_address,
                    "event_latitude": float(rec_event.event.latitude) if rec_event.event.latitude else None,
                    "event_longitude": float(rec_event.event.longitude) if rec_event.event.longitude else None,
                    "venue_name": rec_event.event.venue_name,
                    "venue_rating": rec_event.event.venue_rating,
                    "venue_reviews": rec_event.event.venue_reviews,
                    "link": rec_event.event.link,
                    "thumbnail": rec_event.event.thumbnail,
                    "image": rec_event.event.image,
                    "event_type": rec_event.event.event_type,
                    "has_tickets": rec_event.event.has_tickets,
                    "ticket_info": rec_event.event.ticket_info,
                    "ticket_sources": rec_event.event.ticket_sources,
                    "personalization_score": rec_event.personalization_score,
                    "recommendation_rank": rec_event.recommendation_rank,
                    "source": rec_event.event.source
                }
                events_list.append(event_data)
            
            print(f"✅ Successfully converted {len(events_list)} events for user: {user.email}")
            return events_list
            
        except Exception as e:
            print(f"❌ Error retrieving latest event recommendations for {user.email}: {e}")
            import traceback
            traceback.print_exc()
            return []

    def build_user_profile_for_events(self, user, children):
        """Build a comprehensive user profile for event LLM context."""
        profile = {
            "user_name": user.name or "Parent",
            "email": user.email,
            "favorites": json.loads(user.favorites) if user.favorites else {},
            "children": []
        }
        
        for child in children:
            child_data = {
                "name": child.name,
                "age": child.age,
                "interests": child.interests or [],
                "age_group": self.get_age_group_for_events(child.age)
            }
            profile["children"].append(child_data)
        
        # Add derived insights
        profile["total_children"] = len(children)
        profile["age_ranges"] = list(set([self.get_age_group_for_events(child.age) for child in children]))
        profile["all_interests"] = list(set([interest for child in children for interest in (child.interests or [])]))
        
        return profile

    def get_age_group_for_events(self, age):
        """Categorize children by age group for better event recommendations."""
        if age <= 2:
            return "toddler"
        elif age <= 5:
            return "preschooler"
        elif age <= 8:
            return "early_elementary"
        elif age <= 12:
            return "late_elementary"
        else:
            return "preteen"

    def generate_event_llm_queries(self, user_profile, location):
        """Use Gemini to generate personalized event search queries based on user profile."""
        try:
            client = genai.Client(api_key=self.GEMINI_API_KEY)
            
            prompt = f"""
            You are a family event recommendation expert. Based on the user profile below, generate 4-6 specific search queries to find the best family events and activities.

            User Profile:
            - Parent: {user_profile['user_name']}
            - Number of children: {user_profile['total_children']}
            - Children details: {json.dumps(user_profile['children'], indent=2)}
            - Age groups represented: {', '.join(user_profile['age_ranges'])}
            - Combined interests: {', '.join(user_profile['all_interests'])}

            Guidelines:
            1. Create event-specific queries that match the children's ages and interests
            2. Focus on family events, workshops, classes, and activities
            3. Consider seasonal and educational events
            4. Include both indoor and outdoor event options
            5. Each query should be 3-6 words
            6. Focus on events that welcome families and children

            Examples of good queries:
            - "kids art workshops"
            - "family science events"
            - "children music classes"
            - "outdoor family activities"

            Return only a JSON array of search query strings, like:
            ["kids art workshops", "family science events", "children outdoor activities"]
            """

            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
            )

            # Parse Gemini response
            llm_output = response.text.strip()
            
            # Try to extract JSON array from response
            try:
                # Sometimes Gemini includes markdown formatting, so let's clean it
                if "```json" in llm_output:
                    llm_output = llm_output.split("```json")[1].split("```")[0].strip()
                elif "```" in llm_output:
                    llm_output = llm_output.split("```")[1].strip()
                
                custom_queries = json.loads(llm_output)
                if isinstance(custom_queries, list):
                    return custom_queries[:6]  # Limit to 6 queries
            except json.JSONDecodeError:
                print("Failed to parse Gemini JSON for events, using fallback")
            
        except Exception as e:
            print(f"Gemini event generation failed: {e}")
        
        # Fallback queries if LLM fails
        return self.generate_fallback_event_queries(user_profile)

    def generate_fallback_event_queries(self, user_profile):
        """Generate fallback event queries based on user profile without LLM."""
        queries = []
        interests = user_profile['all_interests']
        age_groups = user_profile['age_ranges']
        
        # Base queries for different interests
        interest_mapping = {
            "Arts": "kids art workshops",
            "Music": "children music classes",
            "Sports": "family sports events",
            "STEM": "science events kids",
            "Animals": "petting zoo events",
            "Nature": "outdoor family activities",
            "Creative": "craft workshops kids",
            "Food": "cooking classes children"
        }
        
        # Add interest-based queries
        for interest in interests[:3]:  # Limit to top 3 interests
            if interest in interest_mapping:
                queries.append(interest_mapping[interest])
        
        # Add age-appropriate queries
        if "toddler" in age_groups or "preschooler" in age_groups:
            queries.append("toddler events activities")
        if any(age in age_groups for age in ["early_elementary", "late_elementary"]):
            queries.append("kids educational events")
        
        # Add general family queries if we don't have enough
        if len(queries) < 3:
            queries.extend(["family events kids", "children workshops", "kids activities"])
        
        return queries[:5] if queries else ["family events", "kids activities", "children workshops"]

    async def get_recommended_events(self, custom_queries, latitude, longitude, radius, date_range):
        """Use existing events search infrastructure with custom queries."""
        all_events = []
        
        print(f"Getting recommended events for {custom_queries} with radius {radius} miles at {latitude},{longitude}")

        try:
            # Use EventsAPIView logic for searching events
            events_api = EventsAPIView()
            
            # Convert radius from miles to meters for consistency (though events use miles)
            search_radius = radius
            
            for query in custom_queries:
                try:
                    # Build search parameters similar to EventsAPIView
                    city_and_state = events_api.get_city_from_coordinates(latitude, longitude)
                    
                    # Use the async_search_events method from EventsAPIView
                    event_results = await events_api.async_search_events(
                        city_and_state, 
                        [query],  # event_types as list
                        date_range, 
                        1  # page
                    )
                    
                    # Add source query to each event and filter by distance
                    for event in event_results:
                        event["source_query"] = query
                        
                        # Extract coordinates from event address if not already present
                        event_lat = event.get("event_latitude")
                        event_lon = event.get("event_longitude")
                        
                        # If coordinates are missing, try to extract from address
                        if not event_lat or not event_lon:
                            print(f"🗺️ Extracting coordinates for event: {event.get('title', 'Unknown')}")
                            event_lat, event_lon = self.extract_coordinates_from_address(event.get("address", []))
                            if event_lat and event_lon:
                                event["event_latitude"] = event_lat
                                event["event_longitude"] = event_lon
                                print(f"✅ Extracted coordinates: {event_lat}, {event_lon}")
                            else:
                                print(f"❌ Could not extract coordinates for event")

                        print(f"📍 Event: {event.get('title', 'Unknown')} at {event_lat}, {event_lon}")

                        if event_lat and event_lon:
                            # Calculate distance from search location using our own method
                            distance = self.calculate_distance(latitude, longitude, event_lat, event_lon)
                            print(f"📏 Distance: {distance} miles (radius limit: {radius} miles)")
                            
                            if distance and distance <= radius:
                                event["distance"] = distance
                                all_events.append(event)
                                print(f"✅ Event within radius - added to results")
                            else:
                                print(f"❌ Event outside radius - excluded")
                        else:
                            # If we can't get coordinates, include the event but mark distance as unknown
                            event["distance"] = None
                            event["event_latitude"] = None
                            event["event_longitude"] = None
                            all_events.append(event)
                            print(f"⚠️ Event added without distance verification (no coordinates)")
                    
                    
                    print(f"🔍 Found {len(event_results)} events for query: {query}")
                    print(f"✅ Added {len([e for e in event_results if e.get('distance') is not None and e.get('distance') <= radius])} events within radius")
                    
                except Exception as e:
                    print(f"Error searching events with query '{query}': {e}")
                    continue
        
        except Exception as e:
            print(f"Error in get_recommended_events: {e}")
        
        # Remove duplicates based on event ID
        unique_events = []
        seen_ids = set()
        for event in all_events:
            event_id = event.get("id")
            if event_id and event_id not in seen_ids:
                seen_ids.add(event_id)
                unique_events.append(event)
        
        print(f"Returning {len(unique_events)} unique recommended events")
        return unique_events

    def score_events_for_user(self, events, user_profile):
        """Use Gemini AI to intelligently score events based on kid-friendliness, family-friendliness, and user profile matching."""
        print(f"🤖 Starting AI-powered event scoring for {len(events)} events")
        
        # If no events, return early
        if not events:
            return events
            
        try:
            client = genai.Client(api_key=self.GEMINI_API_KEY)
            
            # Process events in batches to avoid token limits
            batch_size = 20
            scored_events = []
            
            for i in range(0, len(events), batch_size):
                batch = events[i:i + batch_size]
                
                # Create event summaries for Gemini analysis
                event_summaries = []
                for idx, event in enumerate(batch):
                    event_summary = {
                        "index": i + idx,  # Use global index, not batch-local index
                        "id": event.get('id', 'unknown'),
                        "title": event.get('title', ''),
                        "description": event.get('description', ''),
                        "event_type": event.get('event_type', ''),
                        "venue_name": event.get('venue_name', ''),
                        "venue_rating": event.get('venue_rating', 0),
                        "venue_reviews": event.get('venue_reviews', 0),
                        "address": event.get('formatted_address', ''),
                        "when": event.get('when', ''),
                        "has_tickets": event.get('has_tickets', False),
                        "ticket_info": event.get('ticket_info', [])
                    }
                    event_summaries.append(event_summary)
                
                prompt = f"""
                You are an expert in family and child-friendly event evaluation. Analyze each event and score them based on how well they match this family's profile and their kid/family-friendliness.

                FAMILY PROFILE:
                - Parent: {user_profile['user_name']}
                - Number of children: {user_profile['total_children']}
                - Children details: {json.dumps(user_profile['children'], indent=2)}
                - Age groups: {', '.join(user_profile['age_ranges'])}
                - Combined interests: {', '.join(user_profile['all_interests'])}
                - User favorites: {json.dumps(user_profile.get('favorites', {}), indent=2)}

                EVENTS TO ANALYZE:
                {json.dumps(event_summaries, indent=2)}

                SCORING CRITERIA (Total: 100 points):
                1. **Kid-Friendliness (30 points)**: How suitable is this event for children? Consider safety, age-appropriateness, engagement level, and whether children are welcomed/encouraged.

                2. **Family-Friendliness (25 points)**: How well does this accommodate families? Consider parent supervision, family activities, stroller access, facilities, etc.

                3. **Profile Match (25 points)**: How well does this event match the children's ages, interests, and family preferences?

                4. **Quality & Reliability (20 points)**: Venue rating, reviews, event organization, ticketing reliability, etc.

                SPECIAL CONSIDERATIONS:
                - Age appropriateness is crucial - events should match the children's developmental stages
                - Educational value adds bonus points
                - Safety and supervision considerations for different age groups
                - Accessibility for families (parking, strollers, etc.)
                - Weather considerations for outdoor events
                - Time appropriateness (not too late for young children)

                IMPORTANT: For each event, you MUST use the exact same "index" value that was provided in the event data above. Do NOT change or renumber the indices.

                Return ONLY a JSON array with scores for each event in the same order:
                [
                    {{
                        "index": 0,
                        "score": 85,
                        "reasoning": "High kid-friendliness with hands-on activities perfect for ages 4-8, excellent venue rating, matches STEM interests"
                    }},
                    {{
                        "index": 1,
                        "score": 45,
                        "reasoning": "Limited kid appeal, more adult-focused, but family-friendly venue"
                    }},
                    ...
                ]
                """

                try:
                    response = client.models.generate_content(
                        model="gemini-2.5-flash",
                        contents=prompt,
                    )

                    # Parse Gemini response
                    llm_output = response.text.strip()
                    
                    # Clean up JSON formatting
                    if "```json" in llm_output:
                        llm_output = llm_output.split("```json")[1].split("```")[0].strip()
                    elif "```" in llm_output:
                        llm_output = llm_output.split("```")[1].strip()
                    
                    scores = json.loads(llm_output)
                    
                    # Apply scores to events in this batch
                    for score_data in scores:
                        if isinstance(score_data, dict) and 'index' in score_data and 'score' in score_data:
                            event_idx = score_data['index']
                            if 0 <= event_idx < len(batch):
                                ai_score = max(0, min(100, score_data['score']))  # Clamp between 0-100
                                reasoning = score_data.get('reasoning', 'AI analysis')
                                
                                batch[event_idx]["personalization_score"] = ai_score
                                batch[event_idx]["ai_reasoning"] = reasoning
                                
                                print(f"🎯 Event '{batch[event_idx].get('title', 'Unknown')}' scored {ai_score}/100")
                                print(f"   Reasoning: {reasoning}")
                    
                    scored_events.extend(batch)
                    print(f"✅ Processed batch {i//batch_size + 1}/{(len(events) + batch_size - 1)//batch_size}")
                    
                except json.JSONDecodeError as e:
                    print(f"❌ Failed to parse Gemini response for batch {i//batch_size + 1}: {e}")
                    print(f"Raw response: {llm_output[:200]}...")
                    # Fallback to rule-based scoring for this batch
                    scored_events.extend(self._fallback_score_events(batch, user_profile))
                    
                except Exception as e:
                    print(f"❌ Gemini API error for batch {i//batch_size + 1}: {e}")
                    # Fallback to rule-based scoring for this batch
                    scored_events.extend(self._fallback_score_events(batch, user_profile))
                
                # Small delay between batches to respect rate limits
                if i + batch_size < len(events):
                    time.sleep(0.5)
            
            print(f"🎉 AI event scoring completed for {len(scored_events)} events")
            return scored_events
            
        except Exception as e:
            print(f"❌ Critical error in AI event scoring: {e}")
            import traceback
            traceback.print_exc()
            # Fallback to rule-based scoring for all events
            return self._fallback_score_events(events, user_profile)

    def _fallback_score_events(self, events, user_profile):
        """Fallback rule-based scoring when AI fails."""
        print(f"⚠️ Using fallback rule-based scoring for {len(events)} events")
        
        for event in events:
            score = 0
            
            # Score based on children's interests
            event_text = f"{event.get('title', '')} {event.get('description', '')} {event.get('event_type', '')}".lower()
            
            for interest in user_profile['all_interests']:
                if interest.lower() in event_text:
                    score += 15
            
            # Score based on age appropriateness
            age_keywords = {
                "toddler": ["toddler", "baby", "infant", "ages 0-2", "under 3"],
                "preschooler": ["preschool", "ages 3-5", "4 year", "5 year"],
                "early_elementary": ["elementary", "ages 6-8", "school age"],
                "late_elementary": ["kids", "children", "ages 9-12", "youth"]
            }
            
            for age_group in user_profile['age_ranges']:
                if age_group in age_keywords:
                    for keyword in age_keywords[age_group]:
                        if keyword in event_text:
                            score += 10
            
            # Boost score for highly rated venues
            venue_rating = event.get('venue_rating', 0)
            if venue_rating and venue_rating >= 4.5:
                score += 10
            elif venue_rating and venue_rating >= 4.0:
                score += 5
            
            # Boost score for events with good reviews
            venue_reviews = event.get('venue_reviews', 0)
            if venue_reviews and venue_reviews > 50:
                score += 5
            
            # Boost for family-friendly indicators
            family_keywords = ["family", "kids", "children", "parent", "all ages"]
            for keyword in family_keywords:
                if keyword in event_text:
                    score += 8
                    break
            
            # Boost for educational events
            educational_keywords = ["workshop", "class", "learn", "educational", "science", "museum"]
            for keyword in educational_keywords:
                if keyword in event_text:
                    score += 5
                    break
            
            event["personalization_score"] = score
            event["ai_reasoning"] = "Rule-based fallback scoring"
            
        return events

    def get_cached_event_recommendations(self, user):
        """Get today's cached event recommendations if they exist and are fresh."""
        try:
            today = datetime.now().date()
            print(f"🔍 Checking for cached event recommendations for {user.email} on {today}")
            
            cached = UserEventRecommendation.objects.filter(
                user=user,
                created_at__date=today
            ).first()
            
            if cached:
                print(f"✅ Found cached event recommendation from {cached.created_at}")
                
                # Verify the user profile hasn't changed by comparing hash
                print(f"🔍 Verifying user profile hash...")
                children = Child.objects.filter(user=user)
                current_profile = self.build_user_profile_for_events(user, children)
                current_hash = self.get_profile_hash_for_events(current_profile)
                
                print(f"📊 Cached hash: {cached.user_profile_hash}")
                print(f"📊 Current hash: {current_hash}")
                
                if cached.user_profile_hash == current_hash:
                    print(f"✅ Profile hash matches - returning cached event recommendations")
                    return cached
                else:
                    print(f"❌ User profile changed for {user.email}, invalidating today's event cache")
                    cached.delete()
                    return None
            else:
                print(f"❌ No cached event recommendations found for {user.email} today")
            
            return None
            
        except Exception as e:
            print(f"❌ Error checking event cache for {user.email}: {e}")
            import traceback
            traceback.print_exc()
            return None

    def get_profile_hash_for_events(self, user_profile):
        """Generate a hash of the user profile for cache invalidation."""
        # Create a stable string representation of the profile
        profile_string = json.dumps(user_profile, sort_keys=True)
        return hashlib.md5(profile_string.encode()).hexdigest()

    def generate_and_cache_event_recommendations(self, user, latitude, longitude, radius, event_types, date_range):
        """Generate and cache event recommendations for a user."""
        try:
            print(f"🔄 generate_and_cache_event_recommendations for {user.email}")
            print(f"📍 Location: {latitude}, {longitude} (radius: {radius} miles)")
            print(f"🎪 Event types: {event_types}, Date range: {date_range}")
            
            # Get user's children and their data
            children = Child.objects.filter(user=user)
            print(f"👶 Found {children.count()} children for profile building")
            
            # Build user profile for LLM
            profile_start = time.time()
            user_profile = self.build_user_profile_for_events(user, children)
            profile_time = round((time.time() - profile_start) * 1000, 2)
            print(f"👤 Built user profile in {profile_time}ms: {user_profile}")
            
            # Generate custom search queries using LLM
            llm_start = time.time()
            custom_queries = self.generate_event_llm_queries(user_profile, f"{latitude},{longitude}")
            llm_time = round((time.time() - llm_start) * 1000, 2)
            print(f"🤖 Generated {len(custom_queries)} LLM queries in {llm_time}ms: {custom_queries}")
            print(f"🤖 Custom queries: {custom_queries}")
            
            
            # Use top 2 queries for testing
            custom_queries = custom_queries[:2]

            # Get recommended events
            search_start = time.time()
            recommended_events = asyncio.run(self.get_recommended_events(
                custom_queries, latitude, longitude, radius, date_range
            ))
            search_time = round((time.time() - search_start) * 1000, 2)
            print(f"🔍 Found {len(recommended_events)} events in {search_time}ms")
            
            # Add personalization scores
            scoring_start = time.time()
            scored_events = self.score_events_for_user(recommended_events, user_profile)
            scoring_time = round((time.time() - scoring_start) * 1000, 2)
            print(f"📊 Scored {len(scored_events)} events in {scoring_time}ms")
            
            # Sort by personalization score
            scored_events.sort(key=lambda x: x.get("personalization_score", 0), reverse=True)
            print(f"📈 Top 3 scored events: {[(e.get('title', 'Unknown'), e.get('personalization_score', 0)) for e in scored_events[:3]]}")
            
            # Limit to top 10 for event recommendations
            daily_recommendations = scored_events[:10]
            print(f"🎯 Selected top {len(daily_recommendations)} events for recommendations")
            
            # Cache the results for today
            today = datetime.now().date()
            profile_hash = self.get_profile_hash_for_events(user_profile)
            
            print(f"💾 Caching event recommendations for {user.email}")
            print(f"📊 Profile hash: {profile_hash}")
            
            # Create or update UserEventRecommendation
            cache_start = time.time()
            cached_recommendation, created = UserEventRecommendation.objects.update_or_create(
                user=user,
                date_generated=today,
                defaults={
                    'user_profile_hash': profile_hash,
                    'custom_queries': custom_queries,
                    'locations_searched': [f"{latitude},{longitude}"],
                    'total_results': len(scored_events)
                }
            )
            
            if created:
                print(f"✅ Created new UserEventRecommendation for {user.email}")
            else:
                print(f"🔄 Updated existing UserEventRecommendation for {user.email}")
                # Clear existing events for this recommendation
                UserRecommendationEvent.objects.filter(user_recommendation=cached_recommendation).delete()
                print(f"🗑️ Cleared existing events for updated recommendation")
            
            # Create Event objects and UserRecommendationEvent relationships
            saved_events = 0
            failed_events = 0
            
            for rank, event_data in enumerate(daily_recommendations, 1):
                try:
                    # Get or create the Event object
                    event, event_created = Event.objects.get_or_create(
                        event_id=event_data.get('id'),
                        defaults={
                            'title': event_data.get('title', ''),
                            'description': event_data.get('description', ''),
                            'start_date': event_data.get('start_date', ''),
                            'when': event_data.get('when', ''),
                            'address': event_data.get('address', []),
                            'formatted_address': event_data.get('formatted_address', ''),
                            'latitude': event_data.get('event_latitude'),
                            'longitude': event_data.get('event_longitude'),
                            'venue_name': event_data.get('venue_name', ''),
                            'venue_rating': event_data.get('venue_rating'),
                            'venue_reviews': event_data.get('venue_reviews'),
                            'link': event_data.get('link', ''),
                            'thumbnail': event_data.get('thumbnail', ''),
                            'image': event_data.get('image', ''),
                            'event_type': event_data.get('event_type', ''),
                            'has_tickets': event_data.get('has_tickets', False),
                            'ticket_info': event_data.get('ticket_info', []),
                            'ticket_sources': event_data.get('ticket_sources', []),
                            'source': event_data.get('source', 'serp_api')
                        }
                    )
                    
                    # Create the recommendation relationship
                    UserRecommendationEvent.objects.create(
                        user_recommendation=cached_recommendation,
                        event=event,
                        personalization_score=event_data.get('personalization_score', 0),
                        searched_location=f"{latitude},{longitude}",
                        source_query=event_data.get('source_query', 'default'),
                        recommendation_rank=rank
                    )
                    
                    saved_events += 1
                    if event_created:
                        print(f"✅ Created new event: {event_data.get('title', 'Unknown')} (rank {rank})")
                    else:
                        print(f"🔄 Updated existing event: {event_data.get('title', 'Unknown')} (rank {rank})")
                    
                except Exception as e:
                    failed_events += 1
                    print(f"❌ Error saving event {event_data.get('id', 'unknown')}: {e}")
                    continue
            
            cache_time = round((time.time() - cache_start) * 1000, 2)
            print(f"💾 Caching completed in {cache_time}ms")
            print(f"📊 Saved {saved_events} events, {failed_events} failed")
            print(f"✅ Generated {len(daily_recommendations)} event recommendations for {user.email}")
            
            return {
                "results": daily_recommendations,
                "user_profile": user_profile,
                "custom_queries": custom_queries,
                "total_results": len(scored_events)
            }
            
        except Exception as e:
            print(f"❌ Error generating event recommendations for {user.email}: {e}")
            import traceback
            traceback.print_exc()
            raise e

    def add_location_to_search_history(self, user, location_key):
        """Add a location to user's search history (max 10 recent locations)."""
        try:
            # Get current search history
            history = user.search_history_locations if user.search_history_locations else []
            
            # Remove location if it already exists (to move it to front)
            if location_key in history:
                history.remove(location_key)
            
            # Add to front of list
            history.insert(0, location_key)
            
            # Keep only last 10 searches
            history = history[:10]
            
            # Update user
            user.search_history_locations = history
            user.save(update_fields=['search_history_locations'])
            
            print(f"Added {location_key} to search history for {user.email}")
            
        except Exception as e:
            print(f"Error updating search history for {user.email}: {e}")

    def extract_coordinates_from_address(self, address_list):
        """Extract coordinates from address using geocoding."""
        if not address_list:
            return None, None
        
        # Join address components
        full_address = ", ".join(address_list) if isinstance(address_list, list) else str(address_list)
        
        try:
            url = f"https://maps.googleapis.com/maps/api/geocode/json?address={full_address}&key=AIzaSyBfW8nU2EoPK1Zg_bYOSREzqmRDwZfUgbM"
            response = requests.get(url)
            data = response.json()
            
            if data.get("results"):
                location = data["results"][0]["geometry"]["location"]
                return location["lat"], location["lng"]
        except Exception as e:
            print(f"❌ Error geocoding address {full_address}: {e}")
        
        return None, None

    def calculate_distance(self, lat1, lon1, lat2, lon2):
        """Calculate distance between two coordinates using haversine formula."""
        if None in [lat1, lon1, lat2, lon2]:
            return None
        
        R = 3958.8  # Radius of the Earth in miles
        lat1, lon1, lat2, lon2 = map(math.radians, [float(lat1), float(lon1), float(lat2), float(lon2)])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        distance = R * c
        return round(distance, 2)

    async def get_coordinates_from_zip(self, zip_code):
        """Reuse existing geocoding method."""
        events_api = EventsAPIView()
        return await events_api.get_coordinates_from_zip(zip_code)


class EmailSubscriptionAPIView(APIView):
    """API endpoint for email newsletter subscriptions with Brevo integration."""
    permission_classes = [AllowAny]
    parser_classes = [JSONParser]
    
    def post(self, request):
        """Handle email subscription with Brevo integration."""
        try:
            email = request.data.get('email', '').strip().lower()
            
            # Validate email
            if not email:
                return Response({
                    'error': 'Email address is required',
                    'success': False
                }, status=status.HTTP_400_BAD_REQUEST)
            
            # Basic email validation
            import re
            email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
            if not re.match(email_pattern, email):
                return Response({
                    'error': 'Please enter a valid email address',
                    'success': False
                }, status=status.HTTP_400_BAD_REQUEST)
            
            # Import Brevo SDK
            import sib_api_v3_sdk
            from sib_api_v3_sdk.rest import ApiException
            
            # Configure Brevo API
            configuration = sib_api_v3_sdk.Configuration()
            configuration.api_key['api-key'] = settings.BREVO_API_KEY
            
            # Create API instance
            api_instance = sib_api_v3_sdk.ContactsApi(sib_api_v3_sdk.ApiClient(configuration))
            
            # Check if email already exists in database
            from .models import EmailSubscription
            subscription, created = EmailSubscription.objects.get_or_create(
                email=email,
                defaults={
                    'is_active': True,
                    'source': 'website'
                }
            )
            
            if not created and subscription.is_active:
                return Response({
                    'message': 'You are already subscribed to our newsletter',
                    'success': True
                }, status=status.HTTP_200_OK)
            
            # Prepare contact data for Brevo
            create_contact = sib_api_v3_sdk.CreateContact(
                email=email,
                list_ids=[int(settings.BREVO_LIST_ID)]  # Add to newsletter list
            )
            
            try:
                # Create contact in Brevo
                api_response = api_instance.create_contact(create_contact)
                
                # Update subscription with Brevo contact ID
                subscription.brevo_contact_id = str(api_response.id) if hasattr(api_response, 'id') else None
                subscription.is_active = True
                subscription.unsubscribed_at = None
                subscription.save()
                
                return Response({
                    'message': 'Thank you for subscribing! Check your email for confirmation.',
                    'success': True
                }, status=status.HTTP_201_CREATED)
                
            except ApiException as e:
                # Handle Brevo API errors
                if e.status == 400:
                    error_body = json.loads(e.body) if e.body else {}
                    error_code = error_body.get('code', '')
                    
                    if error_code == 'duplicate_parameter':
                        # Contact already exists in Brevo, just update our database
                        subscription.is_active = True
                        subscription.unsubscribed_at = None
                        subscription.save()
                        
                        return Response({
                            'message': 'You are already subscribed to our newsletter',
                            'success': True
                        }, status=status.HTTP_200_OK)
                    
                    return Response({
                        'error': 'Invalid email address',
                        'success': False
                    }, status=status.HTTP_400_BAD_REQUEST)
                
                # Log other API errors
                print(f"Brevo API error: {e}")
                return Response({
                    'error': 'There was an error processing your subscription. Please try again.',
                    'success': False
                }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
            
        except Exception as e:
            # Log unexpected errors
            print(f"Email subscription error: {e}")
            return Response({
                'error': 'An unexpected error occurred. Please try again.',
                'success': False
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
