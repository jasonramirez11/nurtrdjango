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
from .models import Place # Import the Place model
from django.core.exceptions import ObjectDoesNotExist
from asgiref.sync import sync_to_async
import concurrent.futures
import logging

load_dotenv()

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

                asyncio.create_task(save_or_update_db(place_data, db_place))
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

                # Save/update DB with combined data in background
                asyncio.create_task(save_or_update_db(place_data, db_place))

                # Return the updated place_data
                place_data['source'] = "database_updated"
                return place_data
            else:
                # If no existing place_data, use API results directly
                api_results["source"] = "api"

                # Start the save/update operation in the background
                asyncio.create_task(save_or_update_db(api_results, db_place))

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

    async def async_fetch_all_places(self, params, session, max_results=60):
        """Fetches all places asynchronously using the new Places API v2."""
        print(f"SERPAPI API Key: {SERP_API_KEY}")

        radius = params["radius"]
        location = params["location"]

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

        def fetch_page_sync(query_str, category_val, start_val=start_val):
            page_params_sync = base_params.copy()
            if start_val > 0:
                page_params_sync["start"] = str(start_val)

            print(f"Fetching page with query: {query_str}")
            page_params_sync["q"] = query_str

            try:
                # Direct blocking call to serpapi
                search_result_sync = serpapi.search(page_params_sync)
                results_sync = search_result_sync.get('local_results', [])
                for res_item in results_sync:
                    res_item['category'] = category_val
                return results_sync
            except Exception as e_sync:
                print(f"Error fetching results with params {page_params_sync}: {e_sync}")
                return []

        types = params["types"]
        query_to_category = {}
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
                futures = {
                    executor.submit(fetch_page_sync, q, query_to_category[q]): q 
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
        
        # Combine results from all pages (all_api_results now holds all items)
        # The original page_results variable might be overwritten or differently used below,
        # this example focuses on collecting into all_api_results.
        # Ensure downstream code uses all_api_results.
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
                    print(f"Filtering out result {result.get('title')} - distance {distance_miles} miles exceeds radius {radius_miles} miles")
                    continue
                    
                result['displayName'] = {
                    'text': result['title'],
                }
                result['formattedAddress'] = result.get('address', 'no address found')
                result['userRatingCount'] = result.get('reviews', 0)
                filtered_results.append(result)
            except Exception as e:
                print(f"Error processing result {result}: {e}")
                
        return filtered_results[:max_results]

    async def async_fetch_all_places_prev(self, params, session):
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
                "maxResultCount": 2 if TESTING else 17,
            }

            print(f"max result count is {5 if TESTING else 17}")

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

        print('Example result:', all_results[0] if all_results else "No results found")
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
