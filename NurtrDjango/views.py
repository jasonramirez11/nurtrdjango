import json
import os
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

load_dotenv()

IP_INFO_API_TOKEN = os.getenv("IP_INFO_API_TOKEN")
IMAGE_DOWNLOAD_URL = os.getenv("IMAGE_DOWNLOAD_URL")
TESTING = os.getenv("TESTING", "False").lower() == "true"

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
            print(f"Successfully uploaded {local_path} to {blob.name}")
            # Make the blob publicly viewable (optional, adjust as needed)
            # blob.make_public()
            return blob.public_url
        except Exception as e:
            print(f"Error uploading {local_path} to {blob.name}: {e}")
            time.sleep(random.randint(2,3))

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
    try:
        response = requests.get(image_url, timeout=10)  # Set a timeout for the request
        if response.status_code == 200:
            with open(local_file_path, 'wb') as f:
                f.write(response.content)
            return local_file_path
        else:
            print(f"Failed to download {image_url}: Status {response.status_code}")
            return None
    except Exception as e:
        print(f"Error downloading {image_url}: {str(e)}")
        return None

def check_existing_images(bucket_name: str, place_id: str) -> List[str]:
    """
    Check if images for the given place_id exist in the specified GCS bucket.

    Args:
        bucket_name (str): The name of the GCS bucket.
        place_id (str): The ID of the place to check.

    Returns:
        List[str]: A list of existing image URLs if found, otherwise an empty list.
    """
    print(f"Checking for existing images in bucket '{bucket_name}' for Place ID '{place_id}'...")
    start_time = time.time()
    storage_client = Client()
    bucket = storage_client.bucket(bucket_name)
    prefix = f"places/{place_id}/"  # Assuming images are stored under a folder named after the place_id

    blobs = bucket.list_blobs(prefix=prefix)  # List blobs with the specified prefix
    existing_images = [blob.public_url for blob in blobs]  # Adjust based on your image format

    end_time = time.time()
    print(f"Finished checking for existing images for Place ID '{place_id}'.")
    print(f"Total time for checking {len(existing_images)} existing images: {end_time - start_time:.2f} seconds")
    return existing_images

def process_images_for_place(place_id, image_urls):
    """
    Process and download images for a place (synchronous).
    This entire function runs in a worker thread.
    
    Args:
        place_id (str): The place ID.
        image_urls (list): List of image URLs to process.
        
    Returns:
        list: Processed image URLs from GCS.
    """
    if not image_urls:
        print(f"No image URLs provided for place ID {place_id}")
        return []
        
    try:
        # Check for existing images first
        existing_image_links = check_existing_images(bucket_name='nurtr-places', place_id=place_id)
        
        if existing_image_links:
            print(f"Using existing images for place ID {place_id}")
            return existing_image_links
            
        # Create temp directory if needed
        os.makedirs("temp_images", exist_ok=True)
        
        # Download images sequentially (since the entire function is already in a thread)
        local_image_paths = []
        for url in image_urls:
            path = download_image(url)
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
        types = data.get("types", [])
        min_price = data.get("minPrice", 0)
        max_price = data.get("maxPrice", 500)
        filters = data.get("filters", [])
        page = int(data.get("page", 1))
        #items_per_page = 4

        #print('places request ', request)
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

            all_results = await self.fetch_place_images(all_results, max_images=1)
            all_results = [p for p in all_results if p.get("imagePlaces", [])]

            print(f"Getting or uploading images for {len(all_results)} places...")

            """
            async def process_place(place):
                print(f"Processing place ID {place['id']}...")
                place_id = place["id"]
                image_places = place.get("imagePlaces", [])[:3]  # Limit to first 3 photos

                if not image_places:
                    print(f"No images found for place ID {place_id}")
                    return place

                data = {'image_urls': image_places, 'place_id': place["id"]}

                try:
                    async with aiohttp.ClientSession() as session:
                        async with session.post(IMAGE_DOWNLOAD_URL, json=data, timeout=30) as response:
                            if response.status == 200:
                                response_data = await response.json()
                                print(f"Image download successful for place ID {place['id']}")
                                place["imagePlaces"] = response_data.get("gcs_image_urls", [])
                            else:
                                print(f"Image download failed for place ID {place['id']}: {response.status}")
                                place["imagePlaces"] = []
                except Exception as e:
                    print(f"Error during image download for place ID {place['id']}: {e}")
                    place["imagePlaces"] = []

                return place
            
            start_time = time.time()
            all_results = await asyncio.gather(*[process_place(place) for place in all_results])
            end_time = time.time()
            """

            start_time = time.time()
            with ThreadPoolExecutor(max_workers=20) as executor:
                # Prepare processing tasks for all places
                future_to_place = {}
                for place in all_results:
                    image_urls = place.get("imagePlaces", [])[:3]
                    if image_urls:
                        future = executor.submit(
                            process_images_for_place, 
                            place["id"], 
                            image_urls
                        )
                        future_to_place[future] = place
                
                # Process completed tasks as they finish
                for future in as_completed(future_to_place):
                    place = future_to_place[future]
                    try:
                        processed_urls = future.result()
                        place["imagePlaces"] = processed_urls
                        print(f"Processed {len(processed_urls)} images for place {place['id']}")
                    except Exception as e:
                        print(f"Error processing place {place['id']}: {e}")
                        place["imagePlaces"] = []

            end_time = time.time()
            execution_time = round((end_time - start_time) * 1000, 2)
            print(f"Finished processing images for {len(all_results)} places. Execution time: {execution_time} ms")
        
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
        
        #photos = details_data.get("photos", [])[:3]  # Limit to first 5 photos
        #image_places = []

        def process_place(place):
            print(f"Processing place ID {place['id']}...")
            place_id = place["id"]
            image_photos = place.get("photos", [])[:3]

            if not image_photos:
                print(f"No images found for place ID {place_id}")
                return place

            data = {'image_urls': image_photos, 'place_id': place["id"]}

            try:
                response = requests.post(IMAGE_DOWNLOAD_URL, json=data)
                if response.status_code == 200:
                    print(f"Image download successful for place ID {place['id']}")
                    place["images"] = response.json().get("gcs_image_urls", [])
                else:
                    print(f"Image download failed for place ID {place['id']}: {response.status_code}")
                    place["images"] = []
            except Exception as e:
                print(f"Error during image download for place ID {place['id']}: {e}")
                place["images"] = []

            return place

        return process_place(details_data)

        """
        for photo in photos:
            name = photo.get("name")
            if name:
                image_url = f"https://places.googleapis.com/v1/{name}/media?key={self.API_KEY}&maxHeightPx=600"
                #proxied_image_url = f"/api/serve-image?url={google_image_url}"  # Proxy through your backend
                image_places.append(image_url)
        details_data["images"] = image_places  # Attach new key
        """

    def get(self, request, place_id=None, is_authenticated=False):
        """Endpoint to fetch place details by ID."""
        if not place_id:
            return Response({"error": "Place ID is required"}, status=HTTP_400_BAD_REQUEST)

        print('request query params', request.query_params)
        is_authenticated = request.query_params.get('is_authenticated', 'false').lower() == 'true'

        # Rate limiting for unauthenticated users
        if not is_authenticated:
            rate_limit_response = self.check_rate_limit(
                request, 
                MAX_UNAUTHENTICATED_REQUESTS_PER_IP, 
                CACHE_TIMEOUT_SECONDS
            )
            if rate_limit_response:
                return rate_limit_response
            
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
                "maxResultCount": 5 if TESTING else 17,
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

            """
            local_image_paths = []
            for image_place in image_places:
                # Download the image and save locally
                local_file_path = f"temp_images/{uuid.uuid4()}.jpg"
                await self.download_image(image_place, local_file_path)
                local_image_paths.append(local_file_path)

            gcs_place_image_links = upload_place_images_to_bucket(
                bucket_name='nurtr-places',
                place_id=place["id"],
                image_paths=local_image_paths
            )

            place["imagePlacesGCS"] = gcs_place_image_links
            """

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
