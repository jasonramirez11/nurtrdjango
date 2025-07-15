from django.contrib.auth import authenticate
from rest_framework import serializers
from django.contrib.auth import get_user_model
from .models import UserActivity, Place

User = get_user_model()

class LoginSerializer(serializers.Serializer):
    email = serializers.EmailField(required=True)
    password = serializers.CharField(required=True, write_only=True)

    def validate(self, data):
        email = data.get('email')
        password = data.get('password')

        # Check for missing fields
        if not email:
            raise serializers.ValidationError({"email": "Email is required."})
        if not password:
            raise serializers.ValidationError({"password": "Password is required."})

        # Authenticate the user
        user = authenticate(email=email, password=password)

        if not user:
            # User does not exist or password is incorrect
            if not User.objects.filter(email=email).exists():
                raise serializers.ValidationError({"email": "No account found with this email."})
            else:
                raise serializers.ValidationError({"password": "Incorrect password."})

        # Check if the user is active
        if not user.is_active:
            raise serializers.ValidationError({"non_field_errors": "This account is inactive."})

        return user


class UserActivitySerializer(serializers.ModelSerializer):
    # Include comprehensive place data
    place_title = serializers.CharField(source='place.title', read_only=True)
    place_address = serializers.CharField(source='place.formatted_address', read_only=True)
    place_description = serializers.CharField(source='place.description', read_only=True)
    place_category = serializers.CharField(source='place.category', read_only=True)
    place_id_external = serializers.CharField(source='place.place_id', read_only=True)
    place_latitude = serializers.DecimalField(source='place.latitude', max_digits=10, decimal_places=7, read_only=True)
    place_longitude = serializers.DecimalField(source='place.longitude', max_digits=10, decimal_places=7, read_only=True)
    place_rating = serializers.FloatField(source='place.rating', read_only=True)
    place_reviews = serializers.IntegerField(source='place.reviews', read_only=True)
    place_type = serializers.CharField(source='place.type', read_only=True)
    place_types = serializers.JSONField(source='place.types', read_only=True)
    place_hours = serializers.JSONField(source='place.hours', read_only=True)
    place_images = serializers.JSONField(source='place.place_images', read_only=True)
    place_reviews_link = serializers.URLField(source='place.reviews_link', read_only=True)
    place_photos_link = serializers.URLField(source='place.photos_link', read_only=True)
    
    class Meta:
        model = UserActivity
        fields = [
            'id', 'place', 'place_title', 'place_address', 'place_description',
            'place_category', 'place_id_external', 'place_latitude', 'place_longitude',
            'place_rating', 'place_reviews', 'place_type', 'place_types',
            'place_hours', 'place_images', 'place_reviews_link', 'place_photos_link',
            'scheduled_date', 'start_time', 'end_time',
            'title', 'notes', 'status', 'created_at', 'updated_at'
        ]
        read_only_fields = [
            'id', 'created_at', 'updated_at', 'place_title', 'place_address',
            'place_description', 'place_category', 'place_id_external', 'place_latitude',
            'place_longitude', 'place_rating', 'place_reviews', 'place_type', 'place_types',
            'place_hours', 'place_images', 'place_reviews_link', 'place_photos_link'
        ]
    
    def to_internal_value(self, data):
        # Convert place_id string to Place object before validation
        if 'place' in data and isinstance(data['place'], str):
            try:
                place = Place.objects.get(place_id=data['place'])
                data = data.copy()  # Don't modify original data
                data['place'] = place.id  # Use the database primary key
                print(f"✅ Converted place_id '{data['place']}' to Place pk: {place.id}")
            except Place.DoesNotExist:
                print(f"❌ Place with place_id '{data['place']}' not found")
                # Let the validation handle this error
        
        return super().to_internal_value(data)
    
    def validate_place(self, value):
        print(f"🔍 Validating place: {value} (type: {type(value)})")
        # At this point, value should be a Place instance or pk
        if hasattr(value, 'id'):
            print(f"✅ Place object validated: {value.id} - {value.title}")
        else:
            print(f"✅ Place pk validated: {value}")
        return value
    
    def validate(self, data):
        start_time = data.get('start_time')
        end_time = data.get('end_time')
        
        if start_time and end_time and start_time >= end_time:
            raise serializers.ValidationError("End time must be after start time")
        
        return data


User = get_user_model()
