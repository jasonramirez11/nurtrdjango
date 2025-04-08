from django.contrib.auth import get_user_model
from django.contrib.auth.password_validation import validate_password
from rest_framework import serializers
from django.core.exceptions import ValidationError
from .models import  Child
import json
User = get_user_model()

class UserSerializer(serializers.ModelSerializer):
    password = serializers.CharField(write_only=True, required=False)  
    is_active = serializers.BooleanField(required=False)  # Allow updating is_active
    favorites = serializers.JSONField(required=False)

    class Meta:
        model = User
        fields = ['id', 'email', 'password', 'is_active', 'phone', 'name', 'joining_date', 'favorites']  
        read_only_fields = ['joining_date']  
        
    def validate_favorites(self, value):
        """Validate that favorites is a dictionary with boolean values."""
        if not isinstance(value, dict):
            raise serializers.ValidationError("Favorites must be a dictionary")
        for key, val in value.items():
            if not isinstance(val, bool):
                raise serializers.ValidationError(f"Value for {key} must be a boolean")
        return value
        
    def update(self, instance, validated_data):
        """Update user details like email, phone, name, is_active, or favorites."""
        password = validated_data.pop('password', None)  

        # Handle favorites update
        favorites = validated_data.pop('favorites', None)
        if favorites is not None:
            instance.favorites = json.dumps(favorites)

        for attr, value in validated_data.items():
            setattr(instance, attr, value)  # Update fields including is_active

        if password:
            try:
                validate_password(password, instance)
                instance.set_password(password)
            except ValidationError as e:
                raise serializers.ValidationError({'password': e.messages})

        instance.save()
        return instance

class ChildSerializer(serializers.ModelSerializer):
    class Meta:
        model = Child
        fields = '__all__'