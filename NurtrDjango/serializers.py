from django.contrib.auth import authenticate
from rest_framework import serializers
from django.contrib.auth import get_user_model
from django.contrib.auth import get_user_model
from rest_framework import serializers

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


User = get_user_model()
