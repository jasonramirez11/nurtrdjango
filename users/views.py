from rest_framework import generics, viewsets, status
from rest_framework.response import Response
from users.serializers import UserSerializer, ChildSerializer
from users.models import CustomUser, Child
from rest_framework.permissions import IsAuthenticated
from NurtrDjango.models import UserActivity
from NurtrDjango.serializers import UserActivitySerializer

class UserCreateView(generics.CreateAPIView):
    queryset = CustomUser.objects.all()
    serializer_class = UserSerializer

class ChildViewSet(viewsets.ModelViewSet):
    queryset = Child.objects.all()
    serializer_class = ChildSerializer
    permission_classes = [IsAuthenticated]

    def get_queryset(self):
        return Child.objects.filter(user=self.request.user)
    
    def perform_create(self, serializer):
        serializer.save(user=self.request.user)


class UserActivityViewSet(viewsets.ModelViewSet):
    serializer_class = UserActivitySerializer
    permission_classes = [IsAuthenticated]
    
    def get_queryset(self):
        return UserActivity.objects.filter(user=self.request.user).select_related('place')
    
    def create(self, request, *args, **kwargs):
        print(f"🔍 UserActivity CREATE - User: {request.user.email}")
        print(f"🔍 UserActivity CREATE - Data: {request.data}")
        
        serializer = self.get_serializer(data=request.data)
        if not serializer.is_valid():
            print(f"❌ UserActivity CREATE - Validation errors: {serializer.errors}")
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        
        try:
            self.perform_create(serializer)
            print(f"✅ UserActivity CREATE - Success: {serializer.data}")
            headers = self.get_success_headers(serializer.data)
            return Response(serializer.data, status=status.HTTP_201_CREATED, headers=headers)
        except Exception as e:
            print(f"❌ UserActivity CREATE - Exception: {str(e)}")
            return Response({'error': str(e)}, status=status.HTTP_400_BAD_REQUEST)
    
    def perform_create(self, serializer):
        serializer.save(user=self.request.user)