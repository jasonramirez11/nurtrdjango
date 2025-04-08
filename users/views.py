from rest_framework import generics
from users.serializers import UserSerializer, ChildSerializer
from users.models import CustomUser, Child
from rest_framework import viewsets
from rest_framework.permissions import IsAuthenticated

class UserCreateView(generics.CreateAPIView):
    queryset = CustomUser.objects.all()
    serializer_class = UserSerializer

class ChildViewSet(viewsets.ModelViewSet):
    queryset = Child.objects.all()
    serializer_class = ChildSerializer
    permission_classes = []  

    def get_queryset(self):
        user_id = self.request.query_params.get('user')  
        if user_id:
            return Child.objects.filter(user_id=user_id)
        return super().get_queryset()