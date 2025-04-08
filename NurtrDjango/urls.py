from django.contrib import admin
from django.urls import path, include, re_path
from rest_framework_nested import routers
from .views import UserViewSet
from users.views import UserCreateView ,ChildViewSet
from django.conf.urls.static import static
from NurtrDjango import settings
from .views import PlacesAPIView, serve_image
from django.http import HttpResponse
from django.urls import get_resolver

def show_urls(request):
    """Temporary view to show all registered URLs."""
    urls = []
    for url_pattern in get_resolver().url_patterns:
        if hasattr(url_pattern, 'url_patterns'):
            for sub_pattern in url_pattern.url_patterns:
                urls.append(str(sub_pattern.pattern))
        else:
            urls.append(str(url_pattern.pattern))
    return HttpResponse("<br>".join(urls))

router = routers.DefaultRouter(trailing_slash=True)
router.register(r'users', UserViewSet, basename='user')
router.register(r'children', ChildViewSet)

def home_view(request):
    return HttpResponse("Welcome to the Nurtr Django backend!")

urlpatterns = [
    path('', home_view),
    path('admin/', admin.site.urls),
    path('api/', include(router.urls)),  
    path('api/places/', PlacesAPIView.as_view(), name='places_api'),
    path('api/places/place-details/<str:place_id>/', PlacesAPIView.as_view(), name='place_details'),
    path('debug/urls/', show_urls),
    # Add a direct path to the authenticate endpoint as a fallback
    re_path(r'^api/users/authenticate/$', UserViewSet.as_view({'post': 'authenticate'}), name='user-authenticate'),
    #path('api/users/', UserCreateView.as_view(), name='user-create'), 
    #path('api/serve-image/', serve_image, name='serve_image'),
]
