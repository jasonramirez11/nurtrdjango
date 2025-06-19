from django.contrib import admin
from users.models import Child
from .models import UserRecommendation, UserRecommendationPlace, UserRecommendationPlace


@admin.register(Child)
class ChildAdmin(admin.ModelAdmin):
    list_display = ('id', 'name', 'age', 'user')
    search_fields = ('name', 'user__email')
    list_filter = ('age',)


class UserRecommendationPlaceInline(admin.TabularInline):
    model = UserRecommendationPlace
    extra = 0
    readonly_fields = ('place', 'personalization_score', 'searched_location', 'source_query', 'recommendation_rank', 'created_at')
    fields = ('recommendation_rank', 'place', 'personalization_score', 'searched_location', 'source_query', 'created_at')
    ordering = ['recommendation_rank']

@admin.register(UserRecommendation)
class UserRecommendationAdmin(admin.ModelAdmin):
    list_display = ('user', 'date_generated', 'total_results', 'get_locations_count', 'get_places_count', 'updated_at')
    search_fields = ('user__email', 'user__name')
    list_filter = ('date_generated', 'created_at', 'updated_at')
    readonly_fields = ('user_profile_hash', 'created_at', 'updated_at')
    ordering = ('-date_generated', '-updated_at')
    inlines = [UserRecommendationPlaceInline]
    
    def get_locations_count(self, obj):
        """Show number of locations searched."""
        return len(obj.locations_searched) if obj.locations_searched else 0
    get_locations_count.short_description = 'Locations Searched'
    
    def get_places_count(self, obj):
        """Show number of recommended places."""
        return obj.recommended_places.count()
    get_places_count.short_description = 'Recommended Places'

@admin.register(UserRecommendationPlace)
class UserRecommendationPlaceAdmin(admin.ModelAdmin):
    list_display = ('get_user', 'place', 'recommendation_rank', 'personalization_score', 'searched_location', 'created_at')
    search_fields = ('user_recommendation__user__email', 'place__title', 'place__place_id')
    list_filter = ('recommendation_rank', 'searched_location', 'created_at')
    ordering = ['user_recommendation__date_generated', 'recommendation_rank']
    
    def get_user(self, obj):
        return obj.user_recommendation.user.email
    get_user.short_description = 'User'
