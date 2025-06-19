from django.db import models
from django.contrib.auth import get_user_model
import uuid

class Place(models.Model):
    """Model to store information about places."""
    title = models.CharField(max_length=255)
    description = models.TextField(blank=True, null=True)  # Added as requested
    category = models.CharField(max_length=255, blank=True, null=True)
    place_id = models.CharField(max_length=255, unique=True, db_index=True)
    data_id = models.CharField(max_length=255, blank=True, null=True)
    reviews_link = models.URLField(max_length=1024, blank=True, null=True)
    photos_link = models.URLField(max_length=1024, blank=True, null=True)
    latitude = models.DecimalField(max_digits=10, decimal_places=7, blank=True, null=True) # Using DecimalField for precision
    longitude = models.DecimalField(max_digits=10, decimal_places=7, blank=True, null=True) # Using DecimalField for precision
    type = models.CharField(max_length=100, blank=True, null=True) # Single primary type
    types = models.JSONField(default=list, blank=True) # List of all types
    address = models.TextField(blank=True, null=True) # Using TextField for potentially long addresses
    extensions = models.JSONField(default=dict, blank=True) # Store extensions structure
    display_name = models.CharField(max_length=255, blank=True, null=True) # From displayName.text
    formatted_address = models.TextField(blank=True, null=True) # From formattedAddress
    rating = models.FloatField(blank=True, null=True) # FloatField for rating
    reviews = models.IntegerField(blank=True, null=True) # IntegerField for review count (userRatingCount)
    hours = models.JSONField(null=True, blank=True) # Stores operating hours JSON
    place_images = models.JSONField(default=list, blank=True) # List of image URLs
    reviews_list = models.JSONField(null=True, blank=True) # Stores the list of reviews JSON
    popular_times = models.JSONField(null=True, blank=True) # Stores popular times JSON
    # Consider adding timestamp fields
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return self.title or self.place_id

    class Meta:
        verbose_name = "Place"
        verbose_name_plural = "Places"
        ordering = ['title']


class UserRecommendation(models.Model):
    """Model to store daily aggregated recommendations for users."""
    RECOMMENDATION_TYPES = [
        ('places', 'Places'),
        ('events', 'Events'),
    ]
    
    user = models.ForeignKey(get_user_model(), on_delete=models.CASCADE, related_name='daily_recommendations')
    recommendation_type = models.CharField(max_length=10, choices=RECOMMENDATION_TYPES, default='places')
    date_generated = models.DateField(help_text="Date these recommendations were generated")
    user_profile_hash = models.CharField(max_length=64, help_text="Hash of user profile for cache invalidation")
    
    # Relationship to actual Place objects instead of JSON
    recommended_places = models.ManyToManyField(Place, through='UserRecommendationPlace', related_name='user_recommendations', blank=True)
    
    # Keep these for metadata
    custom_queries = models.JSONField(help_text="Generated search queries used")
    locations_searched = models.JSONField(help_text="List of locations searched for these recommendations")
    total_results = models.IntegerField(default=0)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        unique_together = ['user', 'date_generated', 'recommendation_type']
        verbose_name = "User Daily Recommendation"
        verbose_name_plural = "User Daily Recommendations"
        ordering = ['-date_generated', '-updated_at']
    
    def __str__(self):
        return f"{self.recommendation_type.title()} recommendations for {self.user.email} on {self.date_generated}"


class Event(models.Model):
    """Model to store information about events."""
    title = models.CharField(max_length=255)
    description = models.TextField(blank=True, null=True)
    event_id = models.CharField(max_length=255, unique=True, db_index=True)
    
    # Date and time fields
    start_date = models.CharField(max_length=255, blank=True, null=True)
    when = models.CharField(max_length=255, blank=True, null=True)
    
    # Location fields
    address = models.JSONField(default=list, blank=True)
    formatted_address = models.TextField(blank=True, null=True)
    latitude = models.DecimalField(max_digits=10, decimal_places=7, blank=True, null=True)
    longitude = models.DecimalField(max_digits=10, decimal_places=7, blank=True, null=True)
    
    # Venue information
    venue_name = models.CharField(max_length=255, blank=True, null=True)
    venue_rating = models.FloatField(blank=True, null=True)
    venue_reviews = models.IntegerField(blank=True, null=True)
    
    # Event details
    link = models.URLField(max_length=1024, blank=True, null=True)
    thumbnail = models.URLField(max_length=1024, blank=True, null=True)
    image = models.URLField(max_length=1024, blank=True, null=True)
    event_type = models.CharField(max_length=100, blank=True, null=True)
    
    # Ticket information
    has_tickets = models.BooleanField(default=False)
    ticket_info = models.JSONField(default=list, blank=True)
    ticket_sources = models.JSONField(default=list, blank=True)
    
    # Additional data
    event_location_map = models.JSONField(default=dict, blank=True)
    source = models.CharField(max_length=50, default="serp_api")
    last_updated = models.DateTimeField(auto_now=True)
    
    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    def __str__(self):
        return self.title or self.event_id
    
    class Meta:
        verbose_name = "Event"
        verbose_name_plural = "Events"
        ordering = ['title']


class UserRecommendationPlace(models.Model):
    """Through model to store recommendation-specific data for each place."""
    user_recommendation = models.ForeignKey(UserRecommendation, on_delete=models.CASCADE)
    place = models.ForeignKey(Place, on_delete=models.CASCADE)
    
    # Store recommendation-specific data
    personalization_score = models.IntegerField(default=0, help_text="How well this place matches the user's profile")
    personalized_explanation = models.TextField(blank=True, null=True, help_text="AI-generated explanation for why this place was recommended")
    searched_location = models.CharField(max_length=255, help_text="Which location search found this place")
    source_query = models.TextField(help_text="Which search query found this place")
    recommendation_rank = models.IntegerField(help_text="Rank of this place in the recommendation list (1=best)")
    
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        unique_together = ['user_recommendation', 'place']
        ordering = ['recommendation_rank']
        verbose_name = "User Recommendation Place"
        verbose_name_plural = "User Recommendation Places"
    
    def __str__(self):
        return f"{self.place.title} (Rank #{self.recommendation_rank} for {self.user_recommendation.user.email})"


class UserEventRecommendation(models.Model):
    """Model to store daily aggregated event recommendations for users."""
    user = models.ForeignKey(get_user_model(), on_delete=models.CASCADE, related_name='daily_event_recommendations')
    date_generated = models.DateField(help_text="Date these event recommendations were generated")
    user_profile_hash = models.CharField(max_length=64, help_text="Hash of user profile for cache invalidation")
    
    # Relationship to actual Event objects
    recommended_events = models.ManyToManyField(Event, through='UserRecommendationEvent', related_name='user_recommendations')
    
    # Keep these for metadata
    custom_queries = models.JSONField(help_text="Generated search queries used")
    locations_searched = models.JSONField(help_text="List of locations searched for these recommendations")
    total_results = models.IntegerField(default=0)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        unique_together = ['user', 'date_generated']
        verbose_name = "User Daily Event Recommendation"
        verbose_name_plural = "User Daily Event Recommendations"
        ordering = ['-date_generated', '-updated_at']
    
    def __str__(self):
        return f"Event recommendations for {self.user.email} on {self.date_generated}"


class UserRecommendationEvent(models.Model):
    """Through model to store recommendation-specific data for each event."""
    user_recommendation = models.ForeignKey(UserEventRecommendation, on_delete=models.CASCADE)
    event = models.ForeignKey(Event, on_delete=models.CASCADE)
    
    # Store recommendation-specific data
    personalization_score = models.IntegerField(default=0, help_text="How well this event matches the user's profile")
    personalized_explanation = models.TextField(blank=True, null=True, help_text="AI-generated explanation for why this event was recommended")
    searched_location = models.CharField(max_length=255, help_text="Which location search found this event")
    source_query = models.TextField(help_text="Which search query found this event")
    recommendation_rank = models.IntegerField(help_text="Rank of this event in the recommendation list (1=best)")
    
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        unique_together = ['user_recommendation', 'event']
        ordering = ['recommendation_rank']
        verbose_name = "User Recommendation Event"
        verbose_name_plural = "User Recommendation Events"
    
    def __str__(self):
        return f"{self.event.title} (Rank #{self.recommendation_rank} for {self.user_recommendation.user.email})"