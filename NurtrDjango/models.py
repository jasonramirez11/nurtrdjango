from django.db import models
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