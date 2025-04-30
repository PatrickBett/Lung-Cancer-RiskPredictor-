from django.db import models
from django.utils import timezone

class PredictionRecord(models.Model):
    """Model to store prediction records"""
    GENDER_CHOICES = [
        ('M', 'Male'),
        ('F', 'Female'),
    ]
    
    YES_NO_CHOICES = [
        (1, 'No'),
        (2, 'Yes'),
    ]
    
    # User Information
    gender = models.CharField(max_length=1, choices=GENDER_CHOICES)
    age = models.PositiveIntegerField()
    
    # Risk Factors
    smoking = models.IntegerField(choices=YES_NO_CHOICES)
    yellow_fingers = models.IntegerField(choices=YES_NO_CHOICES)
    anxiety = models.IntegerField(choices=YES_NO_CHOICES)
    peer_pressure = models.IntegerField(choices=YES_NO_CHOICES)
    chronic_disease = models.IntegerField(choices=YES_NO_CHOICES)
    fatigue = models.IntegerField(choices=YES_NO_CHOICES)
    allergy = models.IntegerField(choices=YES_NO_CHOICES)
    wheezing = models.IntegerField(choices=YES_NO_CHOICES)
    alcohol_consuming = models.IntegerField(choices=YES_NO_CHOICES)
    coughing = models.IntegerField(choices=YES_NO_CHOICES)
    shortness_of_breath = models.IntegerField(choices=YES_NO_CHOICES)
    swallowing_difficulty = models.IntegerField(choices=YES_NO_CHOICES)
    chest_pain = models.IntegerField(choices=YES_NO_CHOICES)
    
    # Prediction results
    prediction = models.BooleanField(null=True)
    probability = models.FloatField(null=True)
    created_at = models.DateTimeField(default=timezone.now)
    
    def __str__(self):
        return f"Prediction for {self.get_gender_display()}, Age: {self.age} ({self.created_at.strftime('%Y-%m-%d %H:%M')})"
    
    class Meta:
        ordering = ['-created_at']