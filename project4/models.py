from django.db import models
import uuid
import json

class Participant(models.Model):
    id          = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    assigned_to = models.CharField(max_length=50)  # 'explanation' or 'control'
    created_at  = models.DateTimeField(auto_now_add=True)
    display_name = models.CharField(max_length=100, default="Participant")

class PreSurvey(models.Model):
    participant = models.OneToOneField(Participant, on_delete=models.CASCADE)
    age         = models.IntegerField()
    gender      = models.CharField(max_length=20)
    familiarity = models.IntegerField()  # Likert 1-7

class QuizRating(models.Model):
    """Ratings collected during the active learning quiz phase"""
    participant = models.ForeignKey(Participant, on_delete=models.CASCADE)
    movie_id    = models.IntegerField()
    rating      = models.FloatField()
    iteration   = models.IntegerField()  # which round of active learning
    timestamp   = models.DateTimeField(auto_now_add=True)
    explanation_shown = models.BooleanField(default=False)

class HeldOutRating(models.Model):
    participant = models.ForeignKey(Participant, on_delete=models.CASCADE)
    movie_id    = models.IntegerField()
    rating      = models.FloatField()
    timestamp   = models.DateTimeField(auto_now_add=True)

class PostSurvey(models.Model):
    participant = models.OneToOneField(Participant, on_delete=models.CASCADE)
    trust       = models.IntegerField()
    transparency= models.IntegerField()
    satisfaction= models.IntegerField()

class StudySession(models.Model):
    """Stores session data and results for analysis"""
    participant = models.OneToOneField(Participant, on_delete=models.CASCADE)
    user_vector = models.TextField()  # JSON serialized numpy array
    final_rmse = models.FloatField(null=True, blank=True)
    completion_status = models.CharField(max_length=50, default='in_progress')
    created_at = models.DateTimeField(auto_now_add=True)
    completed_at = models.DateTimeField(null=True, blank=True)

class Feedback(models.Model):
    """Post-study feedback"""
    participant = models.ForeignKey(Participant, on_delete=models.CASCADE)
    helpfulness = models.IntegerField(null=True, blank=True)
    comments = models.TextField(blank=True)
    timestamp = models.DateTimeField(auto_now_add=True)