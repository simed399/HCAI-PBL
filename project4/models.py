from django.db import models
import uuid

class Participant(models.Model):
    id          = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    assigned_to = models.CharField(max_length=50)  # 'explanation' or 'control'
    created_at  = models.DateTimeField(auto_now_add=True)

class PreSurvey(models.Model):
    participant = models.OneToOneField(Participant, on_delete=models.CASCADE)
    age         = models.IntegerField()
    gender      = models.CharField(max_length=20)
    familiarity = models.IntegerField()  # Likert 1-7

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