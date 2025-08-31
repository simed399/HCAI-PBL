from django.db import models
from django.utils import timezone
import json


class HumanLabelingSession(models.Model):
    """Track human labeling sessions for active learning"""
    
    # Session metadata
    session_id = models.CharField(max_length=100, unique=True)
    strategy = models.CharField(max_length=50)  # uncertainty, margin, entropy
    created_at = models.DateTimeField(default=timezone.now)
    completed_at = models.DateTimeField(null=True, blank=True)
    
    # Progress tracking
    total_samples = models.IntegerField(default=0)
    labeled_samples = models.IntegerField(default=0)
    is_completed = models.BooleanField(default=False)
    
    # Session configuration
    batch_size = models.IntegerField(default=1)
    target_accuracy = models.FloatField(null=True, blank=True)
    max_samples = models.IntegerField(default=100)
    
    # Results tracking
    current_accuracy = models.FloatField(null=True, blank=True)
    initial_accuracy = models.FloatField(null=True, blank=True)
    
    def __str__(self):
        return f"Session {self.session_id} - {self.strategy} ({self.labeled_samples}/{self.total_samples})"
    
    @property
    def progress_percentage(self):
        if self.total_samples == 0:
            return 0
        return min(100, (self.labeled_samples / self.total_samples) * 100)
    
    def is_session_complete(self):
        """Check if session should be completed based on stopping criteria"""
        if self.is_completed:
            return True
        
        # Check if we've reached max samples
        if self.labeled_samples >= self.max_samples:
            return True
            
        # Check if we've reached target accuracy
        if self.target_accuracy and self.current_accuracy and self.current_accuracy >= self.target_accuracy:
            return True
            
        return False


class HumanLabeledSample(models.Model):
    """Store individual human-labeled samples"""
    
    session = models.ForeignKey(HumanLabelingSession, on_delete=models.CASCADE, related_name='samples')
    
    # Sample data
    sample_text = models.TextField()  # Preprocessed text for ML
    sample_original_text = models.TextField(null=True, blank=True)  # Original text for human display
    sample_index = models.IntegerField()  # Index in the original dataset
    
    # Human label
    human_label = models.IntegerField(choices=[(0, 'Negative'), (1, 'Positive')], null=True, blank=True)
    ground_truth_label = models.IntegerField(choices=[(0, 'Negative'), (1, 'Positive')])
    
    # Metadata
    labeled_at = models.DateTimeField(null=True, blank=True)
    uncertainty_score = models.FloatField(null=True, blank=True)
    selection_order = models.IntegerField(default=0)
    
    # Confidence and timing
    labeling_time_seconds = models.FloatField(null=True, blank=True)
    confidence = models.IntegerField(
        choices=[(1, 'Very Unsure'), (2, 'Unsure'), (3, 'Neutral'), (4, 'Sure'), (5, 'Very Sure')],
        null=True, blank=True
    )
    
    class Meta:
        unique_together = ['session', 'sample_index']
        ordering = ['selection_order']
    
    def __str__(self):
        return f"Sample {self.sample_index} - {self.get_human_label_display() or 'Unlabeled'}"
    
    @property
    def is_correct(self):
        """Check if human label matches ground truth"""
        if self.human_label is None:
            return None
        return self.human_label == self.ground_truth_label


class SessionProgress(models.Model):
    """Track model performance during human labeling session"""
    
    session = models.ForeignKey(HumanLabelingSession, on_delete=models.CASCADE, related_name='progress_snapshots')
    
    # Progress metrics
    samples_labeled = models.IntegerField()
    test_accuracy = models.FloatField()
    timestamp = models.DateTimeField(default=timezone.now)
    
    # Additional metrics
    precision = models.FloatField(null=True, blank=True)
    recall = models.FloatField(null=True, blank=True)
    f1_score = models.FloatField(null=True, blank=True)
    
    class Meta:
        ordering = ['timestamp']
    
    def __str__(self):
        return f"Progress: {self.samples_labeled} samples, {self.test_accuracy:.2%} accuracy"