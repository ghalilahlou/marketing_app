# marketing_app/models.py
from django.db import models

class Customer(models.Model):
    name = models.CharField(max_length=255, default="Inconnu")
    email = models.EmailField(unique=True)
    last_purchase = models.DateField(null=True, blank=True)
    churn_risk = models.FloatField(default=0.0)
    segment = models.CharField(max_length=50, blank=True)

    def __str__(self):
        return self.name

class Campaign(models.Model):
    title = models.CharField(max_length=255)
    description = models.TextField(blank=True)
    start_date = models.DateField()
    end_date = models.DateField()
    target_segment = models.CharField(max_length=50)
    performance = models.FloatField(default=0.0)

    def __str__(self):
        return self.title

class AdSpending(models.Model):
    campaign = models.ForeignKey(Campaign, on_delete=models.CASCADE)
    daily_budget = models.DecimalField(max_digits=10, decimal_places=2)
    optimized_budget = models.DecimalField(max_digits=10, decimal_places=2, null=True, blank=True)
    date = models.DateField(auto_now_add=True)

    def __str__(self):
        return f"{self.campaign.title} - {self.date}"

class SocialMediaPost(models.Model):
    platform = models.CharField(max_length=50)
    content = models.TextField()
    sentiment = models.CharField(max_length=20, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.platform} - {self.created_at}"


# marketing_app/models.py

class ChatLog(models.Model):
    user_message = models.TextField()
    bot_response = models.TextField()
    timestamp = models.DateTimeField(auto_now_add=True)
    state = models.CharField(max_length=50, default="start")


    def __str__(self):
        return f"Chat at {self.timestamp}"
