from django.contrib import admin
from .models import Customer, Campaign, AdSpending, SocialMediaPost, ChatLog

@admin.register(Customer)
class CustomerAdmin(admin.ModelAdmin):
    list_display = ('name', 'email', 'churn_risk', 'segment')

@admin.register(Campaign)
class CampaignAdmin(admin.ModelAdmin):
    list_display = ('title', 'start_date', 'end_date', 'target_segment', 'performance')

@admin.register(AdSpending)
class AdSpendingAdmin(admin.ModelAdmin):
    list_display = ('campaign', 'daily_budget', 'optimized_budget', 'date')

@admin.register(SocialMediaPost)
class SocialMediaPostAdmin(admin.ModelAdmin):
    list_display = ('platform', 'sentiment', 'created_at')

@admin.register(ChatLog)
class ChatLogAdmin(admin.ModelAdmin):
    list_display = ('timestamp', 'user_message', 'bot_response')
