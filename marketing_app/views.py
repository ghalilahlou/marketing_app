from django.shortcuts import render
from django.http import JsonResponse
from django.db.models import Sum
from marketing_app.utils.chatbot import get_bot_response
from .models import Customer, Campaign, SocialMediaPost, ChatLog, AdSpending

def dashboard(request):
    total_customers = Customer.objects.count()
    total_campaigns = Campaign.objects.count()
    recent_posts = SocialMediaPost.objects.order_by('-created_at')[:5]
    context = {
        "total_customers": total_customers,
        "total_campaigns": total_campaigns,
        "recent_posts": recent_posts,
    }
    return render(request, "marketing_app/dashboard.html", context)

def chatbot_view(request):
    response = ""
    if request.method == "POST":
        user_input = request.POST.get("message", "")
        response = get_bot_response(user_input)
        # Enregistrer le chat dans la base
        ChatLog.objects.create(user_message=user_input, bot_response=response)
    return render(request, "marketing_app/chatbot.html", {"response": response})

def get_kpi(request):
    """
    Vue pour renvoyer les KPI marketing sous forme de JSON.
    Ces données peuvent être récupérées via AJAX dans le front-end.
    """
    total_customers = Customer.objects.count()
    total_campaigns = Campaign.objects.count()
    total_spending = AdSpending.objects.aggregate(total=Sum('daily_budget'))['total'] or 0

    data = {
        "kpi_value": total_customers,
        "total_campaigns": total_campaigns,
        "total_spending": total_spending,
    }
    return JsonResponse(data)
