# marketing_app/utils/chatbot.py
"""
import os
import pandas as pd
from django.conf import settings
from transformers import pipeline, set_seed
from django.db.models import Sum
from marketing_app.models import Customer, Campaign, AdSpending, SocialMediaPost

# -----------------------------------------------------------------------------
# Configuration des modèles NLP
# -----------------------------------------------------------------------------
# Modèle général pour des réponses polyvalentes
MODEL_GENERAL = "EleutherAI/gpt-neo-1.3B"
# Modèle spécialisé pour du contenu marketing en français
MODEL_MARKETING = "asi/gpt-fr-cased-base"

# Paramètres communs pour la génération de texte
GENERATION_KWARGS = {
    "max_length": 120,
    "num_return_sequences": 1,
    "do_sample": True,
    "temperature": 0.7,
    "top_p": 0.9,
    "truncation": True,
}

# Pipelines (initialement non chargés)
nlp_pipeline_general = None
nlp_pipeline_marketing = None

def initialize_models() -> None:
    Initialise les pipelines pour la génération de texte.
    global nlp_pipeline_general, nlp_pipeline_marketing
    if nlp_pipeline_general is None:
        nlp_pipeline_general = pipeline(
            "text-generation",
            model=MODEL_GENERAL,
            **GENERATION_KWARGS
        )
        set_seed(42)
    if nlp_pipeline_marketing is None:
        nlp_pipeline_marketing = pipeline(
            "text-generation",
            model=MODEL_MARKETING,
            **GENERATION_KWARGS
        )
        set_seed(42)

def generate_text_with_pipeline(pipeline_obj, prompt: str) -> str:
    Génère du texte en utilisant le pipeline spécifié.
    result = pipeline_obj(prompt)
    return result[0]["generated_text"]

def ensemble_generate_text(prompt: str) -> str:
    Génère du texte en combinant les réponses du modèle général et du modèle marketing.
    La logique ici sélectionne la réponse ayant le plus de contenu.
    initialize_models()
    text_general = generate_text_with_pipeline(nlp_pipeline_general, prompt)
    text_marketing = generate_text_with_pipeline(nlp_pipeline_marketing, prompt)
    
    # Nettoyage des réponses en retirant le prompt de base
    response_general = text_general.replace(prompt, "").strip()
    response_marketing = text_marketing.replace(prompt, "").strip()
    
    # Retourne la réponse la plus longue, ou la réponse marketing si elles sont de taille égale.
    if len(response_marketing) >= len(response_general):
        return response_marketing
    return response_general

# -----------------------------------------------------------------------------
# Fonctions utilitaires pour cas d'utilisation marketing
# -----------------------------------------------------------------------------

def get_kpi_info() -> str:
    Récupère et retourne des KPIs issus de la base de données.
    
    total_customers = Customer.objects.count()
    total_campaigns = Campaign.objects.count()
    total_spending = AdSpending.objects.aggregate(total=Sum('daily_budget'))['total'] or 0
    return (f"{total_customers} clients, {total_campaigns} campagnes, "
            f"Budget publicitaire total: {total_spending:.2f}")

def analyze_csv() -> str:
    Analyse le fichier CSV 'marketing_data.csv' placé dans le dossier media.
    Calcule par exemple la moyenne de la colonne 'roi'.
    
    csv_path = os.path.join(settings.BASE_DIR, "media", "marketing_data.csv")
    if not os.path.exists(csv_path):
        return "Fichier CSV introuvable dans le dossier media."
    
    df = pd.read_csv(csv_path)
    if 'roi' not in df.columns:
        return "La colonne 'roi' est absente du CSV."
    mean_roi = df['roi'].mean()
    return f"ROI moyen: {mean_roi:.2f}"

def get_ad_spending_info() -> str:
    Calcule le total des budgets publicitaires enregistrés dans la base.
    daily_budgets = AdSpending.objects.values_list('daily_budget', flat=True)
    total_budget = sum(daily_budgets) if daily_budgets else 0
    return f"Dépenses publicitaires totales: {total_budget:.2f}"

def get_campaign_report() -> str:
    Génère un rapport succinct de toutes les campagnes marketing.
    campaigns = Campaign.objects.all()
    if not campaigns:
        return "Aucune campagne trouvée."
    lines = [f"{camp.title}: du {camp.start_date} au {camp.end_date}" for camp in campaigns]
    return "Campagnes:\n" + "\n".join(lines)

def generate_marketing_content(user_input: str) -> str:
    Utilise le modèle spécialisé marketing pour générer un contenu créatif.
    prompt = f"Generate creative marketing content in French based on: {user_input}\nContent:"
    initialize_models()
    generated = generate_text_with_pipeline(nlp_pipeline_marketing, prompt)
    return generated.replace(prompt, "").strip()

def get_general_response(user_input: str) -> str:
  
    Génère une réponse générale en combinant les réponses des deux pipelines.
    
    prompt = f"You are a helpful marketing assistant in French. Question: {user_input}\nAnswer:"
    return ensemble_generate_text(prompt).replace(prompt, "").strip()

# -----------------------------------------------------------------------------
# Détection d'intentions
# -----------------------------------------------------------------------------

def detect_intent(user_input: str) -> str:
    
    Analyse le texte utilisateur pour déterminer l'intention.
    Possibilités : 
      - 'kpi'
      - 'csv_analysis'
      - 'ad_spending'
      - 'campaign_report'
      - 'marketing_content'
      - 'general_nlp'
    
    text = user_input.lower()
    if any(word in text for word in ["kpi", "statistique", "performance"]):
        return "kpi"
    if any(word in text for word in ["csv", "fichier", "document", "report"]):
        return "csv_analysis"
    if any(word in text for word in ["budget", "spending", "dépense"]):
        return "ad_spending"
    if any(word in text for word in ["campaign", "campagne", "create campaign", "launch campaign"]):
        return "campaign_report"
    if any(word in text for word in ["content", "slogan", "copy", "texte", "accroche"]):
        return "marketing_content"
    return "general_nlp"

# -----------------------------------------------------------------------------
# Fonction principale de réponse du chatbot
# -----------------------------------------------------------------------------

def get_bot_response(user_input: str) -> str:
    
    Renvoie la réponse du chatbot en fonction de l'intention détectée.
    Selon l'intention, une fonction dédiée est appelée ou une réponse générale
    est générée via un ensemble de modèles.
  
    intent = detect_intent(user_input)
    if intent == "kpi":
        return "KPIs: " + get_kpi_info()
    elif intent == "csv_analysis":
        return "Analyse CSV: " + analyze_csv()
    elif intent == "ad_spending":
        return "Dépenses publicitaires: " + get_ad_spending_info()
    elif intent == "campaign_report":
        return "Rapport de campagnes: " + get_campaign_report()
    elif intent == "marketing_content":
        return generate_marketing_content(user_input)
    else:
        return get_general_response(user_input)
"""

# marketing_app/utils/chatbot.py

import os
import logging
import pandas as pd
from django.conf import settings
from transformers import pipeline, set_seed
from django.db.models import Sum
from marketing_app.models import Customer, Campaign, AdSpending, SocialMediaPost

# Configuration du logging pour le débogage
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Configuration des modèles NLP
# -----------------------------------------------------------------------------
# Modèle général pour des réponses polyvalentes (en anglais, mais adapté aux besoins)
MODEL_GENERAL = "EleutherAI/gpt-neo-1.3B"
# Modèle spécialisé pour du contenu marketing en français
MODEL_MARKETING = "asi/gpt-fr-cased-base"

# Paramètres communs pour la génération de texte
GENERATION_KWARGS = {
    "max_length": 200,
    "num_return_sequences": 1,
    "do_sample": True,
    "temperature": 0.7,
    "top_p": 0.9,
    "truncation": True,
}

# Pipelines (initialement non chargés)
nlp_pipeline_general = None
nlp_pipeline_marketing = None

def initialize_models() -> None:
    """
    Initialise les pipelines pour la génération de texte.
    Cette fonction tente de charger les deux modèles. En cas d'erreur,
    un message de log est enregistré.
    """
    global nlp_pipeline_general, nlp_pipeline_marketing
    try:
        if nlp_pipeline_general is None:
            logger.info("Chargement du modèle général : %s", MODEL_GENERAL)
            nlp_pipeline_general = pipeline(
                "text-generation",
                model=MODEL_GENERAL,
                **GENERATION_KWARGS
            )
            set_seed(42)
    except Exception as e:
        logger.error("Erreur lors du chargement du modèle général: %s", e)
        nlp_pipeline_general = None

    try:
        if nlp_pipeline_marketing is None:
            logger.info("Chargement du modèle marketing : %s", MODEL_MARKETING)
            nlp_pipeline_marketing = pipeline(
                "text-generation",
                model=MODEL_MARKETING,
                **GENERATION_KWARGS
            )
            set_seed(42)
    except Exception as e:
        logger.error("Erreur lors du chargement du modèle marketing: %s", e)
        nlp_pipeline_marketing = None

def generate_text_with_pipeline(pipeline_obj, prompt: str) -> str:
    """
    Génère du texte en utilisant le pipeline spécifié.
    En cas d'erreur, retourne un message d'erreur.
    """
    try:
        result = pipeline_obj(prompt)
        return result[0]["generated_text"]
    except Exception as e:
        logger.error("Erreur lors de la génération de texte avec le prompt '%s': %s", prompt, e)
        return "Désolé, une erreur est survenue lors de la génération de la réponse."

def ensemble_generate_text(prompt: str) -> str:
    """
    Génère du texte en combinant les réponses du modèle général et du modèle marketing.
    La logique ici retourne la réponse ayant le plus de contenu.
    """
    initialize_models()
    response_general = ""
    response_marketing = ""
    if nlp_pipeline_general:
        response_general = generate_text_with_pipeline(nlp_pipeline_general, prompt).replace(prompt, "").strip()
    if nlp_pipeline_marketing:
        response_marketing = generate_text_with_pipeline(nlp_pipeline_marketing, prompt).replace(prompt, "").strip()

    # Choisir la réponse la plus complète (ou marketing si de taille égale)
    if len(response_marketing) >= len(response_general):
        return response_marketing
    return response_general

# -----------------------------------------------------------------------------
# Fonctions utilitaires pour divers cas d'utilisation marketing
# -----------------------------------------------------------------------------

def get_kpi_info() -> str:
    """
    Récupère et retourne des KPIs issus de la base de données.
    """
    total_customers = Customer.objects.count()
    total_campaigns = Campaign.objects.count()
    total_spending = AdSpending.objects.aggregate(total=Sum('daily_budget'))['total'] or 0
    return (f"{total_customers} clients, {total_campaigns} campagnes, "
            f"budget publicitaire total: {total_spending:.2f}")

def analyze_csv() -> str:
    """
    Analyse le fichier CSV 'marketing_data.csv' placé dans le dossier media.
    Calcule par exemple la moyenne de la colonne 'roi'.
    """
    csv_path = os.path.join(settings.BASE_DIR, "media", "marketing_data.csv")
    if not os.path.exists(csv_path):
        return "Fichier CSV introuvable dans le dossier media."
    
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        logger.error("Erreur de lecture du CSV: %s", e)
        return "Erreur lors de la lecture du fichier CSV."
    
    if 'roi' not in df.columns:
        return "La colonne 'roi' est absente du CSV."
    mean_roi = df['roi'].mean()
    return f"ROI moyen: {mean_roi:.2f}"

def get_ad_spending_info() -> str:
    """
    Calcule le total des budgets publicitaires enregistrés dans la base.
    """
    daily_budgets = AdSpending.objects.values_list('daily_budget', flat=True)
    total_budget = sum(daily_budgets) if daily_budgets else 0
    return f"Dépenses publicitaires totales: {total_budget:.2f}"

def get_campaign_report() -> str:
    """
    Génère un rapport succinct de toutes les campagnes marketing.
    """
    campaigns = Campaign.objects.all()
    if not campaigns:
        return "Aucune campagne trouvée."
    lines = [f"{camp.title}: {camp.start_date} - {camp.end_date}" for camp in campaigns]
    return "Campagnes:\n" + "\n".join(lines)

def generate_marketing_content(user_input: str) -> str:
    """
    Utilise le modèle marketing pour générer un contenu marketing créatif en français.
    """
    prompt = f"Génère un contenu marketing créatif en français pour : {user_input}\nContenu:"
    initialize_models()
    if nlp_pipeline_marketing:
        generated = generate_text_with_pipeline(nlp_pipeline_marketing, prompt)
        return generated.replace(prompt, "").strip()
    return "Le modèle marketing n'est pas disponible."

def get_general_response(user_input: str) -> str:
    """
    Génère une réponse générale via l'ensemble des modèles.
    """
    prompt = f"You are a helpful marketing assistant in French. Question: {user_input}\nAnswer:"
    return ensemble_generate_text(prompt).replace(prompt, "").strip()

# -----------------------------------------------------------------------------
# Détection d'intentions
# -----------------------------------------------------------------------------

def detect_intent(user_input: str) -> str:
    """
    Analyse le texte utilisateur pour déterminer l'intention.
    Possibilités :
      - 'kpi' pour des statistiques marketing,
      - 'csv_analysis' pour l'analyse de fichiers CSV,
      - 'ad_spending' pour des questions de budget,
      - 'campaign_report' pour des rapports sur les campagnes,
      - 'marketing_content' pour la génération de contenu marketing,
      - 'general_nlp' pour toute autre question.
    """
    text = user_input.lower()
    if any(word in text for word in ["kpi", "statistique", "performance"]):
        return "kpi"
    if any(word in text for word in ["csv", "fichier", "document", "report"]):
        return "csv_analysis"
    if any(word in text for word in ["budget", "spending", "dépense"]):
        return "ad_spending"
    if any(word in text for word in ["campaign", "campagne"]):
        return "campaign_report"
    if any(word in text for word in ["content", "slogan", "copy", "texte", "accroche"]):
        return "marketing_content"
    return "general_nlp"

# -----------------------------------------------------------------------------
# Fonction principale de réponse du chatbot
# -----------------------------------------------------------------------------

def get_bot_response(user_input: str) -> str:
    """
    Renvoie la réponse du chatbot en fonction de l'intention détectée.
    Selon l'intention, une fonction dédiée est appelée.
    """
    intent = detect_intent(user_input)
    logger.info("Intention détectée: %s", intent)
    
    if intent == "kpi":
        response = "KPIs: " + get_kpi_info()
    elif intent == "csv_analysis":
        response = "Analyse CSV: " + analyze_csv()
    elif intent == "ad_spending":
        response = "Dépenses publicitaires: " + get_ad_spending_info()
    elif intent == "campaign_report":
        response = "Rapport de campagnes: " + get_campaign_report()
    elif intent == "marketing_content":
        response = generate_marketing_content(user_input)
    else:
        response = get_general_response(user_input)
    
    # Vérification et log de la réponse générée
    if not response:
        logger.warning("Aucune réponse générée pour l'entrée: %s", user_input)
        return "Désolé, je n'ai pas pu générer de réponse. Veuillez réessayer."
    
    logger.info("Réponse générée: %s", response)
    return response
