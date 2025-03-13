# marketing_app/utils/predictive_models.py
import numpy as np
import joblib

def load_churn_model():
    # Essaye de charger un modèle pré-entraîné (il faudra l’entraîner et le sauvegarder)
    try:
        model = joblib.load("marketing_app/models/churn_model.pkl")
    except FileNotFoundError:
        model = None
    return model

def predict_churn(customer_data):
    """
    customer_data : tableau numpy contenant les caractéristiques du client.
    Retourne le risque de churn (entre 0 et 1).
    """
    model = load_churn_model()
    if model:
        prediction = model.predict(customer_data.reshape(1, -1))
        return float(prediction[0])
    else:
        # En l'absence d'un modèle, retourne une valeur aléatoire
        return np.random.rand()
