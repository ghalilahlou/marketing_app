import numpy as np
import joblib

def load_churn_model():
    try:
        model = joblib.load("marketing_app/models/churn_model.pkl")
    except FileNotFoundError:
        model = None
    return model

def predict_churn(customer_data):
    model = load_churn_model()
    if model:
        prediction = model.predict(customer_data.reshape(1, -1))
        return float(prediction[0])
    else:
        return np.random.rand()
