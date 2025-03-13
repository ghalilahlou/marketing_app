from textblob import TextBlob

def analyze_sentiment(text):
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    if polarity > 0.1:
        return "Positif"
    elif polarity < -0.1:
        return "Négatif"
    else:
        return "Neutre"
