# model.py

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Expanded dataset for better prediction accuracy
data = {
    "text": [
        "Win money now",
        "Free lottery ticket",
        "Hello friend",
        "Meeting at 5",
        "Claim prize now",
        "Let's talk tomorrow",
        "URGENT: Verify your account immediately",
        "Click here to claim your reward",
        "Limited time offer - Act now",
        "Congratulations you've won",
        "Confirm your banking details",
        "Hi, how are you doing?",
        "Let me know when you're free",
        "Lunch tomorrow at noon?",
        "Can't wait to see you",
        "Nigerian prince needs help",
        "Make $5000 per week",
        "Buy cheap medications",
        "Viagra and cialis online",
        "are you there?",
        "Thanks for the update",
        "Let's schedule a meeting",
        "Good morning!",
        "See you soon"
    ],
    "label": [1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0]
}

df = pd.DataFrame(data)

# Vectorization with better parameters
vectorizer = TfidfVectorizer(lowercase=True, stop_words='english', max_features=500)
X = vectorizer.fit_transform(df["text"])
y = df["label"]

# Model
model = MultinomialNB()
model.fit(X, y)

def predict_message(message):
    """Predict if message is spam and return result with confidence."""
    if not message or len(message.strip()) == 0:
        return {
            "prediction": 0,
            "confidence": 0,
            "result_text": "Empty message"
        }
    
    transformed = vectorizer.transform([message])
    prediction = model.predict(transformed)[0]
    confidence = model.predict_proba(transformed)[0]
    
    # Get confidence percentage for the predicted class
    confidence_score = max(confidence) * 100
    
    return {
        "prediction": int(prediction),
        "confidence": round(confidence_score, 2),
        "result_text": "🚨 SPAM DETECTED" if prediction == 1 else "✅ NOT SPAM"
    }