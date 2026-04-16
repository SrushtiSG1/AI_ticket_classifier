import joblib

# Load models
model_category = joblib.load("model/category_model.pkl")
model_priority = joblib.load("model/priority_model.pkl")
vectorizer = joblib.load("model/vectorizer.pkl")

def predict_ticket(text):
    # Transform input
    text_vec = vectorizer.transform([text])
    
    # Predictions
    category = model_category.predict(text_vec)[0]
    priority = model_priority.predict(text_vec)[0]
    
    return category, priority