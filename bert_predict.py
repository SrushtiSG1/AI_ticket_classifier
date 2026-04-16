from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

tokenizer = AutoTokenizer.from_pretrained("model/bert_model")
model = AutoModelForSequenceClassification.from_pretrained("model/bert_model")

def predict_bert(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)

    predicted_class = torch.argmax(outputs.logits).item()
    return predicted_class