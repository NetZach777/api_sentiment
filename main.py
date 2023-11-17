import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import RobertaTokenizer, TFRobertaForSequenceClassification
import tensorflow as tf

app = FastAPI()

# Obtenez le chemin du répertoire actuel
#current_directory = os.path.dirname(os.path.realpath(__file__))

# Utilisez le chemin relatif pour accéder au modèle Roberta dans le répertoire du projet
#MODEL_PATH = os.path.join(current_directory, "model", "roberta_model_V3")

# tokenizer Hugging face
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
# Modèle entraîné
model = TFRobertaForSequenceClassification.from_pretrained('models')


class TweetRequest(BaseModel):
    text: str

@app.post("/predict-sentiment/")
def predict_sentiment(tweet: TweetRequest):
    # Vérifie si le texte est vide ou ne contient que des espaces blancs
    if not tweet.text.strip():
        raise HTTPException(status_code=422, detail="Text cannot be empty or just whitespace.")

    inputs = tokenizer(tweet.text, return_tensors="tf", max_length=512, truncation=True, padding="max_length")
    outputs = model(inputs['input_ids'], attention_mask=inputs['attention_mask'])
    prediction = tf.argmax(outputs.logits, -1).numpy()[0]
    # Assurez-vous que le modèle renvoie des prédictions binaires (0 ou 1)
    if prediction not in [0, 1]:
        raise HTTPException(status_code=500, detail="Model prediction was not 0 or 1.")
    sentiment = "positive" if prediction == 1 else "negative"
    return {"sentiment": sentiment}

