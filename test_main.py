from fastapi.testclient import TestClient
from main import app

client = TestClient(app)
API_ENDPOINT = "/predict-sentiment/"

# Teste la prédiction positive pour un texte positif
def test_predict_sentiment_positive():
    response = client.post(API_ENDPOINT, json={"text": "I love FastAPI!"})
    assert response.status_code == 200
    assert response.json() == {"sentiment": "positive"}

# Teste la prédiction négative pour un texte négatif
def test_predict_sentiment_negative():
    response = client.post(API_ENDPOINT, json={"text": "I hate Mondays!"})
    assert response.status_code == 200
    assert response.json() == {"sentiment": "negative"}

# Teste la gestion d'une requête sans corps
def test_predict_sentiment_no_body():
    response = client.post(API_ENDPOINT)
    assert response.status_code == 422

# Teste la gestion d'un texte vide
def test_predict_sentiment_empty_text():
    response = client.post(API_ENDPOINT, json={"text": ""})
    assert response.status_code == 422

# Teste la gestion de caractères non-ASCII
def test_predict_sentiment_non_ascii():
    response = client.post(API_ENDPOINT, json={"text": "J'adore FastAPI et les crêpes !"})
    assert response.status_code == 200
    assert "sentiment" in response.json()

# Teste la gestion d'un type de contenu incorrect
def test_predict_sentiment_wrong_content_type():
    response = client.post(API_ENDPOINT, content="I love FastAPI!", headers={"Content-Type": "text/plain"})
    assert response.status_code == 422

# Teste que les méthodes GET ne sont pas autorisées
def test_predict_sentiment_get_method_not_allowed():
    response = client.get(API_ENDPOINT)
    assert response.status_code == 405