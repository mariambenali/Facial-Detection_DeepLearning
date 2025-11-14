import io
from PIL import Image
from app.main import app 
from fastapi.testclient import TestClient
from model_DL.detect_and_predict import detect_and_predict_emotion
from tensorflow.keras.models import load_model
import os 




client = TestClient(app)

def test_load():
    model = load_model("/Users/miriambenali/Desktop/Project-Simplon/Facial-Detection_DeepLearning/model_DL/facial_model.keras")
    assert model is not None
    assert hasattr(model, "predict")

    
    
def test_get_emotions():
    response = client.get("/emotions")
    assert response.status_code == 200
    assert isinstance(response.json(), list)

    data = response.json()
    assert isinstance(data, list)

    if data:
        dict = data[0]
        for key in ["id", "predicted_emotion", "score", "date_of_creation"]:
            assert key in dict


def test_env():
    assert os.getenv("DATABASE_URL")







