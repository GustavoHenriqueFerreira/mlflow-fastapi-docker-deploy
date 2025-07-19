# test_main.py
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_predict():
    response = client.post("/predict", json={
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    })
    assert response.status_code == 200
    json_resp = response.json()
    assert "predicted_class" in json_resp
    assert isinstance(json_resp["predicted_class"], int)
