import os
os.environ["API_KEY"] = "test_key"
from fastapi.testclient import TestClient
import sys
import os

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.platform.api import main
main.API_KEY = "test_key"
from src.platform.api.main import app

client = TestClient(app)

def test_get_hazards():
    response = client.get("/api/safety/hazards/types", headers={"X-API-Key": "test_key"})
    assert response.status_code == 200
    data = response.json()
    assert len(data) > 0
    assert any(h["type_key"] == "OVERHANG" for h in data)
    print("GET /hazards/types passed")

def test_create_hazard():
    import random
    rand_id = random.randint(1000, 9999)
    type_key = f"TEST_HAZARD_{rand_id}"
    
    payload = {
        "type_key": type_key,
        "display_name": "Test Hazard",
        "description": "A test hazard",
        "default_severity": 0.5,
        "default_behaviour": {"action": "WARN"}
    }
    response = client.post("/api/safety/hazards/types", json=payload, headers={"X-API-Key": "test_key"})
    if response.status_code != 200:
        print(f"POST failed: {response.text}")
    assert response.status_code == 200
    
    # Verify it's in the list
    response = client.get("/api/safety/hazards/types", headers={"X-API-Key": "test_key"})
    data = response.json()
    assert any(h["type_key"] == type_key for h in data)
    print("POST /hazards/types passed")

if __name__ == "__main__":
    test_get_hazards()
    test_create_hazard()
    print("All API Tests Passed")
