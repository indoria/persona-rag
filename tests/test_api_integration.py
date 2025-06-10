import pytest
from flask import Flask
import json

@pytest.fixture
def client():
    from app.api import app
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_list_journalists(client):
    resp = client.get('/journalists')
    assert resp.status_code == 200
    data = resp.get_json()
    assert isinstance(data, list)
    assert any("Alice" in j["name"] or "Bob" in j["name"] for j in data)

def test_generate_response_api(client):
    # First, get a journalist ID
    resp = client.get('/journalists')
    data = resp.get_json()
    jid = data[0]["id"]
    payload = {"journalist_id": jid, "pitch_text": "What are your thoughts on renewable energy?"}
    resp2 = client.post('/generate_response', data=json.dumps(payload), content_type='application/json')
    assert resp2.status_code == 200
    out = resp2.get_json()
    assert "response" in out
    assert isinstance(out["response"], str)