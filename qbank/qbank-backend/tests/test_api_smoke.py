import pytest
import httpx
BASE = "http://localhost:8000"
def test_health():
    r = httpx.get(f"{BASE}/health"); assert r.status_code==200
def test_login_and_seed():
    r = httpx.post(f"{BASE}/v1/auth/mock-login", json={"user_id":"tester","roles":["author","publisher","student","admin"]})
    assert r.status_code==200; token=r.json()["access_token"]; hdr={"Authorization":f"Bearer {token}"}
    payload={"external_ref":"E2E-1","topic_name":"Cardiology","exam_code":"DEMO-EXAM","stem_md":"Stem","lead_in":"Pick one","rationale_md":"Because","difficulty_label":"medium","options":[{"label":"A","text_md":"Alpha","is_correct":True},{"label":"B","text_md":"Bravo","is_correct":False}]}
    r = httpx.post(f"{BASE}/v1/author/questions", headers=hdr|{"Content-Type":"application/json"}, json=payload); assert r.status_code==200