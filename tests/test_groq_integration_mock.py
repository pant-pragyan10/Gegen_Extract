import time
from gegenextract.extraction.groq_client import GroqClient
from gegenextract.persistence import PersistenceManager
import json


def test_groq_client_mock(tmp_path, monkeypatch):
    db = PersistenceManager(str(tmp_path / "db.sqlite"))
    # mock requests.Session.post to return a fake response
    class FakeResp:
        def __init__(self):
            self.status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            return {"choices": [{"text": '{"name": "Alice"}'}], "usage": {"total_tokens": 10}}

    class FakeSession:
        def post(self, *args, **kwargs):
            return FakeResp()

    client = GroqClient(api_key="key", persistence=db)
    monkeypatch.setattr(client, "session", FakeSession())
    res = client.call("prompt", temperature=0.0)
    assert res["text"] == '{"name": "Alice"}'
    assert res["tokens"] == 10
