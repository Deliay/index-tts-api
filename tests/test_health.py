from fastapi.testclient import TestClient

from index_tts_api import __version__
from index_tts_api.main import app


def test_health_reports_service_and_model_state() -> None:
    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "service": "IndexTTS API",
        "version": __version__,
        "model": {
            "name": "IndexTTS-2.5",
            "state": "not_loaded",
            "version": None,
            "error": None,
        },
    }
