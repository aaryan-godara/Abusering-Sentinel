"""Tests for Phase 6 — FastAPI API.

Tests cover:
- /health
- valid /risk request
- valid /investigation request
- unknown user -> 404
- batch scoring
- batch investigation
- review queue
- ground-truth fields are absent
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
import numpy as np

from src.api.main import app
from src.api.dependencies import load_resources, get_investigator
from src.investigation.schema import GROUND_TRUTH_FIELDS


# Need to load the resources to test
@pytest.fixture(scope="session", autouse=True)
def setup_api_resources():
    load_resources()


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def valid_user_id():
    """Get a valid user ID from the loaded investigator."""
    inv = get_investigator()
    return inv.user_ids[0]


class TestHealth:
    def test_health_check(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["model_loaded"] is True
        assert data["investigator_loaded"] is True


class TestInvestigationEndpoints:
    def test_get_user_investigation_valid(self, client, valid_user_id):
        response = client.get(f"/users/{valid_user_id}/investigation")
        assert response.status_code == 200
        data = response.json()
        assert data["user_id"] == valid_user_id
        assert "risk_score" in data
        assert "graph_evidence" in data

    def test_get_user_investigation_unknown(self, client):
        response = client.get("/users/UNKNOWN_9999/investigation")
        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()
        
    def test_get_user_evidence(self, client, valid_user_id):
        response = client.get(f"/users/{valid_user_id}/evidence")
        assert response.status_code == 200
        data = response.json()
        assert "shared_devices" in data
        assert "shared_ips" in data


class TestRiskEndpoints:
    def test_get_user_risk_valid(self, client, valid_user_id):
        response = client.get(f"/users/{valid_user_id}/risk")
        assert response.status_code == 200
        data = response.json()
        assert data["user_id"] == valid_user_id
        assert "decision" in data
        assert "review_priority" in data
        assert "reasons" in data


class TestQueueEndpoints:
    def test_review_queue(self, client, valid_user_id):
        # Trigger an evaluation that might add to queue depending on score
        # Even if empty, it should return 200
        client.get(f"/users/{valid_user_id}/risk")
        
        response = client.get("/review-queue")
        assert response.status_code == 200
        assert isinstance(response.json(), list)

    def test_review_queue_top_n(self, client):
        response = client.get("/review-queue/top/5")
        assert response.status_code == 200
        assert isinstance(response.json(), list)
        
    def test_review_queue_top_invalid_n(self, client):
        response = client.get("/review-queue/top/-1")
        assert response.status_code == 400


class TestBatchEndpoints:
    def test_batch_score(self, client, valid_user_id):
        response = client.post("/score", json={"user_ids": [valid_user_id, "UNKNOWN"]})
        assert response.status_code == 200
        data = response.json()
        # Should only return the valid one
        assert len(data) == 1
        assert data[0]["user_id"] == valid_user_id

    def test_batch_investigate(self, client, valid_user_id):
        response = client.post("/investigate", json={"user_ids": [valid_user_id, "UNKNOWN"]})
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        assert data[0]["user_id"] == valid_user_id

    def test_batch_empty_invalid(self, client):
        response = client.post("/score", json={"user_ids": []})
        assert response.status_code == 422


class TestGroundTruthFieldsAbsent:
    def _check_dict_for_labels(self, d: dict):
        for key in d:
            assert key not in GROUND_TRUTH_FIELDS, f"Found ground truth field {key}"
        for val in d.values():
            if isinstance(val, dict):
                self._check_dict_for_labels(val)
            elif isinstance(val, list):
                for item in val:
                    if isinstance(item, dict):
                        self._check_dict_for_labels(item)

    def test_no_labels_in_investigation(self, client, valid_user_id):
        response = client.get(f"/users/{valid_user_id}/investigation")
        data = response.json()
        self._check_dict_for_labels(data)
        
    def test_no_labels_in_risk(self, client, valid_user_id):
        response = client.get(f"/users/{valid_user_id}/risk")
        data = response.json()
        self._check_dict_for_labels(data)

