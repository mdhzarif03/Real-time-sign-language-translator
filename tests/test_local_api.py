from __future__ import annotations

import unittest

from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from backend.app.main import app


def observation(sequence: int, timestamp: int) -> dict:
    return {
        "schema_version": 1,
        "type": "observation",
        "stream_id": "test-stream",
        "sequence": sequence,
        "timestamp_ms": timestamp,
        "sign_language": "en-US-ASL",
        "image_width": 640,
        "image_height": 480,
        "landmarks": {"left_hand": None, "right_hand": None, "pose": None, "face": None},
    }


class LocalApiTests(unittest.TestCase):
    def test_health_reports_model_unavailable_without_fabricating_output(self) -> None:
        with TestClient(app, client=("127.0.0.1", 50000)) as client:
            response = client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["recognition"]["state"], "model_unavailable")

    def test_local_websocket_accepts_valid_landmarks_and_rejects_replayed_sequence(self) -> None:
        with TestClient(app, client=("127.0.0.1", 50000)) as client:
            with client.websocket_connect("/api/v1/stream/test-stream", headers={"origin": "http://localhost:5173"}) as socket:
                initial = socket.receive_json()
                self.assertEqual(initial["state"], "model_unavailable")
                socket.send_json(observation(0, 100))
                # The API reports language-specific checkpoint compatibility
                # after reading the first observation.
                self.assertEqual(socket.receive_json()["state"], "model_unavailable")
                socket.send_json(observation(0, 101))
                with self.assertRaises(WebSocketDisconnect) as closed:
                    socket.receive_json()
                self.assertEqual(closed.exception.code, 1008)


if __name__ == "__main__":
    unittest.main()
