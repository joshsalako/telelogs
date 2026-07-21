from __future__ import annotations

import json
import unittest
from urllib.request import Request

from qwen_vllm.checks import CheckError, check_server
from qwen_vllm.config import DeploymentSettings


class FakeResponse:
    def __init__(self, payload: object = None, status: int = 200) -> None:
        self.status = status
        self.body = b"" if payload is None else json.dumps(payload).encode()

    def __enter__(self) -> FakeResponse:
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def getcode(self) -> int:
        return self.status

    def read(self) -> bytes:
        return self.body


class QueueOpener:
    def __init__(self, *responses: FakeResponse) -> None:
        self.responses = list(responses)
        self.requests: list[Request] = []

    def __call__(self, request: Request, *, timeout: float) -> FakeResponse:
        self.requests.append(request)
        return self.responses.pop(0)


class CheckTests(unittest.TestCase):
    def test_successful_checks_and_auth(self) -> None:
        opener = QueueOpener(
            FakeResponse(),
            FakeResponse({"data": [{"id": "Qwen3.6-27B"}]}),
            FakeResponse({"choices": [{"message": {"content": "OK"}}]}),
        )
        settings = DeploymentSettings(api_key="secret")
        self.assertEqual(
            check_server(settings, opener=opener),
            ["health: ok", "model: Qwen3.6-27B", "chat: ok"],
        )
        self.assertEqual(
            [request.full_url for request in opener.requests],
            [
                "http://127.0.0.1:8000/health",
                "http://127.0.0.1:8000/v1/models",
                "http://127.0.0.1:8000/v1/chat/completions",
            ],
        )
        self.assertTrue(
            all(
                request.get_header("Authorization") == "Bearer secret"
                for request in opener.requests
            )
        )
        self.assertEqual(opener.requests[-1].method, "POST")
        payload = json.loads(opener.requests[-1].data or b"")
        self.assertEqual(payload["model"], "Qwen3.6-27B")

    def test_missing_model_fails(self) -> None:
        opener = QueueOpener(FakeResponse(), FakeResponse({"data": []}))
        with self.assertRaisesRegex(CheckError, "does not advertise"):
            check_server(DeploymentSettings(), opener=opener)

    def test_malformed_chat_fails(self) -> None:
        opener = QueueOpener(
            FakeResponse(),
            FakeResponse({"data": [{"id": "Qwen3.6-27B"}]}),
            FakeResponse({"choices": []}),
        )
        with self.assertRaisesRegex(CheckError, r"choices\[0\]"):
            check_server(DeploymentSettings(), opener=opener)


if __name__ == "__main__":
    unittest.main()
