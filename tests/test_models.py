"""The vLLM adapter against a fake OpenAI-compatible server. No GPU, no network."""

import httpx
import pytest

from tiny_api_agent.models import load_model
from tiny_api_agent.models.vllm_model import VLLMModel


def fake_server(reply: str, status: int = 200, seen: list | None = None) -> httpx.Client:
    """An httpx client whose transport answers like vLLM's /v1/chat/completions."""

    def handler(request: httpx.Request) -> httpx.Response:
        if seen is not None:
            seen.append(request)
        if status != 200:
            return httpx.Response(status, json={"error": "boom"})
        return httpx.Response(200, json={"choices": [{"message": {"role": "assistant", "content": reply}}]})

    return httpx.Client(transport=httpx.MockTransport(handler), base_url="http://fake")


def test_complete_sends_openai_shape_and_returns_text():
    seen: list[httpx.Request] = []
    m = VLLMModel("Qwen/Qwen2.5-3B-Instruct", client=fake_server("<final>ok</final>", seen=seen))

    out = m.complete("be brief", [{"role": "user", "content": "hi"}])

    assert out == "<final>ok</final>"
    req = seen[0]
    assert req.url.path == "/v1/chat/completions"
    body = httpx.Response(200, content=req.content).json()
    assert body["model"] == "Qwen/Qwen2.5-3B-Instruct"
    assert body["messages"][0] == {"role": "system", "content": "be brief"}
    assert body["messages"][1] == {"role": "user", "content": "hi"}
    assert body["temperature"] == 0.0
    assert body["max_tokens"] == 1024


def test_server_error_raises_so_eval_records_an_error_row():
    m = VLLMModel("x", client=fake_server("", status=500))
    with pytest.raises(httpx.HTTPStatusError):
        m.complete("s", [{"role": "user", "content": "hi"}])


def test_null_content_becomes_empty_string():
    # vLLM can return content: null on an empty generation; the parser wants a str.
    m = VLLMModel("x", client=fake_server(None))
    assert m.complete("s", [{"role": "user", "content": "hi"}]) == ""


def test_load_model_registers_vllm_and_keeps_slashes_in_the_id():
    m = load_model("vllm:Qwen/Qwen2.5-3B-Instruct")
    assert isinstance(m, VLLMModel)
    assert m.name == "Qwen/Qwen2.5-3B-Instruct"
    with pytest.raises(ValueError, match="needs a model id"):
        load_model("vllm")
