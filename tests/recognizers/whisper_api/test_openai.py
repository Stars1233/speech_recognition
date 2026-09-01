from unittest.mock import MagicMock, patch

import pytest

from speech_recognition import AudioData, Recognizer
from speech_recognition.recognizers.whisper_api import openai

httpx2 = pytest.importorskip("httpx2")
openai_sdk = pytest.importorskip("openai")


@pytest.fixture
def setenv_openai_api_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk_openai_api_key")


def test_transcribe_with_openai_whisper(setenv_openai_api_key):
    # ref: https://github.com/openai/openai-agents-python/blob/v0.22.0/tests/models/test_openai_responses.py#L81
    requests: list[httpx2.Request] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        requests.append(request)
        assert request.method == "POST"
        assert request.url == "https://api.openai.com/v1/audio/transcriptions"
        assert request.headers["Authorization"] == "Bearer sk_openai_api_key"
        assert b'name="model"' in request.content
        assert b"whisper-1" in request.content

        return httpx2.Response(
            200,
            json={"text": "Transcription by OpenAI Whisper"},
            request=request,
        )

    transport = httpx2.MockTransport(handler)

    audio_data = MagicMock(spec=AudioData)
    audio_data.get_wav_data.return_value = b"audio_data"

    with (
        httpx2.Client(transport=transport) as http_client,
        patch("openai.OpenAI", return_value=openai_sdk.OpenAI(http_client=http_client)),
    ):
        actual = openai.recognize(MagicMock(spec=Recognizer), audio_data)

    assert actual == "Transcription by OpenAI Whisper"
    assert len(requests) == 1
    audio_data.get_wav_data.assert_called_once()


def test_transcribe_with_gpt_transcribe(setenv_openai_api_key):
    requests: list[httpx2.Request] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        requests.append(request)
        assert request.method == "POST"
        assert request.url == "https://api.openai.com/v1/audio/transcriptions"
        assert b'name="model"' in request.content
        assert b"gpt-transcribe" in request.content

        return httpx2.Response(
            200,
            json={
                "text": "Transcription by GPT Transcribe",
                "languages": [{"code": "en"}],
            },
            request=request,
        )

    transport = httpx2.MockTransport(handler)

    audio_data = MagicMock(spec=AudioData)
    audio_data.get_wav_data.return_value = b"audio_data"

    with (
        httpx2.Client(transport=transport) as http_client,
        patch("openai.OpenAI", return_value=openai_sdk.OpenAI(http_client=http_client)),
    ):
        actual = openai.recognize(
            MagicMock(spec=Recognizer), audio_data, model="gpt-transcribe"
        )

    assert actual == "Transcription by GPT Transcribe"
    assert len(requests) == 1
    audio_data.get_wav_data.assert_called_once()


def test_transcribe_with_specified_language(setenv_openai_api_key):
    # https://github.com/Uberi/speech_recognition/issues/681
    requests: list[httpx2.Request] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        requests.append(request)
        assert request.method == "POST"
        assert request.url == "https://api.openai.com/v1/audio/transcriptions"
        assert b'name="language"' in request.content
        assert b"\r\n\r\nen\r\n" in request.content

        return httpx2.Response(
            200,
            json={"text": "English transcription"},
            request=request,
        )

    transport = httpx2.MockTransport(handler)

    audio_data = MagicMock(spec=AudioData)
    audio_data.get_wav_data.return_value = b"english_audio"

    with (
        httpx2.Client(transport=transport) as http_client,
        patch("openai.OpenAI", return_value=openai_sdk.OpenAI(http_client=http_client)),
    ):
        actual = openai.recognize(MagicMock(spec=Recognizer), audio_data, language="en")

    assert actual == "English transcription"
    assert len(requests) == 1


def test_transcribe_with_specified_prompt(setenv_openai_api_key):
    requests: list[httpx2.Request] = []

    # https://github.com/Uberi/speech_recognition/pull/676
    def handler(request: httpx2.Request) -> httpx2.Response:
        requests.append(request)
        assert request.method == "POST"
        assert request.url == "https://api.openai.com/v1/audio/transcriptions"
        assert b'name="prompt"' in request.content
        # ref: https://cookbook.openai.com/examples/whisper_prompting_guide
        assert b"Glossary: Aimee, Shawn, BBQ" in request.content

        return httpx2.Response(
            200,
            json={"text": "Prompted transcription"},
            request=request,
        )

    transport = httpx2.MockTransport(handler)

    audio_data = MagicMock(spec=AudioData)
    audio_data.get_wav_data.return_value = b"audio_data"

    with (
        httpx2.Client(transport=transport) as http_client,
        patch("openai.OpenAI", return_value=openai_sdk.OpenAI(http_client=http_client)),
    ):
        actual = openai.recognize(
            MagicMock(spec=Recognizer),
            audio_data,
            prompt="Glossary: Aimee, Shawn, BBQ",
        )

    assert actual == "Prompted transcription"
    assert len(requests) == 1
