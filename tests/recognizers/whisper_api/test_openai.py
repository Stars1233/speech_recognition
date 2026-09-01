from unittest.mock import MagicMock, patch

import pytest
import respx

from speech_recognition import AudioData, Recognizer
from speech_recognition.recognizers.whisper_api import openai

httpx2 = pytest.importorskip("httpx2")
openai_sdk = pytest.importorskip("openai")


@pytest.fixture
def setenv_openai_api_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk_openai_api_key")


def test_transcribe_with_openai_whisper(setenv_openai_api_key):
    def handler(request: httpx2.Request) -> httpx2.Response:
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
        openai_sdk.DefaultHttpx2Client(transport=transport) as http_client,
        patch("openai.OpenAI", return_value=openai_sdk.OpenAI(http_client=http_client)),
    ):
        actual = openai.recognize(MagicMock(spec=Recognizer), audio_data)

    assert actual == "Transcription by OpenAI Whisper"
    audio_data.get_wav_data.assert_called_once()


@respx.mock(assert_all_called=True, assert_all_mocked=True)
def test_transcribe_with_gpt_transcribe(respx_mock, setenv_openai_api_key):
    respx_mock.post(
        "https://api.openai.com/v1/audio/transcriptions",
        data__contains={"model": "gpt-transcribe"},
    ).respond(
        200,
        json={
            "text": "Transcription by GPT Transcribe",
            "languages": [{"code": "en"}],
        },
    )

    audio_data = MagicMock(spec=AudioData)
    audio_data.get_wav_data.return_value = b"audio_data"

    actual = openai.recognize(
        MagicMock(spec=Recognizer), audio_data, model="gpt-transcribe"
    )

    assert actual == "Transcription by GPT Transcribe"
    audio_data.get_wav_data.assert_called_once()


@respx.mock(assert_all_called=True, assert_all_mocked=True)
def test_transcribe_with_specified_language(respx_mock, setenv_openai_api_key):
    # https://github.com/Uberi/speech_recognition/issues/681
    respx_mock.post(
        "https://api.openai.com/v1/audio/transcriptions",
        data__contains={"language": "en"},
    ).respond(
        200,
        json={"text": "English transcription"},
    )

    audio_data = MagicMock(spec=AudioData)
    audio_data.get_wav_data.return_value = b"english_audio"

    actual = openai.recognize(
        MagicMock(spec=Recognizer), audio_data, language="en"
    )

    assert actual == "English transcription"


@respx.mock(assert_all_called=True, assert_all_mocked=True)
def test_transcribe_with_specified_prompt(respx_mock, setenv_openai_api_key):
    # https://github.com/Uberi/speech_recognition/pull/676
    respx_mock.post(
        "https://api.openai.com/v1/audio/transcriptions",
        # ref: https://cookbook.openai.com/examples/whisper_prompting_guide
        data__contains={"prompt": "Glossary: Aimee, Shawn, BBQ"},
    ).respond(
        200,
        json={"text": "Prompted transcription"},
    )

    audio_data = MagicMock(spec=AudioData)
    audio_data.get_wav_data.return_value = b"audio_data"

    actual = openai.recognize(
        MagicMock(spec=Recognizer),
        audio_data,
        prompt="Glossary: Aimee, Shawn, BBQ",
    )

    assert actual == "Prompted transcription"
