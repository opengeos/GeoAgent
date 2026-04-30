"""Tests for OpenGeoAgent voice input helpers."""

import sys
import types

from open_geoagent.dialogs.chat_dock import VoiceTranscriptionWorker


class _EmptySettings:
    def value(self, _key, default="", type=str):  # noqa: A002
        return default


class _Settings:
    def __init__(self, values):
        self.values = dict(values)

    def value(self, key, default="", type=str):  # noqa: A002
        return self.values.get(key, default)


def test_voice_transcription_requires_openai_api_key(monkeypatch, tmp_path) -> None:
    """Voice transcription should fail clearly when only Codex OAuth is present."""
    audio_path = tmp_path / "voice.wav"
    audio_path.write_bytes(b"not-a-real-wav")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_CODEX_ACCESS_TOKEN", "codex-token")

    results = []
    worker = VoiceTranscriptionWorker(str(audio_path), _EmptySettings())
    worker.finished.connect(results.append)
    worker.run()

    assert results
    assert results[0]["success"] is False
    assert "Voice transcription requires an OpenAI API key" in results[0]["error"]


def test_voice_transcription_uses_saved_model(monkeypatch, tmp_path) -> None:
    """Voice transcription should use the QSettings transcription model."""
    from open_geoagent.dialogs.chat_dock import SETTINGS_PREFIX

    audio_path = tmp_path / "voice.wav"
    audio_path.write_bytes(b"not-a-real-wav")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("OPENAI_TRANSCRIPTION_MODEL", "env-model")

    captured = {}

    class _FakeTranscriptions:
        def create(self, **kwargs):
            captured.update(kwargs)
            return types.SimpleNamespace(text="hello")

    class _FakeAudio:
        def __init__(self):
            self.transcriptions = _FakeTranscriptions()

    class _FakeOpenAI:
        def __init__(self, **kwargs):
            captured["client_kwargs"] = kwargs
            self.audio = _FakeAudio()

    openai_module = types.ModuleType("openai")
    openai_module.OpenAI = _FakeOpenAI
    monkeypatch.setitem(sys.modules, "openai", openai_module)

    results = []
    worker = VoiceTranscriptionWorker(
        str(audio_path),
        _Settings({f"{SETTINGS_PREFIX}transcription_model": "gpt-4o-transcribe"}),
    )
    worker.finished.connect(results.append)
    worker.run()

    assert results[0]["success"] is True
    assert results[0]["text"] == "hello"
    assert captured["model"] == "gpt-4o-transcribe"
    assert captured["client_kwargs"]["api_key"] == "sk-test"
