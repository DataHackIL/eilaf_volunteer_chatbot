"""The boot-time wiring check: every gap named at once, each with its fix."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.whatsapp import config, pipeline_singleton, server


@pytest.fixture
def wired(monkeypatch, tmp_path):
    """A fully wired local instance; each test then breaks one thing."""
    for name in ("ACCESS_TOKEN", "PHONE_NUMBER_ID", "VERIFY_TOKEN"):
        monkeypatch.setattr(config, name, "set")
    monkeypatch.setattr(config, "GENERATOR", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "set")
    monkeypatch.setattr(config, "IS_PROD", False)
    monkeypatch.setattr(config, "corpus_files", lambda: [Path("doc.json")])
    monkeypatch.setattr(config, "ENV_FILE", tmp_path / ".env")
    (tmp_path / ".env").touch()


def test_a_wired_instance_has_no_problems(wired):
    assert config.wiring_problems() == []


def test_every_missing_setting_is_named_at_once(wired, monkeypatch):
    for name in ("ACCESS_TOKEN", "PHONE_NUMBER_ID", "VERIFY_TOKEN"):
        monkeypatch.setattr(config, name, "")
    problems = config.wiring_problems()
    assert len(problems) == 3
    for name in (
        "WHATSAPP_ACCESS_TOKEN",
        "WHATSAPP_PHONE_NUMBER_ID",
        "WHATSAPP_VERIFY_TOKEN",
    ):
        assert any(p.startswith(name) for p in problems)


def test_the_chosen_engines_key_is_required(wired, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY")
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    (problem,) = config.wiring_problems()
    assert "GEMINI_API_KEY" in problem and "EILAF_GENERATOR" in problem


def test_any_one_of_an_engines_keys_will_do(wired, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY")
    monkeypatch.setenv("GOOGLE_API_KEY", "set")
    assert config.wiring_problems() == []


def test_echo_needs_no_key(wired, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY")
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.setattr(config, "GENERATOR", "echo")
    assert config.wiring_problems() == []


def test_an_unknown_engine_lists_the_real_ones(wired, monkeypatch):
    monkeypatch.setattr(config, "GENERATOR", "gpt")
    (problem,) = config.wiring_problems()
    assert "'gpt'" in problem and "gemini" in problem and "bedrock" in problem


def test_every_engine_name_can_be_built():
    assert set(pipeline_singleton.ENGINES) == set(config.ENGINE_KEYS)


def test_an_empty_corpus_is_a_problem(wired, monkeypatch):
    monkeypatch.setattr(config, "corpus_files", lambda: [])
    (problem,) = config.wiring_problems()
    assert "corpus is empty" in problem and "curate_data.sh" in problem


def test_prod_also_needs_the_app_secret_and_the_bucket(wired, monkeypatch):
    monkeypatch.setattr(config, "IS_PROD", True)
    monkeypatch.setattr(config, "APP_SECRET", "")
    monkeypatch.setattr(config, "REPORT_BUCKET", "")
    problems = config.wiring_problems()
    assert any(p.startswith("WHATSAPP_APP_SECRET") for p in problems)
    assert any(p.startswith("EILAF_REPORT_BUCKET") for p in problems)


def test_outside_prod_the_secret_and_bucket_are_optional(wired, monkeypatch):
    monkeypatch.setattr(config, "APP_SECRET", "")
    monkeypatch.setattr(config, "REPORT_BUCKET", "")
    assert config.wiring_problems() == []


def test_a_missing_env_file_is_mentioned_only_when_something_is_missing(
    wired, monkeypatch, tmp_path
):
    """Under compose there is no file, and that is fine while nothing is missing."""
    monkeypatch.setattr(config, "ENV_FILE", tmp_path / "absent")
    assert config.wiring_problems() == []
    monkeypatch.setattr(config, "VERIFY_TOKEN", "")
    assert any("copy .env.example" in p for p in config.wiring_problems())


def test_the_server_refuses_to_boot_and_lists_everything(wired, monkeypatch):
    monkeypatch.setattr(config, "VERIFY_TOKEN", "")
    monkeypatch.setattr(config, "corpus_files", lambda: [])
    loaded = []
    monkeypatch.setattr(server, "get_pipeline", lambda: loaded.append(1))
    with pytest.raises(RuntimeError) as raised:
        with TestClient(server.app):
            pass
    assert "WHATSAPP_VERIFY_TOKEN" in str(raised.value)
    assert "corpus is empty" in str(raised.value)
    assert not loaded  # failed before the slow model load
