"""Test for Mother's Mouth (Gemini) and Mother's Heart/Mind (Agent/Plan) decoupling."""

import json
import os
import time
from pathlib import Path
import tiktoken
import pytest

from information_boltzmann.core.fly_mother_world import GeminiApiMother, MotherWorld


def test_gemini_mother_dynamic_plan_injection(tmp_path):
    """Verify GeminiApiMother dynamically picks up plan changes from raising.json."""
    plan_file = tmp_path / "raising.json"
    plan_data = {
        "mother": {
            "persona": "A loving mother talking with her baby.",
            "temperature": 0.6,
            "max_reply_tokens": 50,
        },
        "parenting": {
            "current_topic": "the garden",
            "intention": "talk about red flowers and butterflies",
        },
    }
    plan_file.write_text(json.dumps(plan_data), encoding="utf-8")

    mother = GeminiApiMother(plan_path=str(plan_file))
    prompt1 = mother.get_system_prompt()
    assert "the garden" in prompt1
    assert "red flowers and butterflies" in prompt1

    # Modify the plan (Mother's Heart changes mind)
    time.sleep(0.05)
    plan_data["parenting"]["current_topic"] = "space and stars"
    plan_data["parenting"]["intention"] = "whisper about the bright moon"
    plan_file.write_text(json.dumps(plan_data), encoding="utf-8")

    # Prompt must update automatically on next call
    prompt2 = mother.get_system_prompt()
    assert "space and stars" in prompt2
    assert "whisper about the bright moon" in prompt2
    assert "red flowers" not in prompt2


def test_monitor_json_schema(tmp_path):
    """Verify that monitor.json schema matches the contract for the background Mother Mind Agent."""
    # Dummy mock mother
    class MockMother:
        asynchronous = False
        def reply(self, transcript):
            return "Good baby."

    tok = tiktoken.get_encoding("gpt2")
    world = MotherWorld(MockMother(), tok)
    world.speak(tok.encode("hello")[0])
    world.speak(tok.encode(".")[0])

    mon = world.monitor()
    assert "fly_ttr" in mon
    assert "fly_repeat_bigrams" in mon
    assert "fly_identical_consecutive_turns" in mon

    # Verify export format
    mon_dst = tmp_path / "monitor.json"
    mon_data = {
        "trained": 1024,
        "time": "2026-10-14 12:00:00",
        "monitor": mon,
        "recent_transcript": [{"who": w, "text": t} for w, t in world.transcript[-10:]],
        "plan_topic": "animals",
        "plan_intent": "bark like a doggy",
        "plan_mtime": 12345.6,
    }
    mon_dst.write_text(json.dumps(mon_data, ensure_ascii=False, indent=1), encoding="utf-8")
    loaded = json.loads(mon_dst.read_text(encoding="utf-8"))
    assert loaded["trained"] == 1024
    assert loaded["plan_topic"] == "animals"
    assert loaded["monitor"]["fly_turns"] >= 1
