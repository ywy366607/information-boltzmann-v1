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
    assert "silent_beats" in loaded["monitor"]
    assert "mother_calls" in loaded["monitor"]
    assert "mother_waits" in loaded["monitor"]


def test_mother_world_beat_based_rhythm_and_dark_room_call():
    """Verify that consecutive silent yields trigger a proactive call ('Mengli?')."""
    class MockMother:
        asynchronous = False
        def reply(self, transcript):
            return "Good baby."

    tok = tiktoken.get_encoding("gpt2")
    world = MotherWorld(MockMother(), tok, ticks_per_beat=8, pause_beats_before_reply=1,
                        silent_beats_threshold=3, call_prompt="Mengli?")
    
    assert world.ticks_per_beat == 8
    assert world.silent_beats == 0

    # Pop mother's opening turn so it's fly's turn
    while world.turn == 'mother':
        w, who = world.next_word()
        assert who == 'mother'
    assert world.turn == 'fly'

    # Fly yields 1st time without speaking
    world.yield_turn()
    assert world.silent_beats == 1
    # Drain mother's reply
    while world.turn == 'mother':
        world.next_word()

    # Fly yields 2nd time without speaking
    world.yield_turn()
    assert world.silent_beats == 2
    while world.turn == 'mother':
        world.next_word()

    # Fly yields 3rd time: threshold reached! Mother proactively calls "Mengli?"
    world.yield_turn()
    assert world.stats['mother_calls'] == 1
    assert world.silent_beats == 0  # reset after call
    assert world.transcript[-1] == ('mother', 'Mengli?')

    # When fly speaks, silent_beats stays 0
    while world.turn == 'mother':
        world.next_word()
    world.speak(tok.encode("hello")[0])
    assert world.silent_beats == 0


def test_mailbox_mother_actions_speak_wait_call(tmp_path):
    """Verify MailboxMother protocol with 'speak', 'wait', and 'call' actions."""
    from information_boltzmann.core.fly_mother_world import MailboxMother

    box_dir = tmp_path / "mailbox"
    mailbox = MailboxMother(root=str(box_dir), timeout=10.0)

    # 1. Post a request with silent_beats
    tok = tiktoken.get_encoding("gpt2")
    world = MotherWorld(mailbox, tok, silent_beats_threshold=3, call_prompt="Mengli?")
    world.silent_beats = 2
    world.yield_turn()

    # Outbox should have the request
    outbox_files = list((box_dir / "outbox").glob("*.json"))
    assert len(outbox_files) == 1
    out_data = json.loads(outbox_files[0].read_text(encoding="utf-8"))
    assert out_data["silent_beats"] == 3
    assert out_data["fly_status"] == "silent"
    assert out_data["suggested_action"] == "call"
    rid = out_data["id"]

    # 2. Agent responds with action="call"
    inbox_file = box_dir / "inbox" / f"{rid}.json"
    inbox_file.write_text(json.dumps({"id": rid, "action": "call", "text": "Mengli?"}), encoding="utf-8")

    # wait_for_mother handles "call"
    beats = world.wait_for_mother(poll_interval_s=0.01)
    assert beats >= 1
    assert world.stats['mother_calls'] == 1
    assert world.transcript[-1] == ('mother', 'Mengli?')

    # Drain mother turn
    while world.turn == 'mother':
        world.next_word()

    # 3. Test action="wait" (Agent decides to wait one more beat)
    world.yield_turn()
    outbox_files = list((box_dir / "outbox").glob("*.json"))
    assert len(outbox_files) == 1
    rid2 = json.loads(outbox_files[0].read_text(encoding="utf-8"))["id"]
    inbox_file2 = box_dir / "inbox" / f"{rid2}.json"
    inbox_file2.write_text(json.dumps({"id": rid2, "action": "wait"}), encoding="utf-8")

    beats2 = world.wait_for_mother(poll_interval_s=0.01)
    assert beats2 >= 1
    assert world.stats['mother_waits'] == 1

    # Drain
    while world.turn == 'mother':
        world.next_word()

    # 4. Test action="speak" (Agent responds with a sentence)
    world.yield_turn()
    outbox_files = list((box_dir / "outbox").glob("*.json"))
    assert len(outbox_files) == 1
    rid3 = json.loads(outbox_files[0].read_text(encoding="utf-8"))["id"]
    inbox_file3 = box_dir / "inbox" / f"{rid3}.json"
    inbox_file3.write_text(json.dumps({"id": rid3, "action": "speak", "text": "Mama loves Mengli."}), encoding="utf-8")

    beats3 = world.wait_for_mother(poll_interval_s=0.01)
    assert beats3 >= 1
    assert world.transcript[-1] == ('mother', 'Mama loves Mengli.')

