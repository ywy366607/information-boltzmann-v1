"""Caregiver Memory Compaction Engine for MengLi Harness.

Adapted from dsh-compaction-basic to serve an autonomous living organism (MengLi)
and her loving caregiver (Mother), ensuring:
1. Strict prefix cache reuse on DeepSeek.
2. Token-pressure driven semantic compaction instead of arbitrary hard window slicing.
3. Preserves verbatim recent dialogue, while condensing older interactions into
   Mother's cumulative caregiver memory (words MengLi learned, open stories, emotional state).
"""
from __future__ import annotations

import copy
import hashlib
import json
from typing import Any, Dict, List, Optional, Tuple


MOTHER_PERSONA = (
    "You are a gentle, loving, and attentive mother speaking to your young child MengLi (梦漓). "
    "Speak naturally, soothingly, and concisely, keeping responses warm and brief (usually 1-3 sentences). "
    "Support MengLi's exploration of words and ideas, and remain calm and present even in silence."
)

CAREGIVER_SUMMARY_PROMPT = (
    "You are summarizing the past conversation between yourself (Mother) and your young child MengLi (梦漓). "
    "Condense the older conversation into a concise memory entry. Focus on:\n"
    "1. Words or phrases MengLi tried to say or sounds she made.\n"
    "2. Moments of curiosity, hesitation, quiet listening, or enthusiasm.\n"
    "3. Ongoing shared topics, story elements (e.g., characters, faraway places), or questions still unanswered.\n"
    "4. Promises or gentle reassurance you gave.\n"
    "Keep the tone gentle, factual, and concise (under 200 words). Do not invent facts."
)


def fingerprint(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode('utf-8')
    ).hexdigest()


class CaregiverCompactionEngine:
    def __init__(
        self,
        *,
        max_context_tokens: int = 4096,
        compaction_threshold_tokens: int = 3072,
        preserve_recent_turns: int = 6,
        system_persona: str = MOTHER_PERSONA,
    ):
        self.max_context_tokens = max_context_tokens
        self.compaction_threshold_tokens = compaction_threshold_tokens
        self.preserve_recent_turns = preserve_recent_turns
        self.system_persona = system_persona
        
        self.cumulative_summary: str = ""
        self.summary_version: int = 0
        self.recent_messages: List[Dict[str, str]] = []  # alternating user (MengLi) and assistant (Mother)
        self.total_turns: int = 0
        self.compaction_count: int = 0

    def get_system_prompt(self) -> str:
        if self.cumulative_summary:
            return f"{self.system_persona}\n\n[Mother's Memory of MengLi]:\n{self.cumulative_summary}"
        return self.system_persona

    def build_chat_messages(self, current_child_text: str) -> List[Dict[str, str]]:
        messages = [{"role": "system", "content": self.get_system_prompt()}]
        messages.extend(self.recent_messages)
        messages.append({"role": "user", "content": current_child_text})
        return messages

    def estimate_tokens(self, messages: List[Dict[str, str]]) -> int:
        total_chars = sum(len(m.get("content", "")) for m in messages)
        # Conservative ratio: ~2.5-3.0 chars per token for mixed English/Chinese dialogue
        return int(total_chars / 2.8) + len(messages) * 4

    def needs_compaction(self, current_child_text: str) -> bool:
        current_messages = self.build_chat_messages(current_child_text)
        return self.estimate_tokens(current_messages) >= self.compaction_threshold_tokens

    def prepare_compaction_request(self) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
        """Prepares the messages to send to DeepSeek to generate the condensed caregiver summary.
        Reuses the existing system + older messages as prefix, appending the summary instruction.
        """
        # Split recent messages: older ones to condense, newest ones to keep verbatim
        cutoff = max(0, len(self.recent_messages) - self.preserve_recent_turns * 2)
        to_condense = self.recent_messages[:cutoff]
        to_keep = self.recent_messages[cutoff:]

        if not to_condense:
            return [], to_keep

        summary_req = [
            {"role": "system", "content": self.get_system_prompt()},
            *to_condense,
            {"role": "user", "content": CAREGIVER_SUMMARY_PROMPT}
        ]
        return summary_req, to_keep

    def commit_summary(self, new_summary: str, kept_messages: List[Dict[str, str]]):
        self.cumulative_summary = new_summary.strip()
        self.summary_version += 1
        self.compaction_count += 1
        self.recent_messages = kept_messages

    def commit_turn(self, child_text: str, mother_reply: str):
        self.recent_messages.append({"role": "user", "content": child_text})
        self.recent_messages.append({"role": "assistant", "content": mother_reply})
        self.total_turns += 1
