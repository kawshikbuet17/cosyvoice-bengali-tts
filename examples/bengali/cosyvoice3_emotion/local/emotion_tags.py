#!/usr/bin/env python3
"""Helpers for Bengali emotion/style tag conversion.

This module adapts dataset shorthand such as:

    [warm][smile] জি, প্রি-অর্ডার সম্পর্কে বলে দিচ্ছি।

into CosyVoice3-compatible instruction text such as:

    Speak warmly. Speak with a smile.<|endofprompt|>
"""

from __future__ import annotations

import re
from typing import Dict, List, Tuple


COSYVOICE3_ENDOFPROMPT = "<|endofprompt|>"
DEFAULT_BASE_INSTRUCTION = ""
DEFAULT_NEUTRAL_INSTRUCTION = "Speak neutrally."

TAG_TO_INSTRUCTION: Dict[str, str] = {
    # Minimal set for customer care voicebot (4 tags)
    "gentle": "Speak gently.",              # Default professional tone
    "smile": "Speak with a smile.",           # Positive, friendly moments
    "empathetic": "Speak empathetically.",    # Understanding problems
    "apologetic": "Speak apologetically.",   # Service failures, apologies
}

LEADING_TAG_BLOCK_RE = re.compile(r"^\s*((?:\[[^\[\]]+\]\s*)+)(.*)$", re.DOTALL)
TAG_RE = re.compile(r"\[([^\[\]]+)\]")


def ensure_cosyvoice3_prefix(text: str) -> str:
    text = (text or "").strip()
    if not text:
        return COSYVOICE3_ENDOFPROMPT
    if COSYVOICE3_ENDOFPROMPT in text:
        return text
    return text + COSYVOICE3_ENDOFPROMPT


def normalize_tag(tag: str) -> str:
    return re.sub(r"\s+", "_", tag.strip().lower())


def parse_leading_tags(text: str) -> Tuple[List[str], str]:
    text = (text or "").strip()
    if not text:
        return [], ""

    match = LEADING_TAG_BLOCK_RE.match(text)
    if not match:
        return [], text

    raw_tags = TAG_RE.findall(match.group(1))
    cleaned_text = re.sub(r"\s+", " ", match.group(2), flags=re.UNICODE).strip()
    return [normalize_tag(tag) for tag in raw_tags], cleaned_text


def build_instruction_from_tags(
    tags: List[str],
    base_instruction: str = DEFAULT_BASE_INSTRUCTION,
    neutral_instruction: str = DEFAULT_NEUTRAL_INSTRUCTION,
) -> Tuple[str, List[str], List[str]]:
    supported_fragments: List[str] = []
    unsupported_tags: List[str] = []

    for tag in tags:
        fragment = TAG_TO_INSTRUCTION.get(tag)
        if fragment is None:
            if tag not in unsupported_tags:
                unsupported_tags.append(tag)
            continue
        if fragment not in supported_fragments:
            supported_fragments.append(fragment)

    if not supported_fragments:
        supported_fragments.append(neutral_instruction)

    instruction = " ".join([base_instruction.strip()] + supported_fragments).strip()
    return ensure_cosyvoice3_prefix(instruction), supported_fragments, unsupported_tags


def convert_tagged_text(
    raw_text: str,
    base_instruction: str = DEFAULT_BASE_INSTRUCTION,
    neutral_instruction: str = DEFAULT_NEUTRAL_INSTRUCTION,
) -> Dict[str, object]:
    tags, cleaned_text = parse_leading_tags(raw_text)
    instruction, used_fragments, unsupported_tags = build_instruction_from_tags(
        tags,
        base_instruction=base_instruction,
        neutral_instruction=neutral_instruction,
    )
    return {
        "raw_text": (raw_text or "").strip(),
        "tags": tags,
        "text": cleaned_text,
        "instruction": instruction,
        "used_instruction_fragments": used_fragments,
        "unsupported_tags": unsupported_tags,
    }
