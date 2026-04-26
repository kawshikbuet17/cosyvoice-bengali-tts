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
    "warm": "Speak warmly.",
    "smile": "Speak with a smile.",
    "pause": "Pause briefly before speaking.",
    "angry": "Speak angrily.",
    "sad": "Speak sadly.",
    "neutral": "Speak neutrally.",
    "laugh": "Include light laughter.",
    "laughter": "Include light laughter.",
    "happy": "Speak happily.",
    "soft": "Use a soft voice.",
    "loud": "Speak loudly.",
    "fast": "Speak quickly.",
    "slow": "Speak slowly.",

    # tags found in your dataset. syntax: grep -rhoP '\[\K[[:alpha:]_]+(?=\])' *.txt
    "clear": "Speak clearly.",
    "serious": "Speak in a serious tone.",
    "gentle": "Speak gently.",
    "reassuring": "Speak reassuringly.",
    "closing": "Use a closing tone.",
    "farewell": "Speak in a farewell tone.",
    "quickly": "Speak quickly.",
    "accountable": "Speak with an accountable tone.",
    "checking": "Speak as if checking information.",
    "receptive": "Speak in a receptive and attentive tone.",
    "apologetic": "Speak apologetically.",
    "calming": "Speak in a calming tone.",
    "regretful": "Speak with regret.",
    "welcoming": "Speak in a welcoming tone.",
    "informative": "Speak informatively.",
    "attentive": "Speak attentively.",
    "verifying": "Speak as if verifying information.",
    "cooperative": "Speak cooperatively.",
    "empathetic": "Speak empathetically.",
    "polite": "Speak politely.",
    "proactive": "Speak proactively.",
    "appreciative": "Speak appreciatively.",
    "confidently": "Speak confidently.",

    # additional tags from unsupported_tag_stats.tsv
    "in_a_thankful_tone": "Speak with a thankful tone.",
    "in_a_reassuring_tone": "Speak in a reassuring tone.",
    "with_a_warm_smile": "Speak with a warm smile.",
    "in_a_gentle_tone": "Speak in a gentle tone.",
    "in_an_empathetic_tone": "Speak in an empathetic tone.",
    "in_an_enthusiastic_tone": "Speak enthusiastically.",
    "normal_tone": "Speak in a normal tone.",
    "briefly_and_clearly": "Speak briefly and clearly.",
    "in_a_confirming_tone": "Speak in a confirming tone.",
    "slowly_and_clearly": "Speak slowly and clearly.",
    "confident_tone": "Speak with a confident tone.",
    "service_tone": "Speak in a service tone.",
    "slight_emphasis": "Use slight emphasis.",
    "warm_tone": "Speak with a warm tone.",
    "active_listening": "Speak with active listening.",
    "alert_tone": "Speak with an alert tone.",
    "calm_tone": "Speak with a calm tone.",
    "checking_tone": "Speak with a checking tone.",
    "empathetic_tone": "Speak with an empathetic tone.",
    "firm_tone": "Speak with a firm tone.",
    "helpful_closing": "Use a helpful closing tone.",
    "keyboard_typing": "Include keyboard typing sound.",
    "light_laugh": "Include light laughter.",
    "polite_tone": "Speak with a polite tone.",
    "professional_closing": "Use a professional closing tone.",
    "professional_tone": "Speak with a professional tone.",
    "quick_and_professional_tone": "Speak quickly and professionally.",
    "satisfied_tone": "Speak with a satisfied tone.",
    "serious_tone": "Speak with a serious tone.",
    "sincere_tone": "Speak with a sincere tone.",
    "soft_tone": "Speak with a soft tone.",
    "solution_oriented": "Speak in a solution-oriented manner.",
    "suggestive_tone": "Speak with a suggestive tone.",
    "technical_support": "Speak in a technical support tone.",
    "thanking_tone": "Speak with a thanking tone.",
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
