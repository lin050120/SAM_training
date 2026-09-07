"""Every Chinese string the UI shows must have a Japanese translation.

ui/i18n.py translates by exact string lookup and deliberately leaves unknown
strings in Chinese, so a forgotten entry degrades silently — a new page can ship
fully untranslated and nothing fails. This test walks the real Blocks tree and
fails with the exact missing strings, which is the only signal that catches it.

Scope matches what wire_language_switch() actually translates: label / info /
placeholder on every component, the value of Markdown and Button, and Radio /
Dropdown choice labels. Runtime data (Textbox values, JSON, log tails) is out of
scope by design.
"""

from __future__ import annotations

import re

import pytest

gr = pytest.importorskip("gradio")

from ui.i18n import (  # noqa: E402
    _TEXT_ATTRS,
    _VALUE_TYPES,
    LANGUAGE_CHOICES,
    _ja_text,
)

# A string needs translating only if it actually contains Chinese/Japanese text.
# Pure ASCII labels ("train", "all", "CVAT") are intentionally left alone.
_CJK = re.compile(r"[㐀-䶿一-鿿]")

# Language names in the switch itself must stay readable in both languages.
_EXEMPT = {label for label, _ in LANGUAGE_CHOICES} | {
    # ui/i18n.py:build_language_radio - the switch's own label is already
    # trilingual, and wire_language_switch() skips that component outright.
    "Language / 语言 / 言語",
}

# Runtime-interpolated text cannot live in a static lookup table: the string
# differs per machine. ui/inference_page.py builds this one from the detected
# torch/CUDA state at page-build time.
_EXEMPT_PREFIXES = ("**CUDA 检测**:",)


def _candidates(block) -> list[str]:
    out: list[str] = []
    for attr in _TEXT_ATTRS:
        value = getattr(block, attr, None)
        if isinstance(value, str):
            out.append(value)
    if isinstance(block, _VALUE_TYPES):
        value = getattr(block, "value", None)
        if isinstance(value, str):
            out.append(value)
    choices = getattr(block, "choices", None)
    if choices:
        for choice in choices:
            label = choice[0] if isinstance(choice, tuple) else choice
            if isinstance(label, str):
                out.append(label)
    return out


def test_every_chinese_ui_string_has_a_japanese_translation():
    from app import build_app

    demo = build_app()
    missing: set[str] = set()
    for block in demo.blocks.values():
        for text in _candidates(block):
            stripped = text.strip()
            if not stripped or stripped in _EXEMPT:
                continue
            if stripped.startswith(_EXEMPT_PREFIXES):
                continue
            if _CJK.search(stripped) and _ja_text(stripped) is None:
                missing.add(stripped)

    assert not missing, (
        f"{len(missing)} UI string(s) have no entry in ui.i18n.JA:\n"
        + "\n".join(f"  {s!r}" for s in sorted(missing))
    )


def test_language_switch_returns_one_update_per_tracked_block():
    from app import build_app

    demo = build_app()
    updates = demo.i18n_switch("ja")
    assert len(updates) == len(demo.i18n_tracked)
    # Switching back must restore the exact Chinese source strings.
    for update, (_, attrs) in zip(demo.i18n_switch("zh"), demo.i18n_tracked):
        for attr, (original, _translated) in attrs.items():
            assert update[attr] == original
