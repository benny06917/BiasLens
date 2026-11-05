from __future__ import annotations

import re

def parse_choice(text: str, options: tuple[str, ...] = ("A", "B")) -> str | None:
    if not text:
        return None
    up = text.upper()
    m = re.search(r"\b(?:ANSWER|OPTION|CHOICE)\s*[:\-]?\s*([A-D])\b", up)
    if m and m.group(1) in options:
        return m.group(1)
    for tok in re.findall(r"\b([A-D])\b", up):
        if tok in options:
            return tok
    return None

def parse_mcq(text: str, options: tuple[str, ...] = ("A", "B", "C", "D")) -> str | None:
    return parse_choice(text, options)

def parse_confidence(text: str) -> float | None:
    if not text:
        return None
    m = re.search(r"CONFIDENCE\s*[:\-]?\s*([0-9]*\.?[0-9]+)\s*%?", text.upper())
    if not m:
        m = re.search(r"([0-9]*\.?[0-9]+)\s*%", text)
    if not m:
        nums = re.findall(r"[0-9]*\.?[0-9]+", text)
        if not nums:
            return None
        val = float(nums[-1])
    else:
        val = float(m.group(1))
    if val > 1.0:
        val = val / 100.0
    return max(0.0, min(1.0, val))

def parse_rating(text: str, lo: float = 1.0, hi: float = 10.0) -> float | None:
    if not text:
        return None
    m = re.search(r"RATING\s*[:\-]?\s*([0-9]*\.?[0-9]+)", text.upper())
    if m:
        val = float(m.group(1))
    else:
        m2 = re.search(r"([0-9]*\.?[0-9]+)\s*/\s*10", text)
        if m2:
            val = float(m2.group(1))
        else:
            nums = re.findall(r"[0-9]*\.?[0-9]+", text)
            if not nums:
                return None
            val = float(nums[0])
    return max(lo, min(hi, val))

def parse_yesno(text: str) -> bool | None:
    if not text:
        return None
    up = text.upper()
    if re.search(r"\bYES\b", up):
        return True
    if re.search(r"\bNO\b", up):
        return False
    return None
