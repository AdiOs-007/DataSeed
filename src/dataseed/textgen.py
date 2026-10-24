"""Synthetic text generators (names, e-mails, phones, free text, templates).

All values are *obviously synthetic*: e-mails use reserved domains (``example.com`` by default),
phone numbers use the 555-01XX range that North American numbering reserves for fiction, and
words come from a small made-up vocabulary. DataSeed does not guarantee that a generated value
is unused in the real world, and it does not model semantic realism of free text.
"""
from __future__ import annotations

import numpy as np

from .exceptions import GenerationError

FIRST_NAMES = ["Aarav", "Priya", "Liam", "Sofia", "Mateo", "Aisha", "Noah", "Mei", "Omar", "Hannah",
               "Arjun", "Elena", "Kofi", "Yuki", "Lucas", "Fatima", "Rohan", "Isabella", "Ethan", "Anika",
               "Diego", "Zara", "Kabir", "Chloe", "Ivan", "Nadia", "Samuel", "Meera", "Leo", "Amara",
               "Vikram", "Grace", "Hugo", "Leila", "Dev", "Olivia", "Tariq", "Sana", "Jonas", "Ananya"]
LAST_NAMES = ["Sharma", "Garcia", "Okafor", "Tanaka", "Silva", "Khan", "Mueller", "Nguyen", "Patel", "Rossi",
              "Singh", "Martin", "Haddad", "Kim", "Costa", "Ibrahim", "Reddy", "Lopez", "Novak", "Gupta",
              "Anderson", "Mehta", "Ali", "Kowalski", "Fernandez", "Joshi", "Chen", "Duarte", "Verma", "Larsen",
              "Nair", "Brown", "Ahmed", "Santos", "Iyer", "Walker", "Bose", "Moreau", "Rao", "Clarke"]
CITIES = ["Mumbai", "Delhi", "Bengaluru", "Chennai", "Kolkata", "Pune", "Hyderabad", "Jaipur", "London",
          "Manchester", "Toronto", "Vancouver", "New York", "Chicago", "Austin", "Seattle", "Berlin",
          "Madrid", "Paris", "Lisbon", "Sydney", "Melbourne", "Singapore", "Dubai", "Nairobi", "Lagos",
          "Cairo", "Tokyo", "Osaka", "Seoul"]
STREETS = ["Maple", "Oak", "Cedar", "Lake", "Hill", "Park", "River", "Sunset", "Station", "Market",
           "Garden", "Church", "Mill", "Spring", "Forest"]
STREET_TYPES = ["Street", "Road", "Avenue", "Lane", "Drive"]
COMPANY_A = ["Blue", "North", "Bright", "Apex", "Nova", "Summit", "Delta", "Quantum", "Green", "Silver"]
COMPANY_B = ["Labs", "Systems", "Works", "Dynamics", "Partners", "Logistics", "Foods", "Analytics"]
COMPANY_C = ["Ltd", "Inc", "LLC", "Group"]
WORDS = ["alpha", "river", "stone", "market", "signal", "ember", "orbit", "violet", "harbor", "canvas",
         "meadow", "pixel", "lantern", "quartz", "willow", "summit", "ripple", "cobalt", "thistle", "maple",
         "echo", "fable", "grove", "horizon", "island", "jasper", "kernel", "lumen", "mosaic", "nectar"]
EMAIL_DOMAINS = ["example.com", "example.org", "example.net"]


def _pick(rng, options, m):
    return rng.choice(np.asarray(options, dtype=object), size=m)


def _cat(*parts):
    out = np.asarray(parts[0])
    for p in parts[1:]:
        out = np.char.add(out.astype(str), np.asarray(p).astype(str))
    return out.astype(object)


def fill_template(pattern: str, m: int, rng: np.random.Generator) -> np.ndarray:
    """``#`` -> random digit, ``?`` -> random capital letter, anything else is literal."""
    cols = []
    for ch in pattern:
        if ch == "#":
            cols.append(rng.integers(0, 10, m).astype(str).astype(object))
        elif ch == "?":
            cols.append(np.array(list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"), dtype=object)[rng.integers(0, 26, m)])
        else:
            cols.append(np.full(m, ch, dtype=object))
    if not cols:
        return np.full(m, "", dtype=object)
    out = cols[0].astype(str)
    for c in cols[1:]:
        out = np.char.add(out, c.astype(str))
    return out.astype(object)


def generate_text(kind: str | None, m: int, rng: np.random.Generator, pattern: str | None = None,
                  domain: str | None = None) -> np.ndarray:
    """Generate ``m`` synthetic strings of the given ``kind`` (see ``schema.TEXT_KINDS``)."""
    kind = kind or ("template" if pattern else "word")
    if kind == "template":
        if not pattern:
            raise GenerationError("text_kind 'template' requires a pattern.")
        return fill_template(pattern, m, rng)
    if kind == "first_name":
        return _pick(rng, FIRST_NAMES, m)
    if kind == "last_name":
        return _pick(rng, LAST_NAMES, m)
    if kind == "name":
        return _cat(_pick(rng, FIRST_NAMES, m), " ", _pick(rng, LAST_NAMES, m))
    if kind == "email":
        local = _cat(np.char.lower(_pick(rng, FIRST_NAMES, m).astype(str)), ".",
                     np.char.lower(_pick(rng, LAST_NAMES, m).astype(str)))
        dom = np.full(m, domain, dtype=object) if domain else _pick(rng, EMAIL_DOMAINS, m)
        return _cat(local, "@", dom)
    if kind == "username":
        return _cat(np.char.lower(_pick(rng, FIRST_NAMES, m).astype(str)), rng.integers(10, 9999, m))
    if kind == "phone":
        area = rng.integers(201, 990, m)
        line = rng.integers(0, 100, m)
        return _cat("+1-", area, "-555-01", np.char.zfill(line.astype(str), 2))
    if kind == "address":
        return _cat(rng.integers(1, 999, m), " ", _pick(rng, STREETS, m), " ", _pick(rng, STREET_TYPES, m),
                    ", ", _pick(rng, CITIES, m))
    if kind == "company":
        return _cat(_pick(rng, COMPANY_A, m), " ", _pick(rng, COMPANY_B, m), " ", _pick(rng, COMPANY_C, m))
    if kind == "city":
        return _pick(rng, CITIES, m)
    if kind == "word":
        return _pick(rng, WORDS, m)
    if kind == "sentence":
        n = 6
        parts = [_pick(rng, WORDS, m) for _ in range(n)]
        out = parts[0].astype(str)
        for p in parts[1:]:
            out = np.char.add(np.char.add(out, " "), p.astype(str))
        return np.char.add(np.char.capitalize(out), ".").astype(object)
    raise GenerationError(f"Unknown text_kind {kind!r}.")
