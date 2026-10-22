"""
labels_canonical.py — Single source of truth for SurgQ-VQA v2 vocabulary.

Cholec80 and CholecT50 use different string conventions for the same surgical
concepts (CamelCase vs kebab-case; "GallbladderRetraction" vs
"gallbladder-extraction"; the "carlot-triangle-dissection" typo).  This module
maps every raw label from either dataset to one canonical snake_case token,
used throughout clip extraction, QA generation, CoT generation, and training.

Conventions
-----------
- All canonical tokens are lower-case ASCII with words joined by underscores.
- ``canonicalize_*`` accepts raw-string or int-id; returns canonical str or None.
- ``natural_*`` maps a canonical token to its human-readable form (spaces, no
  underscores) for use in natural-language CoT and answer sentences.
- Vocabulary union covers both datasets so the same token set is used
  regardless of source.
"""
from __future__ import annotations

# =============================================================================
# Canonical vocabularies (the authoritative lists)
# =============================================================================

# 7 phases of laparoscopic cholecystectomy.  Matches the Cholec80 paper ordering
# (Twinanda et al., 2017).  CholecT50's "gallbladder-extraction" refers to the
# same surgical step as Cholec80's "GallbladderRetraction"; we use the earlier
# terminology as canonical to stay consistent with downstream surgical-VQA work.
PHASES = [
    "preparation",
    "calot_triangle_dissection",
    "clipping_cutting",
    "gallbladder_dissection",
    "gallbladder_packaging",
    "cleaning_coagulation",
    "gallbladder_retraction",
]

# 7 instruments.  CholecT50 annotates 6 (no SpecimenBag); Cholec80 annotates 7.
# Canonical set is the union.
INSTRUMENTS = [
    "grasper",
    "bipolar",
    "hook",
    "scissors",
    "clipper",
    "irrigator",
    "specimen_bag",
]

# 10 verbs from the CholecT50 taxonomy (Nwoye et al., 2022).  Cholec80 has no
# verb annotations, so verbs apply only on CholecT50 clips.
VERBS = [
    "grasp",
    "retract",
    "dissect",
    "coagulate",
    "clip",
    "cut",
    "aspirate",
    "irrigate",
    "pack",
    "null_verb",
]

# 15 targets from the CholecT50 taxonomy.
TARGETS = [
    "gallbladder",
    "cystic_plate",
    "cystic_duct",
    "cystic_artery",
    "cystic_pedicle",
    "blood_vessel",
    "fluid",
    "abdominal_wall_cavity",
    "liver",
    "adhesion",
    "omentum",
    "peritoneum",
    "gut",
    "specimen_bag",
    "null_target",
]


# =============================================================================
# Raw → canonical mappings
# =============================================================================
# Covers every phase string we have seen in raw Cholec80 phase files and raw
# CholecT50 category tables.  Any unknown string raises ValueError in
# canonicalize_phase (strict mode) — keeps silent data corruption out.

_PHASE_ALIASES = {
    # Cholec80 (Twinanda et al. 2017) — CamelCase
    "Preparation":              "preparation",
    "CalotTriangleDissection":  "calot_triangle_dissection",
    "ClippingCutting":          "clipping_cutting",
    "GallbladderDissection":    "gallbladder_dissection",
    "GallbladderPackaging":     "gallbladder_packaging",
    "CleaningCoagulation":      "cleaning_coagulation",
    "GallbladderRetraction":    "gallbladder_retraction",
    # CholecT50 (Nwoye et al. 2022) — kebab-case, note the "carlot" typo
    "preparation":                  "preparation",
    "carlot-triangle-dissection":   "calot_triangle_dissection",   # typo fix
    "calot-triangle-dissection":    "calot_triangle_dissection",   # tolerate both
    "clipping-and-cutting":         "clipping_cutting",
    "gallbladder-dissection":       "gallbladder_dissection",
    "gallbladder-packaging":        "gallbladder_packaging",
    "cleaning-and-coagulation":     "cleaning_coagulation",
    "gallbladder-extraction":       "gallbladder_retraction",
    "gallbladder-retraction":       "gallbladder_retraction",
}

_INSTRUMENT_ALIASES = {
    # Cholec80 column headers — Title-Case
    "Grasper":     "grasper",
    "Bipolar":     "bipolar",
    "Hook":        "hook",
    "Scissors":    "scissors",
    "Clipper":     "clipper",
    "Irrigator":   "irrigator",
    "SpecimenBag": "specimen_bag",
    # CholecT50 category names — lower-case
    "grasper":      "grasper",
    "bipolar":      "bipolar",
    "hook":         "hook",
    "scissors":     "scissors",
    "clipper":      "clipper",
    "irrigator":    "irrigator",
    "specimen_bag": "specimen_bag",
    "specimenbag":  "specimen_bag",
}

# Verbs and targets are only provided by CholecT50 and its strings already
# match canonical form — identity mapping, kept explicit for safety.
_VERB_ALIASES   = {v: v for v in VERBS}
_TARGET_ALIASES = {t: t for t in TARGETS}


# =============================================================================
# Canonicalization functions (strict — unknown input raises)
# =============================================================================

def canonicalize_phase(name: str) -> str:
    """Map any raw phase string to its canonical snake_case token."""
    if name in _PHASE_ALIASES:
        return _PHASE_ALIASES[name]
    raise ValueError(f"Unknown phase string: {name!r}")


def canonicalize_instrument(name: str) -> str:
    if name in _INSTRUMENT_ALIASES:
        return _INSTRUMENT_ALIASES[name]
    raise ValueError(f"Unknown instrument string: {name!r}")


def canonicalize_verb(name: str) -> str:
    if name in _VERB_ALIASES:
        return _VERB_ALIASES[name]
    raise ValueError(f"Unknown verb string: {name!r}")


def canonicalize_target(name: str) -> str:
    if name in _TARGET_ALIASES:
        return _TARGET_ALIASES[name]
    raise ValueError(f"Unknown target string: {name!r}")


def canonicalize_triplet(instrument: str, verb: str, target: str) -> str:
    """Return a canonical "instrument_verb_target" triplet string."""
    return f"{canonicalize_instrument(instrument)}_" \
           f"{canonicalize_verb(verb)}_" \
           f"{canonicalize_target(target)}"


# =============================================================================
# Canonical → human-readable (for CoT natural-language sentences)
# =============================================================================

def natural_phase(canon: str) -> str:
    """'calot_triangle_dissection' -> 'calot triangle dissection'."""
    return canon.replace("_", " ")


def natural_instrument(canon: str) -> str:
    """'specimen_bag' -> 'specimen bag'."""
    return canon.replace("_", " ")


def natural_verb(canon: str) -> str:
    if canon == "null_verb":
        return "no identifiable action"
    return canon.replace("_", " ")


def natural_target(canon: str) -> str:
    if canon == "null_target":
        return "no specific target"
    return canon.replace("_", " ")


def natural_triplet(triplet_str: str) -> str:
    """Pretty-print a canonical triplet string.  Expects exactly 3 parts split
    by underscores *at word boundaries*, but real triplets contain instrument
    words that are themselves multi-token (e.g. specimen_bag).  The canonical
    triplet form joins ``i_v_t`` where ``i`` may contain underscores.  We split
    instead on the known vocabulary."""
    # Greedy longest-prefix match against known instruments, then verbs.
    for i in sorted(INSTRUMENTS, key=len, reverse=True):
        if triplet_str.startswith(i + "_"):
            rest = triplet_str[len(i) + 1:]
            for v in sorted(VERBS, key=len, reverse=True):
                if rest.startswith(v + "_"):
                    t = rest[len(v) + 1:]
                    if t in TARGETS:
                        return (f"{natural_instrument(i)} {natural_verb(v)} "
                                f"{natural_target(t)}")
    # Fallback — underscores to spaces, drop nothing.
    return triplet_str.replace("_", " ")


# =============================================================================
# Test / smoke check
# =============================================================================

def _smoke_test() -> None:
    """Quick self-check: every known raw form survives canonicalization and
    every canonical form appears in its vocabulary list."""
    # Phase: every alias value must be in PHASES.
    for raw, canon in _PHASE_ALIASES.items():
        assert canon in PHASES, f"{raw!r} -> {canon!r} not in PHASES"
    # Instrument.
    for raw, canon in _INSTRUMENT_ALIASES.items():
        assert canon in INSTRUMENTS, f"{raw!r} -> {canon!r} not in INSTRUMENTS"
    # Round-trip a known CholecT50 triplet.
    t = canonicalize_triplet("grasper", "dissect", "cystic_plate")
    assert t == "grasper_dissect_cystic_plate", t
    assert natural_triplet(t) == "grasper dissect cystic plate"
    # Round-trip a Cholec80 phase.
    assert canonicalize_phase("GallbladderRetraction") == "gallbladder_retraction"
    # Typo fix.
    assert canonicalize_phase("carlot-triangle-dissection") == \
           "calot_triangle_dissection"
    # specimen_bag in both vocabularies.
    assert canonicalize_instrument("SpecimenBag") == "specimen_bag"
    # Pretty-print.
    assert natural_phase("calot_triangle_dissection") == "calot triangle dissection"
    assert natural_instrument("specimen_bag") == "specimen bag"
    print("labels_canonical smoke test: OK")


if __name__ == "__main__":
    _smoke_test()
