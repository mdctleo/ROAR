#!/usr/bin/env python3
"""System-type detection for campaign folders.

Detection is **owned by the parsers**: each parser exposes a ``matches()`` predicate
next to its ``_resolve_base`` that shares the same marker constants it parses with, so
the detector can't drift from what the parser actually accepts. This module is a thin
ordered registry over those predicates; it is the single source of truth used by the
``/upload`` API path (``src/api.py``) and the CLI/inserter (``src/insert_adrs_campaign.py``,
``src/benchmark_direct_insert.py``).

Order matters only where marker sets overlap: SkyDiscover and OpenEvolve both have a
``checkpoints/`` dir, so SkyDiscover is tried first and claims it only when a
SkyDiscover-specific marker is present (see ``SkyDiscoverParser.matches``); a native
OpenEvolve run then falls through to ``OpenEvolveParser.matches``. shinka/gepa/coding_agent
have disjoint markers, so their relative order is immaterial.
"""

from pathlib import Path
from typing import Callable

from parse_shinka import ShinkaParser
from parse_gepa import GEPAParser
from parse_coding_agent import matches_coding_agent
from oneoff_parsers.parse_gepa_curated import matches_gepa_curated
from parse_skydiscover import SkyDiscoverParser
from parse_openevolve import OpenEvolveParser

# Ordered (system_type, matches) registry. First match wins.
#
# gepa_curated sits after coding_agent: both the curated-GEPA and coding_agent bundles are
# summary.json + iterations.jsonl, disambiguated only by summary.framework, so coding_agent
# (framework=="coding_agent") is tried first and gepa_curated claims framework=="gepa".
# Native gepa/coding_agent runs are unaffected (native gepa has no summary.json).
_REGISTRY: list[tuple[str, Callable[[Path], bool]]] = [
    ("shinka", ShinkaParser.matches),
    ("gepa", GEPAParser.matches),
    ("coding_agent", matches_coding_agent),
    ("gepa_curated", matches_gepa_curated),
    ("skydiscover", SkyDiscoverParser.matches),
    ("openevolve", OpenEvolveParser.matches),
]


def detect_system_type(folder_path: Path) -> str | None:
    """Return the system type of a campaign folder, or None if unrecognized.

    Tries each parser's ``matches()`` predicate in registry order and returns the first
    hit. The predicates mirror each parser's own marker/base-resolution logic, so a
    positive result means the corresponding ``parse_*`` will accept this folder.
    """
    folder_path = Path(folder_path)
    for system_type, matches in _REGISTRY:
        try:
            if matches(folder_path):
                return system_type
        except OSError:
            continue
    return None
