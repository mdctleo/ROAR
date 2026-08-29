#!/usr/bin/env python3
"""ONE-OFF ingestion driver for the `ADRS-exp-logs` corpus (author: Audrey Cheng).

The corpus is laid out as `<benchmark>/<...>/<run>/` over three ADRS frameworks
(openevolve, gepa, shinka), each run storing its framework's native output under a
`results/` subdir. This driver:

  1. Walks the tree and detects each run by its native marker files (robust to the
     irregular nesting: benchmark -> run, benchmark -> system-group -> run, etc.).
  2. Writes `author_input.json` (Audrey Cheng) into each run dir.
  3. Dispatches to the right NATIVE parser (parse_openevolve / parse_gepa / parse_shinka).
  4. Tags each campaign with a name (`<benchmark>/<rel path>`), the benchmark's research
     question, and the model parsed from the run dir name.
  5. Inserts via CampaignInserter (only with --insert; default is a dry run).

Loose "winner file" folders (ns3/best_programs, ns3/nondeter_best_programs) have no
lineage and are skipped.

IMPORTANT (double-blind): the `author_input.json` files this writes carry the REAL name
"Audrey Cheng" into raw-data/. They MUST be scrubbed/anonymized (author -> "Roar Author")
before building any public raw-data.tar.gz, alongside the raw-data dir re-anonymization.
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

# Allow importing top-level modules (adrs_models, parse_*, insert_adrs_campaign) from src/.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from adrs_models import ADRSParsedCampaign  # noqa: E402

DEFAULT_CORPUS = Path("/ws/raw-data/ADRS-exp-logs/ADRS-exp-logs")
AUTHOR = "Audrey Cheng"

# Benchmark -> research question (recovered from each run's experiment_config.yaml task_sys_msg).
RQ_MAP = {
    "cbl": "Can't Be Late: single-region deadline-aware spot-vs-on-demand cloud scheduling to minimize cost (sky_spot strategies).",
    "cbl-multi": "Can't Be Late (multi-region): cloud cost optimization across regions (cant_be_late_multi).",
    "cloudcast": "Cloud infrastructure / overlay optimization (Cloudcast).",
    "llm-sql": "LLM prompt-caching / data optimization for SQL workloads.",
    "mas": "Evolving a multi-agent system architecture.",
    "ns3": "Datacenter TCP congestion control (DatacenterTCPBench, ns-3).",
    "other_cases": "Systems optimization task (eplb / prism / telemetry_repair / txn_scheduling).",
}
OTHER_CASES_SUB = {
    "eplb": "Expert-parallelism load balancing (EPLB) for MoE expert placement.",
    "prism": "GPU model placement optimization (prism).",
    "telemetry_repair": "Network telemetry repair algorithm.",
    "txn_scheduling": "Database transaction scheduling optimization.",
}

SKIP_DIRS = {"best_programs", "nondeter_best_programs", "__MACOSX"}


def _has(dirp: Path, *names: str) -> bool:
    """True if any of `names` exists in dirp or dirp/results."""
    for n in names:
        if (dirp / n).exists() or (dirp / "results" / n).exists():
            return True
    return False


def classify(dirp: Path) -> str | None:
    """Identify the framework of a run dir by its native marker files, else None."""
    if _has(dirp, "evolution_db.sqlite"):
        return "shinka"
    if _has(dirp, "gepa_result.json", "candidates.json", "gepa_state.bin"):
        return "gepa"
    if (dirp / "checkpoints").is_dir() or (dirp / "results" / "checkpoints").is_dir():
        return "openevolve"
    return None


def find_runs(root: Path) -> list[tuple[Path, str]]:
    """Recursively find (run_dir, framework); stop descending once a run is identified."""
    runs: list[tuple[Path, str]] = []

    def rec(d: Path):
        t = classify(d)
        if t:
            runs.append((d, t))
            return  # a run dir is a leaf for our purposes; don't descend into results/
        for child in sorted(d.iterdir()):
            if not child.is_dir() or child.name.startswith(".") or child.name in SKIP_DIRS:
                continue
            rec(child)

    rec(root)
    return runs


def parse_model_from_name(name: str) -> list[str] | None:
    """Extract the LLM name from a run dir name (gpt-5, gemini-3-pro-preview, claude-*)."""
    low = name.lower()
    if "gemini" in low:
        return ["gemini-3-pro-preview"]
    if "gpt-5" in low or "gpt5" in low:
        return ["gpt-5"]
    m = re.search(r"(claude-[\w.-]+)", low)
    if m:
        return [m.group(1)]
    return None


def resolve_research_question(benchmark: str, run_dir: Path) -> str | None:
    """Benchmark RQ; for other_cases, resolve the sub-task from path/config text."""
    if benchmark == "other_cases":
        haystack = str(run_dir).lower()
        cfg = run_dir / "results" / "experiment_config.yaml"
        if not cfg.exists():
            cfg = run_dir / "experiment_config.yaml"
        try:
            if cfg.is_file():
                haystack += "\n" + cfg.read_text().lower()
        except OSError:
            pass
        for key, rq in OTHER_CASES_SUB.items():
            if key in haystack:
                return rq
    return RQ_MAP.get(benchmark)


def inject_author_input(run_dir: Path, author: str, version: str = "") -> None:
    """Write author_input.json into the run dir if not already present."""
    path = run_dir / "author_input.json"
    if not path.exists():
        data: dict = {"author": author}
        if version:
            data["version"] = version
        with open(path, "w") as f:
            json.dump(data, f, indent=2)


def _parse(framework: str, run_dir: Path) -> ADRSParsedCampaign | None:
    if framework == "gepa":
        from parse_gepa import parse_gepa_campaign

        return parse_gepa_campaign(run_dir)
    if framework == "openevolve":
        from parse_openevolve import parse_openevolve_campaign

        return parse_openevolve_campaign(run_dir)
    if framework == "shinka":
        from parse_shinka import parse_shinka_campaign

        return parse_shinka_campaign(run_dir)
    return None


def process_run(
    run_dir: Path, framework: str, corpus_root: Path, insert: bool, db_url: str | None
) -> bool:
    """Inject author, parse, tag campaign fields, and optionally insert."""
    rel = run_dir.relative_to(corpus_root)
    benchmark = rel.parts[0]

    # Only mutate raw-data during a real ingestion; dry runs stay side-effect-free.
    if insert:
        inject_author_input(run_dir, AUTHOR)

    parsed = _parse(framework, run_dir)
    if not parsed:
        print(f"  FAILED to parse: {rel} [{framework}]")
        return False

    # Post-parse tagging.
    parsed.campaign.name = str(rel)
    if not parsed.campaign.author:
        parsed.campaign.author = AUTHOR
    if not parsed.campaign.research_question:
        parsed.campaign.research_question = resolve_research_question(benchmark, run_dir)
    if not parsed.campaign.models_used:
        parsed.campaign.models_used = parse_model_from_name(run_dir.name)

    n_meas = sum(len(m) for m in parsed.measurements.values())
    print(
        f"  {rel} [{framework}] -> {len(parsed.candidates)} candidates, "
        f"{len(parsed.candidate_edges)} edges, {n_meas} measurements, "
        f"{len(parsed.artifacts)} artifacts, models={parsed.campaign.models_used}"
    )

    if insert:
        from insert_adrs_campaign import CampaignInserter

        inserter = CampaignInserter(database_url=db_url)
        if not inserter.insert(parsed):
            print(f"  INSERT FAILED: {rel}")
            return False
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("corpus", nargs="?", type=Path, default=DEFAULT_CORPUS,
                    help=f"Corpus root (default: {DEFAULT_CORPUS})")
    ap.add_argument("--insert", action="store_true",
                    help="Insert into the DB (default: dry run, parse only)")
    ap.add_argument("--db-url", default=None,
                    help="DATABASE_URL for insertion (default: env / library default)")
    ap.add_argument("--benchmark", default=None,
                    help="Only process runs under this top-level benchmark dir")
    ap.add_argument("--framework", default=None, choices=["gepa", "openevolve", "shinka"],
                    help="Only process this framework")
    ap.add_argument("--limit", type=int, default=None, help="Process at most N runs")
    args = ap.parse_args()

    corpus_root = args.corpus.resolve()
    if not corpus_root.is_dir():
        print(f"Error: corpus root not found: {corpus_root}")
        return 1

    if args.db_url:
        os.environ["DATABASE_URL"] = args.db_url

    runs = find_runs(corpus_root)
    if args.benchmark:
        runs = [r for r in runs if r[0].relative_to(corpus_root).parts[0] == args.benchmark]
    if args.framework:
        runs = [r for r in runs if r[1] == args.framework]
    if args.limit is not None:
        runs = runs[: args.limit]

    counts: dict[str, int] = {}
    for _, fw in runs:
        counts[fw] = counts.get(fw, 0) + 1
    mode = "INSERT" if args.insert else "DRY RUN"
    print(f"[{mode}] {len(runs)} runs under {corpus_root}: {counts}\n")

    ok = 0
    fail = 0
    for run_dir, framework in runs:
        try:
            if process_run(run_dir, framework, corpus_root, args.insert, args.db_url):
                ok += 1
            else:
                fail += 1
        except Exception as e:  # keep going on a bad run
            print(f"  ERROR {run_dir.relative_to(corpus_root)}: {e}")
            fail += 1

    print(f"\nDone. {ok} ok, {fail} failed, {len(runs)} total ({mode}).")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    exit(main())
