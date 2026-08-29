# Parser native formats — what each parser reads

This is a plain-English reference for the ADRS parsers: what each source system writes
natively, what our parser reads from it, and why. **Guiding principle:** every parser that
is *not* a one-off should read a system's **native output format** as directly as possible.
Curated/normalized variations are handled by one-off parsers under `oneoff_parsers/`.

| Parser | Location | Reads | Kind |
|---|---|---|---|
| `parse_openevolve.py` | `src/` | native OpenEvolve checkpoints/best | permanent, native |
| `parse_gepa.py` | `src/` | native GEPA `gepa_result.json` (GEPAResult) | permanent, native |
| `parse_shinka.py` | `src/` | native ShinkaEvolve `evolution_db.sqlite` | permanent, native |
| `oneoff_parsers/parse_gepa_curated.py` | `oneoff_parsers/` | curated harness bundle (summary.json/.rs) | one-off |
| `oneoff_parsers/ingest_adrs_exp_logs.py` | `oneoff_parsers/` | the ADRS-exp-logs corpus (driver) | one-off |

---

## OpenEvolve — `parse_openevolve.py` (permanent, native)

**What the tool is.** An LLM evolves programs using island-based MAP-Elites (a population
split into "islands", each keeping its best variants).

**What it writes (native output).** One run folder containing:
- `checkpoints/checkpoint_N/` — a snapshot taken every N iterations. Inside each:
  `programs/<id>.json` (one file per program ever tried, holding its code, metrics, the
  `parent_id` it evolved from, generation, timestamp, and the prompts the LLM saw) and
  `metadata.json` (island/archive bookkeeping).
- `best/best_program.py` + `best/best_program_info.json` — the winner and its stats.
- `logs/` — run logs.

(In the ADRS-exp-logs corpus these sit under a `results/` subdir; the parser handles either
location.)

**What the parser reads.** Walks all checkpoints, gathers every program (deduped by id) into
candidates, draws parent→child lineage from each `parent_id`, turns each program's metrics
into measurements, and keeps the best program + files as artifacts.

**Why it's built this way.** `evolution_trace.jsonl` — which the parser used to *require* —
is **not** native OpenEvolve output. It is written by a separate instrumentation layer
(EvoTrace) and is present only for some contributors. The parser now treats it as optional
(timestamps come from the programs themselves when it is absent), so it reads plain,
unmodified OpenEvolve output. The `other_context_ids` context-edge branch is likewise dormant
on native data (not a native `Program` field) — `parent_id` carries the lineage.

---

## GEPA — `parse_gepa.py` (permanent, native)

**What the tool is.** Optimizes prompts/programs by reflective evolution
(github.com/gepa-ai/gepa), tracking a set of candidates and which candidate each one came
from.

**What it writes (native output).** `optimize()` returns a `GEPAResult`; runs also save
`gepa_state.bin` (a resume checkpoint). The result is serialized to `gepa_result.json`, a
self-contained file with:
- `candidates` — list; `candidates[i]` is a dict of named components, e.g. `{"program": code}`.
- `parents` — `parents[i]` is a list of parent indices (or `[None]` for a root). Supports
  multiple parents (crossover).
- `val_aggregate_scores` — one score per candidate; `val_subscores` — per-instance scores.
- `best_idx` — index of the best candidate.

The `candidates/candidate_NNN.py` files are copies of those candidates' code.

**What the parser reads.** Reads `gepa_result.json` (accepts GEPA's own `candidates.json` /
`run_log.json` as a fallback), makes one candidate per entry, draws lineage from `parents`
(multi-parent capable), records scores as measurements, flags the best via `best_idx`, and
keeps the code files + result JSON as artifacts.

**Minimal mode.** Some runs never dumped `gepa_result.json` and left only a `gepa_state.bin`
checkpoint + `best_program.py` (+ `test_metrics.json`). For those, the parser ingests a single
"best" candidate with the final metrics — no per-candidate lineage — rather than failing.

**Why rewritten.** The previous `parse_gepa.py` did not read GEPA's real output — it read a
curated harness bundle. That reader now lives at `oneoff_parsers/parse_gepa_curated.py`.

---

## ShinkaEvolve — `parse_shinka.py` (permanent, native)

**What the tool is.** Sample-efficient program evolution with islands and an archive of good
solutions to cross-pollinate between them (github.com/SakanaAI/ShinkaEvolve).

**What it writes (native output).** A run folder with:
- `evolution_db.sqlite` — a SQLite database that is the source of truth. Its `programs` table
  stores every program with its code, direct `parent_id`, the "inspiration" programs that also
  informed it (`archive_inspiration_ids`, `top_k_inspiration_ids`), generation, island, scores,
  metrics (`public_metrics`/`private_metrics` as JSON), and whether it passed (`correct`). An
  `archive` table lists the kept programs.
- `gen_N/` folders (per-generation `main.py`, `original.py`, `edit.diff`), `meta_memory.json`,
  `experiment_config.yaml`, `evolution_run.log`.

**What the parser reads.** Opens the SQLite DB (read-only), makes one candidate per program
row, draws a `parent` edge from `parent_id` and `context` edges from the inspiration lists,
records scores/metrics as measurements, reads run settings + model names from
`experiment_config.yaml` (YAML optional; a regex fallback recovers `num_islands`/`llm_models`),
and keeps the `gen_N/` files + logs as artifacts.

**Why new.** There was no Shinka parser before.

---

## Curated GEPA bundle — `oneoff_parsers/parse_gepa_curated.py` (one-off)

**What it is.** *Not* a framework output — a normalized bundle a ROAR harness wrote for some
earlier runs (`summary.json`, `iterations.jsonl`, `effective_context.md`, Rust candidate
files). The existing Certus GEPA data is in this shape.

**What it reads.** Exactly what the old top-level `parse_gepa.py` read (behavior unchanged);
it was moved here so the top-level `parse_gepa.py` could become native. `coding_agent` has its
own parser (`parse_coding_agent.py`) and is unaffected.

---

## Corpus driver — `oneoff_parsers/ingest_adrs_exp_logs.py` (one-off)

**Not a format parser.** It walks the ADRS-exp-logs `benchmark/system/run` tree, detects each
run by its native marker files, writes `author_input.json` (Audrey Cheng) into each run, tags
each campaign with its benchmark's research question and the model parsed from the run dir
name, points the right native parser at the run, and inserts everything — skipping the loose
"winner file" folders (`ns3/best_programs`, `ns3/nondeter_best_programs`) that carry no history.

> **Double-blind note.** The `author_input.json` files the driver writes carry the real name
> "Audrey Cheng" into `raw-data/`. They must be scrubbed/anonymized (author → `Roar Author`)
> before building any public `raw-data.tar.gz`, alongside the raw-data directory
> re-anonymization.
