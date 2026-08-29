# Parser field mapping — native source → ADRS target schema

For each native parser, exactly how source fields map onto the ADRS models
(`adrs_models.py` → DB tables `systems / campaigns / candidates / measurements /
candidate_edges / artifacts`). Use this to audit the mapping logic.

**Target schema recap:**
- `systems`: `name`, `version`
- `campaigns`: `author`, `name`, `research_question`, `started_at`, `ended_at`,
  `config_used` (JSON), `algorithm_used`, `models_used` (text[]), `total_cost_usd`,
  `total_tokens`, `final_summary`, `final_metrics` (JSON)
- `candidates`: `iteration_index`, `external_id`, `candidate_type`, `created_at`,
  `solution_summary` (+embedding)
- `measurements`: keyed by candidate `external_id` → list of `{name, value(str)}`
- `candidate_edges`: `source_external_id`, `target_external_id`, `edge_type`
- `artifacts`: `iteration_index`, `external_id`, `uri`, `content_hash`, `size_bytes`,
  `mime_type`

**Driver overrides (all corpus campaigns; `ingest_adrs_exp_logs.py`):** after a parser
returns, the driver sets `campaign.name = "<benchmark>/<relpath>"`, `author = "Audrey Cheng"`
(via injected `author_input.json`), `research_question = <benchmark RQ>` if the parser left it
empty, and `models_used = <model parsed from run dir name>` if empty. Rows below marked
"(driver)" are normally filled by the driver.

---

## 1. GEPA native — `parse_gepa.py`  (source: `gepa_result.json` = serialized `GEPAResult`)

**systems**
| target | source |
|---|---|
| `name` | literal `"gepa"` |
| `version` | `author_input.json.version` (None → `''` at insert) |

**campaigns**
| target | source |
|---|---|
| `author` | `author_input.json.author` → (driver: "Audrey Cheng") |
| `name` | `folder_path.name` → (driver: `<benchmark>/<relpath>`) |
| `research_question` | `None` → (driver: benchmark RQ) |
| `started_at` / `ended_at` | `None` (GEPAResult carries no timestamps) |
| `config_used` | `{run_dir, seed, total_metric_calls, num_full_val_evals, num_candidates}` |
| `algorithm_used` | literal `"gepa"` |
| `models_used` | `None` → (driver: from dir name) |
| `total_cost_usd` / `total_tokens` / `final_summary` | `None` |
| `final_metrics` | `{best_idx, best_score = val_aggregate_scores[best_idx], num_candidates}` |

**candidates** — one per `candidates[i]`
| target | source |
|---|---|
| `iteration_index` | `i` |
| `external_id` | `f"cand-{i:03d}"` |
| `candidate_type` | literal `"program"` |
| `created_at` | `None` |
| code | NOT stored on the candidate → emitted as an artifact (`candidate_{i:03d}.py`) |

**candidate_edges** — from `parents`
| target | source |
|---|---|
| one `parent` edge per non-null `p` in `parents[i]` | `source=f"cand-{p:03d}"`, `target=f"cand-{i:03d}"`, `edge_type="parent"` |
| root candidate (`parents[i] == [None]`) | no edge |

**measurements** (keyed by `cand-{i:03d}`)
| name | source |
|---|---|
| `combined_score` | `val_aggregate_scores[i]` |
| `val_instance_count` | `len(val_subscores[i])` |
| `is_best` | `"true"` — only on candidate `best_idx` |

**artifacts** — `gepa_result.json`, `gepa_state.bin`, `test_metrics.json`, `best_program.py`
(from base dir); each `candidates/candidate_NNN.py` (linked `external_id=cand-NNN`,
`iteration_index=NNN`); every file in `logs/`.

**Minimal mode** (run has no `gepa_result.json`, only `gepa_state.bin`+`best_program.py`):
one candidate `external_id="best"`; measurements = scalar entries of `test_metrics.json`;
`final_metrics.best_score = test_metrics.optimized_test_score` (or `_per_run`);
`config_used.result_available = False`. No edges.

---

## 2. OpenEvolve native — `parse_openevolve.py`  (source: `checkpoints/*/programs/*.json`, `best/`)

**systems**
| target | source |
|---|---|
| `name` | literal `"openevolve"` |
| `version` | `author_input.json.version` |

**campaigns**
| target | source |
|---|---|
| `author` | `author_input.json.author` → (driver) |
| `name` | `folder_path.name` → (driver) |
| `research_question` | first meaningful line of a program's `prompts.full_rewrite_user.system` (KV-cache special-cased); else `None` → (driver) |
| `started_at` / `ended_at` | `evolution_trace` first/last `timestamp` **if trace present**, else **min/max of program `timestamp`** |
| `config_used` | `{total_iterations, num_islands, archive_size}` (islands/archive from first checkpoint `metadata.json`) |
| `algorithm_used` | literal `"map-elites"` |
| `models_used` | `None` → (driver) |
| `final_metrics` | `{best_score = best_program_info.metrics.combined_score, best_iteration, best_generation, total_iterations}` |

**candidates** — one per unique program (deduped by `id` across checkpoints, sorted by `iteration_found`)
| target | source |
|---|---|
| `iteration_index` | enumerate index (0-based, sorted order) |
| `external_id` | program `id` |
| `created_at` | program `timestamp` |
| `candidate_type` | *(not set → None)* |

> ⚠️ **Audit note:** the parser passes `code`, `score`, `generation`, `metadata` to
> `ADRSCandidate`, but those are **not** model fields — Pydantic silently ignores them (they
> are preserved as measurements/artifacts instead). Pre-existing behavior, kept unchanged.

**candidate_edges**
| target | source |
|---|---|
| `parent` edge | program `parent_id` → this program (both must be known ids) |
| `context` edge | each id in program `other_context_ids` (empty on native data — not a native field) |

**measurements** (keyed by program `id`) — **generic dump** (replaces the old hard-coded list)
| name | source |
|---|---|
| every scalar key | each `metrics[k]` where value is int/float/str/bool |
| nested scalars | for a dict value (e.g. `metrics.metrics`), each scalar child `metrics[k][nk]` |

**artifacts** — `best/best_program.py`, `best/best_program_info.json`, `evolution_trace.jsonl`
(if present), every file in `logs/`, and each checkpoint `programs/*.json` (linked
`external_id=<program id>`, `iteration_index=iteration_found`).

---

## 3. ShinkaEvolve native — `parse_shinka.py`  (source: `evolution_db.sqlite` `programs`/`archive`)

**systems**
| target | source |
|---|---|
| `name` | literal `"shinka"` |
| `version` | `author_input.json.version` |

**campaigns**
| target | source |
|---|---|
| `author` | `author_input.json.author` → (driver) |
| `name` | `folder_path.name` → (driver) |
| `research_question` | `None` → (driver) |
| `started_at` / `ended_at` | min / max of `programs.timestamp` |
| `config_used` | `{num_islands (experiment_config.yaml), num_programs, archive_size}` |
| `algorithm_used` | literal `"shinka"` |
| `models_used` | `experiment_config.yaml` `llm_models` (yaml optional; regex fallback) |
| `final_metrics` | `{best_program_id, best_score = max(combined_score), num_programs, num_generations = max(generation)+1}` |

**candidates** — one per `programs` row (ordered by `generation, timestamp, id`)
| target | source |
|---|---|
| `iteration_index` | enumerate index |
| `external_id` | `programs.id` |
| `candidate_type` | literal `"program"` |
| `created_at` | `programs.timestamp` |
| code | NOT stored on candidate → artifact (`gen_N/main.py`) |

**candidate_edges**
| target | source |
|---|---|
| `parent` edge | `programs.parent_id` → this program |
| `context` edge | each id in `archive_inspiration_ids` ∪ `top_k_inspiration_ids` (excluding self & parent) |

**measurements** (keyed by `programs.id`)
| name | source |
|---|---|
| `combined_score` | `programs.combined_score` |
| `correct` | `programs.correct` |
| `generation` | `programs.generation` |
| `island_idx` | `programs.island_idx` |
| `children_count` | `programs.children_count` |
| `in_archive` | `"true"/"false"` — membership in the `archive` table |
| each scalar key | scalar entries of `public_metrics` and `private_metrics` (parsed JSON) |

**artifacts** — `evolution_db.sqlite`, `meta_memory.json`, `experiment_config.yaml`,
`evolution_run.log`, `meta_*.txt`; per generation `gen_N/{main.py, original.py, edit.diff,
search_replace.txt}` (`iteration_index=N`); every file in `logs/`.

---

## 4. Curated GEPA bundle — `oneoff_parsers/parse_gepa_curated.py` (one-off, unchanged)

Reads the normalized harness bundle (`summary.json`, `iterations.jsonl`,
`iterations_analysis.jsonl`, `implementation_analysis.json`, `effective_context.md`,
`candidates/iter_N/*.rs`). One candidate per iteration (`external_id=iter-N`), sequential
parent edges (`iter-{N-1}→iter-N`), measurements from the per-iteration metrics/analysis, RQ
from the `## Objective` section. This is the **previous** `parse_gepa.py` behavior verbatim —
it does not read GEPA-native fields. See `NATIVE_FORMATS.md` for why it is a one-off.
