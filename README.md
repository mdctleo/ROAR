# ADRS - Agentic Discovery Research Storage

A PostgreSQL-based system for storing and analyzing data from agentic research campaigns.

## Architecture Overview

The parsing system uses a **common intermediate format** based on Pydantic models. This allows multiple research systems to be supported through separate parsers that all convert to the same intermediate representation before database insertion.

```
┌──────────────┐       ┌──────────────┐
│ SkyDiscover  │       │  Future      │
│   Output     │       │  System      │
└──────┬───────┘       └──────┬───────┘
       │                      │
       │ parse_skydiscover.py │ parse_X.py
       ▼                      ▼
┌──────────────────────────────────────────────────────────┐
│            Common ADRS Models (adrs_models.py)           │
│  - ADRSCampaign, ADRSCandidate, ADRSMeasurement, etc.    │
│  - Pydantic-based validation                             │
└────────────────────────────┬─────────────────────────────┘
                             │
                             │ insert_adrs_campaign.py
                             ▼
┌──────────────────────────────────────────────────────────┐
│                  PostgreSQL Database                      │
│  - campaigns, candidates, measurements, candidate_edges   │
└──────────────────────────────────────────────────────────┘
```

---

## Database Schema

### Entity Overview

The database consists of 6 core tables that capture the structure and results of agentic research campaigns:

| Table | Purpose | Foreign Keys |
|-------|---------|--------------|
| `systems` | Research frameworks (SkyDiscover, etc.) | - |
| `campaigns` | Top-level research runs | `system_id` → systems |
| `candidates` | Proposed solutions/hypotheses/programs | `campaign_id` → campaigns |
| `candidate_edges` | Relationships between candidates (parent/context) | `source_candidate_id` → candidates, `target_candidate_id` → candidates |
| `measurements` | Named values (metrics, results, outcomes) | `candidate_id` → candidates |
| `artifacts` | File references (stored on disk, not in DB) | `campaign_id` → campaigns, `candidate_id` → candidates |

### Schema SQL

```sql
CREATE TABLE systems (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name TEXT NOT NULL,
    version TEXT
);

CREATE TABLE campaigns (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    system_id UUID REFERENCES systems(id) ON DELETE CASCADE,
    author TEXT NOT NULL,
    name TEXT,
    research_question TEXT,
    research_question_embedding vector(384),
    started_at TIMESTAMP,
    ended_at TIMESTAMP,
    config_used JSONB,
    algorithm_used TEXT,
    models_used TEXT[],
    total_cost_usd DOUBLE PRECISION,
    total_tokens BIGINT,
    final_summary TEXT,
    final_metrics JSONB,
    evaluator_setup JSONB
);

CREATE TABLE candidates (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    campaign_id UUID REFERENCES campaigns(id) ON DELETE CASCADE,
    iteration_index INTEGER NOT NULL,
    external_id TEXT,
    candidate_type TEXT,
    created_at TIMESTAMP,
    solution_summary TEXT,
    solution_summary_embedding vector(384),
    direct_code_embedding vector(768),
    context_code_diversity DOUBLE PRECISION,
    UNIQUE (campaign_id, external_id)
);

CREATE TABLE candidate_edges (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source_candidate_id UUID REFERENCES candidates(id) ON DELETE CASCADE,
    target_candidate_id UUID REFERENCES candidates(id) ON DELETE CASCADE,
    edge_type TEXT DEFAULT 'parent'
);

CREATE TABLE measurements (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    candidate_id UUID REFERENCES candidates(id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    value TEXT NOT NULL
);

CREATE TABLE artifacts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    campaign_id UUID REFERENCES campaigns(id) ON DELETE CASCADE,
    candidate_id UUID REFERENCES candidates(id) ON DELETE CASCADE,
    iteration_index INTEGER,
    uri TEXT NOT NULL,
    content_hash TEXT,
    size_bytes BIGINT,
    mime_type TEXT
);
```

### Indexes

- `idx_candidates_campaign_id` — lookup candidates by campaign
- `idx_candidates_iteration_index` — lookup candidates by (campaign, iteration)
- `idx_candidate_edges_source` / `idx_candidate_edges_target` / `idx_candidate_edges_type` — edge traversal
- `idx_measurements_candidate_id` / `idx_measurements_name` — measurement lookups
- `idx_artifacts_campaign_id` — artifacts by campaign
- `idx_campaigns_research_question_embedding` — HNSW cosine similarity on research question vectors
- `idx_campaigns_models_used` — GIN index for array containment queries
- `idx_candidates_solution_summary_embedding` — HNSW cosine similarity on solution summary vectors
- `idx_candidates_direct_code_embedding` — HNSW cosine similarity on direct code embedding vectors

### Plain-English Description

#### `systems`

A `system` is the ADRS framework or implementation that produced the data.

Examples: `skydiscover`

A system record declares where the output came from, storing the framework name and version.

#### `campaigns`

A `campaign` is one top-level investigation or discovery run, linked to a `system` via `system_id`.

For SkyDiscover/OpenEvolve, a campaign corresponds to one run/output directory.

A campaign declares:
- Which system produced it (system_id)
- Name and research question
- Research question embedding (384-dim vector, backfilled separately)
- Run timestamps (started_at, ended_at)
- Models and algorithm used
- Cost and token usage
- Final summary and metrics
- Evaluator setup (for cross-campaign comparison)

#### `candidates`

A `candidate` is the central object in the schema - **the thing the system proposed**.

Each candidate belongs directly to a campaign and has an `iteration_index` indicating which step of exploration it came from.

For SkyDiscover/OpenEvolve, each candidate represents **one unique program**. The `iteration_index` indicates when it was first discovered.

**Note:** The actual content (code for programs) is stored in artifacts, not in the candidates table. This keeps the database lean and avoids duplication when the same program appears across multiple checkpoints. Use the `external_id` to look up the corresponding artifact.

**Precomputed analytics columns:**
- `context_code_diversity`: Average pairwise cosine distance among context candidates' code embeddings. Computed during embedding import to avoid expensive O(n²) queries at analytics time.

#### `candidate_edges`

A `candidate_edge` describes a relationship between two candidates, enabling lineage tracking and graph visualizations. The `edge_type` column distinguishes relationship types:

- `parent` (default): The target candidate was derived from the source candidate
- `context`: The source candidate was shown as historical context when generating the target (e.g., BoN history)

For SkyDiscover/OpenEvolve, edges represent:
- `parent`: Parent program that was mutated to create the child
- `context`: Programs shown as history in Best-of-N sampling


#### `measurements`

A `measurement` is a named value associated with a candidate. All evaluation outcomes, metrics, and observations are stored as simple name-value pairs.

Table structure:
- `candidate_id`: Links to the candidate
- `name`: Measurement name (e.g., "status", "makespan", "predicted")  
- `value`: Measurement value as text (e.g., "SUCCEED", "327.0", "performance improves")

#### `artifacts`

An `artifact` is a reference to a file stored on disk. The database stores only metadata (URI, hash, size, type) for tracking and integrity verification.

Files are preserved in the campaign directory structure and accessed via the `uri` field. Artifacts link to a campaign and optionally to an `iteration_index` if they are iteration-specific.

--

### SkyDiscover Mapping

#### System Fields

| ADRS Column | SkyDiscover Source |
|-------------|-------------------|
| `name` | `"skydiscover"` |
| `version` | `"0.3.0"` |

#### Campaign Fields

| ADRS Column | SkyDiscover Source |
|-------------|-------------------|
| `system_id` | FK to systems table (skydiscover) |
| `name` | Folder name or `summary.json → output_dir` |
| `research_question` | `config.yaml → research_question` (if present) |
| `started_at` | First timestamp in log files |
| `ended_at` | Last timestamp in log files |
| `models_used` | `config.yaml → llm.models` (JSONB) |
| `algorithm_used` | `"openevolve"` |
| `total_cost_usd` | Not tracked |
| `total_tokens` | Not tracked |
| `final_summary` | Not tracked |
| `final_metrics` | `summary.json` (best_score, etc.) |
| `evaluator_setup` | LLM extraction from `evaluator.py` code (JSONB) |

#### Candidate Fields

One candidate per unique program. The `iteration_index` indicates when the program was first discovered (`iteration_found`). Iteration 0 contains seed programs (initial population before search begins). Multiple programs can share the same `iteration_found` value.

| ADRS Column | SkyDiscover Source |
|-------------|-------------------|
| `campaign_id` | FK to campaigns table |
| `iteration_index` | `program.iteration_found` (when first discovered) |
| `external_id` | `program.id` (UUID) |
| `candidate_type` | `"program"` |
| `created_at` | `program.timestamp` |

Note: Program code is stored in artifacts, not in the candidates table. Use the `external_id` (UUID) to look up the corresponding program file in the `checkpoints/` artifacts.

#### Measurements

All metrics from program evaluation:
- `status`: VALID or INVALID (based on validity metric)
- `combined_score`: Overall fitness score
- `validity`: 1.0 or 0.0
- `result_summary`: Formatted string of all metrics
- Any other metrics from the evaluator (stored individually)

#### Candidate Edges

Two edge types for evolution history:
- `parent` (edge_type='parent'): The program that was mutated to create this one (`parent_id`)
- `context` (edge_type='context'): Programs shown as history in Best-of-N sampling (`other_context_ids`)

#### Artifacts

Config file, summary, best program, logs, checkpoint metadata.

---

## Development

### Run the Development Server

```bash
 uv run uvicorn api:app --host 0.0.0.0 --port 8000
```

---

## API Usage

### Upload Endpoint

Upload campaign data archives via HTTP. Uses background processing with polling for large files.

**Endpoints:**
- `POST /upload` - Submit archive, returns job_id immediately
- `GET /upload/status/{job_id}` - Poll for processing progress
- `GET /upload/jobs` - List all jobs

**Parameters:**
- `file` (required): tar.gz archive containing campaign data
- `author` (required): Email address of the uploader
- `system_type` (optional): `skydiscover` or `auto` (default: `auto`)

**Example:**
```bash
# 1. Upload (returns immediately with job_id)
curl -X POST http://localhost:8000/upload \
  -F 'file=@campaigns.tar.gz' \
  -F 'author=user@example.com'
# Response: {"job_id": "abc-123", "status_url": "/upload/status/abc-123", ...}

# 2. Poll for status
curl http://localhost:8000/upload/status/abc-123
# Response: {"status": "processing", "progress": 50, "total": 100, ...}
# When complete: {"status": "complete", "result": {...}}
```

**Future:** Raw files will be stored in IBM Cloud Object Storage instead of pod-local disk, enabling stateless pods and horizontal scaling.

---

## Field Mapping from Source Systems

This section documents how source system fields map to the simplified ADRS schema.

---

## Embedding Backfill

Embeddings are decoupled from insertion for fast campaign ingestion. After inserting campaigns, run the backfill scripts separately:

```bash
# Fast insertion (no embedding or LLM calls)
python insert_adrs_campaign.py --all /path/to/campaigns

# Backfill research question embeddings (campaigns table)
python research_questions_embed.py --batch-size 500

# Backfill candidate solution summary embeddings (LLM summarize → text embed, 384-dim)
python candidate_embed.py --mode summarize --batch-size 500

# Backfill candidate direct code embeddings (code embedder, 768-dim)
python candidate_embed.py --mode direct --batch-size 500
```

Models used:
- `BAAI/bge-small-en-v1.5` (384-dim) — research questions and solution summaries
- `jinaai/jina-embeddings-v2-base-code` (768-dim) — direct code embeddings

All scripts process only rows where the target embedding column is NULL, so they're safe to re-run. Set `FASTEMBED_GPU=1` to enable CUDA acceleration.

### GPU Cluster Workflow

For large batches, use the export/embed/import pipeline to run embeddings on a GPU cluster:

```bash
# 1. Export (runs locally, LLM calls cached for summarize mode)
python export_for_embedding.py --mode summarize --output summaries.jsonl
python export_for_embedding.py --mode direct --output code_to_embed.jsonl

# 2. Embed on GPU cluster (upload files, submit job)
FASTEMBED_GPU=1 python embed_batch.py -i summaries.jsonl -o summary_embeddings.jsonl --model text
FASTEMBED_GPU=1 python embed_batch.py -i code_to_embed.jsonl -o code_embeddings.jsonl --model code

# 3. Import (runs locally, updates database)
python import_embeddings.py --mode summarize --summaries summaries.jsonl --embeddings summary_embeddings.jsonl
python import_embeddings.py --mode direct --embeddings code_embeddings.jsonl
```

All scripts support `--dry-run` to preview what would be processed.

---

## OpenShift Deployment

### Quick Start (Fresh Deploy)

```bash
# 1. Login to OpenShift
oc login <your-cluster-url>

# 2. Select your project
oc project leolin2-ns

# 3. Create build config with resource limits for heavy ML dependencies
oc new-build --name=adrs-data --binary --strategy=docker
oc patch bc/adrs-data -p '{"spec":{"resources":{"limits":{"memory":"4Gi","cpu":"1","ephemeral-storage":"20Gi"},"requests":{"memory":"4Gi","cpu":"1","ephemeral-storage":"20Gi"}}}}'

# 4. Build the image (from src/ directory, excluding dev files)
oc start-build adrs-data --from-dir=./src --exclude='\.venv|__pycache__|\.pyc$|^data$' --follow

# 5. Deploy everything
oc apply -f k8s-openshift.yaml

# 6. Watch pods start (Ctrl+C when ready)
oc get pods -w

# 7. Test the API
curl http://adrs-data.leolin2-ns.vpc-int.res.ibm.com/health
```

### Clean Up and Redeploy (After Code Changes)

```bash
# 1. Delete all resources
oc delete -f k8s-openshift.yaml
oc delete bc/adrs-data
oc delete is/adrs-data
oc delete pods --all

# 2. Verify clean
oc get all

# 3. Rebuild and deploy
oc new-build --name=adrs-data --binary --strategy=docker
oc patch bc/adrs-data -p '{"spec":{"resources":{"limits":{"memory":"4Gi","cpu":"1","ephemeral-storage":"20Gi"},"requests":{"memory":"4Gi","cpu":"1","ephemeral-storage":"20Gi"}}}}'
oc start-build adrs-data --from-dir=./src --exclude='\.venv|__pycache__|\.pyc$|^data$' --follow
oc apply -f k8s-openshift.yaml

# 4. Watch pods
oc get pods -w
```

### Useful Commands

```bash
# View logs
oc logs -l app=adrs-data -f

# Check pod status
oc get pods

# Access via port-forward (alternative to route)
oc port-forward svc/adrs-data 8000:8000

# Restart deployment
oc rollout restart deployment/adrs-data
```

### API URL

```
http://adrs-data.leolin2-ns.vpc-int.res.ibm.com
```
