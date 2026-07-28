#!/usr/bin/env python3
"""Q3: Solution Diversity Analytics (Direct Code Embedding Approach).

Research questions:
1. Does exploring diverse algorithmic approaches lead to better outcomes?
2. Does early diversity predict better final outcomes?
3. Are the top-performing candidates diverse, or do they converge to similar approaches?
4. What factors influence whether an iteration improves upon prior results?

This module uses the direct_code_embedding column (768-dim jina-embeddings-v2-base-code)
which embeds the raw solution code directly, compared to q3_diversity_summary_first.py
which uses solution_summary_embedding (LLM-generated summaries embedded with text model).

File organization:
- Shared utilities and database queries
- Tab 1: Diversity vs Score (scatter plot)
- Tab 2: Early Diversity (early iterations vs outcome)
- Tab 3: Top-K Diversity (winners convergence analysis)
- Tab 4: Factor Importance
- High-level API functions
"""

import io
import time
from functools import lru_cache
from typing import Any

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import psycopg
from psycopg.rows import dict_row
from scipy import stats
from sklearn.metrics.pairwise import cosine_similarity


# ===========================================================================
# SHARED: Configuration and Constants
# ===========================================================================

# In-memory cache for expensive queries (cleared on module reload)
_QUERY_CACHE: dict[str, tuple[float, Any]] = {}
_CACHE_TTL_SECONDS = 300  # 5 minute TTL


def _get_cached(key: str) -> Any | None:
    """Get cached value if not expired."""
    if key in _QUERY_CACHE:
        timestamp, value = _QUERY_CACHE[key]
        if time.time() - timestamp < _CACHE_TTL_SECONDS:
            return value
        del _QUERY_CACHE[key]
    return None


def _set_cached(key: str, value: Any) -> None:
    """Store value in cache with current timestamp."""
    _QUERY_CACHE[key] = (time.time(), value)

DATABASE_URL_DEFAULT = "postgresql://postgres:postgres@localhost:5432/adrs"

# Font sizes for scaling: 6" figure → ~2.2" at 0.32\textwidth (37% scale)
# ~16pt base → ~6pt printed (minimum readable)
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 16,
    'axes.titlesize': 18,
    'axes.labelsize': 16,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 14,
    'figure.titlesize': 18,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

COLORS = {
    'high': '#22c55e',  # Green
    'low': '#ef4444',   # Red
}

PALETTE = [
    '#648FFF',  # Blue
    '#785EF0',  # Purple
    '#DC267F',  # Magenta
    '#FE6100',  # Orange
    '#FFB000',  # Gold
    '#000000',  # Black
    '#808080',  # Gray
]

# Extended visual encoding palettes for grouped visualizations
COLOR_PALETTE = [
    '#e41a1c',  # Red
    '#377eb8',  # Blue
    '#4daf4a',  # Green
    '#984ea3',  # Purple
    '#ff7f00',  # Orange
    '#a65628',  # Brown
    '#f781bf',  # Pink
    '#17becf',  # Cyan
    '#bcbd22',  # Olive
    '#1f1f1f',  # Near-black
]

MARKER_PALETTE = ['o', 's', '^', 'D', 'v', 'P', 'X']

from analytics.utils import abbreviate_problem as _abbreviate_problem
from analytics.utils import normalize_model_name
LINESTYLE_PALETTE = ['-', '--', '-.', ':', (0, (3, 1, 1, 1))]

GROUP_BYS = ["algorithm", "model", "model_algorithm"]


# ===========================================================================
# SHARED: Utility Functions
# ===========================================================================

def _get_database_url() -> str:
    import os
    return os.environ.get("DATABASE_URL", DATABASE_URL_DEFAULT)


def get_problems(database_url: str | None = None) -> list[str]:
    """Get list of unique problems that have diversity data.

    Queries the database for all distinct research questions that have
    campaigns with direct_code_embedding data.
    """
    url = database_url or _get_database_url()

    cache_key = f"problems_list:{url}"
    cached = _get_cached(cache_key)
    if cached is not None:
        return cached

    with psycopg.connect(url, row_factory=dict_row) as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT DISTINCT camp.research_question
                FROM campaigns camp
                JOIN candidates c ON c.campaign_id = camp.id
                WHERE c.direct_code_embedding IS NOT NULL
                  AND camp.research_question IS NOT NULL
                ORDER BY camp.research_question
            """)
            rows = cur.fetchall()

    problems = [_truncate_problem(r["research_question"]) for r in rows]
    problems = [p for p in problems if p]  # Filter None

    _set_cached(cache_key, problems)
    return problems


# For backward compatibility with API imports
PROBLEMS = None  # Populated lazily; use get_problems() instead


def _truncate_problem(rq: str | None) -> str | None:
    """Truncate research question to a reasonable display length.

    Also replaces commas with semicolons to avoid breaking comma-delimited
    query parameters in the API.
    """
    if not rq:
        return None
    max_len = 50
    truncated = rq[:max_len] + "..." if len(rq) > max_len else rq
    return truncated.replace(",", ";")


def _parse_embedding(embedding_str: str | None) -> list[float] | None:
    """Parse embedding from PostgreSQL vector string format."""
    if not embedding_str:
        return None
    cleaned = embedding_str.strip("[]")
    return [float(x) for x in cleaned.split(",")]


def _generate_empty_figure(message: str) -> bytes:
    """Generate placeholder figure for empty data."""
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.text(0.5, 0.5, message, ha='center', va='center', fontsize=12)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


# ===========================================================================
# SHARED: Database Queries (using direct_code_embedding)
# ===========================================================================

def query_campaign_diversity_and_scores(database_url: str | None = None) -> list[dict[str, Any]]:
    """Query campaign-level diversity scores and best scores using direct code embeddings.

    Overview:
        For each campaign (a single optimization run), computes how diverse
        the candidate solutions are from each other. A high diversity score
        means the run explored many different algorithmic approaches; a low
        score means candidates converged to similar solutions.

    How diversity is calculated:
        1. Each candidate has a "direct code embedding" - a 768-dim vector
           (from jina-embeddings-v2-base-code) representing the semantic meaning
           of the raw solution code.

        2. For each pair of candidates in a campaign, we compute cosine similarity
           using pgvector's <=> (cosine distance) operator in the database.

        3. Diversity score = 1 - average_similarity = average cosine distance
           Higher diversity (closer to 1) = candidates are dissimilar from each other

    Returns:
        List of dicts with campaign metadata, diversity_score, and best_score.
        Campaigns with <2 embeddings are excluded.

    Results are cached for 5 minutes.
    """
    url = database_url or _get_database_url()

    cache_key = f"campaign_diversity_scores:{url}"
    cached = _get_cached(cache_key)
    if cached is not None:
        return cached

    with psycopg.connect(url, row_factory=dict_row) as conn:
        with conn.cursor() as cur:
            # Compute diversity scores entirely in PostgreSQL using pgvector
            # This avoids loading all embeddings into Python memory
            cur.execute("""
                WITH campaign_scores AS (
                    SELECT
                        c.campaign_id,
                        MAX(m.value) as best_score
                    FROM candidates c
                    JOIN measurements m ON m.candidate_id = c.id AND m.name = 'combined_score'
                    GROUP BY c.campaign_id
                ),
                campaign_diversity AS (
                    SELECT
                        c1.campaign_id,
                        COUNT(*) as n_pairs,
                        -- pgvector <=> is cosine distance = 1 - cosine_similarity
                        -- So diversity = avg(cosine_distance)
                        AVG(c1.direct_code_embedding <=> c2.direct_code_embedding) as diversity_score
                    FROM candidates c1
                    JOIN candidates c2 ON c1.campaign_id = c2.campaign_id AND c1.id < c2.id
                    WHERE c1.direct_code_embedding IS NOT NULL
                      AND c2.direct_code_embedding IS NOT NULL
                    GROUP BY c1.campaign_id
                ),
                campaign_counts AS (
                    SELECT campaign_id, COUNT(*) as n_candidates
                    FROM candidates
                    WHERE direct_code_embedding IS NOT NULL
                    GROUP BY campaign_id
                    HAVING COUNT(*) >= 2
                )
                SELECT
                    camp.id as campaign_id,
                    camp.name as campaign_name,
                    camp.research_question,
                    camp.models_used,
                    camp.algorithm_used,
                    cs.best_score,
                    cd.diversity_score,
                    cc.n_candidates
                FROM campaigns camp
                JOIN campaign_scores cs ON cs.campaign_id = camp.id
                JOIN campaign_diversity cd ON cd.campaign_id = camp.id
                JOIN campaign_counts cc ON cc.campaign_id = camp.id
                WHERE cs.best_score IS NOT NULL
                  AND camp.research_question IS NOT NULL
                ORDER BY camp.id
            """)
            rows = cur.fetchall()

    results = []
    for row in rows:
        problem = _truncate_problem(row["research_question"])
        if problem is None:
            continue

        results.append({
            "campaign_id": str(row["campaign_id"]),
            "campaign_name": row["campaign_name"],
            "problem": problem,
            "research_question": row["research_question"],
            "models_used": row["models_used"],
            "algorithm_used": row["algorithm_used"],
            "n_candidates": row["n_candidates"],
            "diversity_score": round(float(row["diversity_score"]), 4),
            "best_score": float(row["best_score"]),
        })

    _set_cached(cache_key, results)
    return results


def query_campaigns_with_best_embeddings(
    database_url: str | None = None,
) -> list[dict[str, Any]]:
    """Query campaigns with the direct code embedding of their best-scoring candidate.

    For each campaign, finds the candidate with the highest combined_score
    and returns that candidate's embedding along with campaign metadata.

    This is used for computing diversity ACROSS top winners (not within-run diversity).

    Results are cached for 5 minutes.
    """
    url = database_url or _get_database_url()

    cache_key = f"campaigns_best_embeddings:{url}"
    cached = _get_cached(cache_key)
    if cached is not None:
        return cached

    with psycopg.connect(url, row_factory=dict_row) as conn:
        with conn.cursor() as cur:
            cur.execute("""
                WITH best_candidates AS (
                    SELECT DISTINCT ON (c.campaign_id)
                        c.campaign_id,
                        c.id as candidate_id,
                        c.direct_code_embedding::text as embedding,
                        m.value as score
                    FROM candidates c
                    JOIN measurements m ON m.candidate_id = c.id AND m.name = 'combined_score'
                    WHERE c.direct_code_embedding IS NOT NULL
                    ORDER BY c.campaign_id, m.value DESC
                )
                SELECT
                    bc.campaign_id,
                    bc.candidate_id,
                    bc.embedding,
                    bc.score as best_score,
                    camp.research_question,
                    camp.models_used,
                    camp.algorithm_used
                FROM best_candidates bc
                JOIN campaigns camp ON bc.campaign_id = camp.id
                WHERE camp.research_question IS NOT NULL
                ORDER BY bc.campaign_id
            """)
            rows = cur.fetchall()

    results = []
    for row in rows:
        emb = _parse_embedding(row["embedding"])
        if emb:
            problem = _truncate_problem(row["research_question"])
            if problem is None:
                continue

            results.append({
                "campaign_id": str(row["campaign_id"]),
                "problem": problem,
                "models_used": row["models_used"],
                "algorithm_used": row["algorithm_used"],
                "best_score": float(row["best_score"]),
                "best_embedding": emb,
            })

    _set_cached(cache_key, results)
    return results


# ===========================================================================
# TAB 1: Diversity vs Score
# ===========================================================================

def process_diversity_vs_score_scatter(
    campaigns: list[dict[str, Any]],
    group_by: str | None = None,
    problems: list[str] | None = None,
) -> dict[str, Any]:
    """Process campaign data for diversity vs score scatter plot."""
    if problems:
        campaigns = [c for c in campaigns if c.get("problem") in problems]

    points = []
    for camp in campaigns:
        if group_by == "model":
            group = ", ".join(sorted(normalize_model_name(m) for m in camp["models_used"])) if camp.get("models_used") else "unknown"
        elif group_by == "algorithm":
            group = camp.get("algorithm_used") or "unknown"
        elif group_by == "model_algorithm":
            model = ", ".join(sorted(normalize_model_name(m) for m in camp["models_used"])) if camp.get("models_used") else "unknown"
            alg = camp.get("algorithm_used") or "unknown"
            group = f"{model} + {alg}"
        else:
            group = "all"

        points.append({
            "campaign_id": camp["campaign_id"],
            "problem": camp["problem"],
            "diversity": camp["diversity_score"],
            "score": camp["best_score"],
            "group": group,
        })

    if not points:
        return {
            "group_by": group_by,
            "problems": problems,
            "points": [],
            "group_stats": [],
            "overall_correlation": None,
        }

    all_div = [p["diversity"] for p in points]
    all_score = [p["score"] for p in points]
    if len(points) >= 4:
        overall_rho, _ = stats.spearmanr(all_div, all_score)
        overall_corr = float(overall_rho)
    else:
        overall_corr = None

    by_group: dict[str, list[dict]] = {}
    for p in points:
        g = p["group"]
        if g not in by_group:
            by_group[g] = []
        by_group[g].append(p)

    group_stats = []
    for group_name in sorted(by_group.keys()):
        group_points = by_group[group_name]
        if len(group_points) < 3:
            continue

        divs = [p["diversity"] for p in group_points]
        scores = [p["score"] for p in group_points]
        if len(group_points) >= 4:
            rho, p_val = stats.spearmanr(divs, scores)
        else:
            rho, p_val = None, None

        group_stats.append({
            "group": group_name,
            "n": len(group_points),
            "correlation": round(rho, 4) if rho is not None and not np.isnan(rho) else None,
            "spearman_rho": round(rho, 4) if rho is not None and not np.isnan(rho) else None,
            "spearman_p": round(p_val, 4) if p_val is not None and not np.isnan(p_val) else None,
            "diversity_mean": round(float(np.mean(divs)), 4),
            "score_mean": round(float(np.mean(scores)), 4),
        })

    return {
        "group_by": group_by,
        "problems": problems,
        "points": points,
        "group_stats": group_stats,
        "overall_correlation": round(overall_corr, 4) if overall_corr and not np.isnan(overall_corr) else None,
        "total_campaigns": len(points),
    }


def generate_diversity_vs_score_scatter_figure(
    data: dict[str, Any],
) -> bytes:
    """Generate scatter plot of diversity vs score with optional grouping."""
    points = data.get("points", [])
    if not points:
        return _generate_empty_figure("No diversity data available")

    group_by = data.get("group_by")
    group_stats = data.get("group_stats", [])
    problems = data.get("problems")

    if group_by is None:
        fig, ax = plt.subplots(figsize=(10, 7))
        x = [p["diversity"] for p in points]
        y = [p["score"] for p in points]

        ax.scatter(x, y, c=COLOR_PALETTE[0], alpha=0.6, s=40, edgecolors='white', linewidths=0.5)

        if len(x) >= 2:
            z = np.polyfit(x, y, 1)
            p_fit = np.poly1d(z)
            x_line = np.linspace(min(x), max(x), 100)
            ax.plot(x_line, p_fit(x_line), '--', color=COLOR_PALETTE[0], alpha=0.8, linewidth=2)

        # corr = data.get("overall_correlation")
        # corr_str = f"r = {corr:.3f}" if corr is not None else "r = N/A"
        # ax.set_title(f'Code Diversity vs Final Score (All Runs)\n{corr_str}, n={len(points)}',
        #              fontweight='bold')
        ax.set_xlabel('Run Diversity (1 - avg cosine similarity of code embeddings)')
        ax.set_ylabel('Best Score')
        ax.grid(True, alpha=0.3)

        # if problems:
        #     ax.set_title(ax.get_title() + f'\nProblems: {", ".join(problems)}', fontsize=10)

    elif group_by == "model_algorithm":
        algorithms = sorted(set(g["group"].rsplit(" + ", 1)[1] for g in group_stats if " + " in g["group"]))
        models = sorted(set(g["group"].rsplit(" + ", 1)[0] for g in group_stats if " + " in g["group"]))

        if not algorithms:
            return _generate_empty_figure("No model+algorithm groups found")

        n_algs = len(algorithms)
        n_models = len(models)
        fig_width = min(20, max(10, 5 * n_algs))
        fig, axes = plt.subplots(1, n_algs, figsize=(fig_width, 6), squeeze=False, sharey=True)
        axes = axes[0]

        if n_models <= 10:
            model_colors = {m: plt.cm.tab10(i / 10) for i, m in enumerate(models)}
        elif n_models <= 20:
            model_colors = {m: plt.cm.tab20(i / 20) for i, m in enumerate(models)}
        else:
            model_colors = {m: plt.cm.viridis(0.1 + 0.8 * i / n_models) for i, m in enumerate(models)}

        model_styles = {m: (LINESTYLE_PALETTE[i % len(LINESTYLE_PALETTE)],
                           MARKER_PALETTE[i % len(MARKER_PALETTE)])
                       for i, m in enumerate(models)}

        for ax_idx, algorithm in enumerate(algorithms):
            ax = axes[ax_idx]

            for model in models:
                group_name = f"{model} + {algorithm}"
                gs = next((g for g in group_stats if g["group"] == group_name), None)
                if not gs:
                    continue

                group_points = [p for p in points if p["group"] == group_name]
                if not group_points:
                    continue

                x = [p["diversity"] for p in group_points]
                y = [p["score"] for p in group_points]
                color = model_colors[model]
                linestyle, marker = model_styles[model]
                short_name = model[:15] + "..." if len(model) > 15 else model

                ax.scatter(x, y, c=[color], marker=marker, alpha=0.6, s=40,
                          edgecolors='white', linewidths=0.5,
                          label=short_name if ax_idx == 0 else None)

                if len(x) >= 2:
                    z = np.polyfit(x, y, 1)
                    p_fit = np.poly1d(z)
                    x_line = np.linspace(min(x), max(x), 100)
                    ax.plot(x_line, p_fit(x_line), linestyle=linestyle, color=color,
                           alpha=0.7, linewidth=1.8)

            ax.set_xlabel('Run Diversity')
            if ax_idx == 0:
                ax.set_ylabel('Best Score')
            ax.grid(True, alpha=0.3)
            # if n_algs > 1:
            #     ax.set_title(algorithm, fontweight='bold', fontsize=11)

        # overall_corr = data.get("overall_correlation")
        # corr_str = f"Overall r = {overall_corr:.3f}" if overall_corr is not None else ""
        # prob_str = f' | {", ".join(problems)}' if problems else ""
        # if n_algs == 1:
        #     alg_suffix = f' ({algorithms[0]})'
        # else:
        #     alg_suffix = ''
        # fig.suptitle(f'Code Diversity vs Score by Model + Algorithm{alg_suffix}\n{corr_str}, n={len(points)}{prob_str}',
        #              fontweight='bold', fontsize=12, y=1.02)

        plt.tight_layout()

        handles, labels = axes[0].get_legend_handles_labels()
        if handles:
            n_legend_cols = min(6, max(1, n_models))
            fig.legend(handles, labels, loc='upper center', fontsize=9,
                      ncol=n_legend_cols, bbox_to_anchor=(0.5, -0.02))

    else:
        groups = sorted(gs["group"] for gs in group_stats)
        n_groups = len(groups)

        fig_width = max(10, min(16, 2.5 * max(n_groups // 3, 4)))
        fig, ax = plt.subplots(figsize=(fig_width, 7))

        if n_groups <= 10:
            colors = [plt.cm.tab10(i / 10) for i in range(n_groups)]
        elif n_groups <= 20:
            colors = [plt.cm.tab20(i / 20) for i in range(n_groups)]
        else:
            colors = [plt.cm.viridis(0.1 + 0.8 * i / n_groups) for i in range(n_groups)]

        for i, gs in enumerate(group_stats):
            group_name = gs["group"]
            group_points = [p for p in points if p["group"] == group_name]
            if not group_points:
                continue

            x = [p["diversity"] for p in group_points]
            y = [p["score"] for p in group_points]

            color = colors[i]
            marker = MARKER_PALETTE[i % len(MARKER_PALETTE)]
            linestyle = LINESTYLE_PALETTE[i % len(LINESTYLE_PALETTE)]

            label = group_name
            if len(label) > 20:
                label = label[:18] + "..."
            rho = gs.get("spearman_rho")
            p_val = gs.get("spearman_p")
            if rho is not None:
                if p_val is not None and p_val < 0.001:
                    label += f" (ρ={rho:.2f}, p<.001)"
                elif p_val is not None:
                    label += f" (ρ={rho:.2f}, p={p_val:.3f})"
                else:
                    label += f" (ρ={rho:.2f})"

            ax.scatter(x, y, c=[color], marker=marker, alpha=0.6, s=40,
                      edgecolors='white', linewidths=0.5, label=label)

            if len(x) >= 2:
                z = np.polyfit(x, y, 1)
                p_fit = np.poly1d(z)
                x_line = np.linspace(min(x), max(x), 100)
                ax.plot(x_line, p_fit(x_line), linestyle=linestyle, color=color,
                       alpha=0.7, linewidth=1.8)

        ax.set_xlabel('Run Diversity (1 - avg cosine similarity of code embeddings)')
        ax.set_ylabel('Best Score')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=10, framealpha=0.9)

    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=300)
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


# ===========================================================================
# TAB 2: Early Diversity
# ===========================================================================

def _compute_early_diversity_campaign_data(
    candidates: list[dict[str, Any]] | None = None,
    early_fraction: float = 0.25,
    database_url: str | None = None,
) -> list[dict[str, Any]]:
    """Compute early diversity for each campaign using direct code embeddings.

    Computes diversity in SQL using pgvector to avoid loading all embeddings into memory.
    """
    url = database_url or _get_database_url()

    with psycopg.connect(url, row_factory=dict_row) as conn:
        with conn.cursor() as cur:
            # Compute early diversity entirely in SQL
            # early_fraction = 0.25 means first 25% of iterations
            cur.execute("""
                WITH campaign_stats AS (
                    -- Get max iteration and candidate count per campaign
                    SELECT
                        c.campaign_id,
                        MAX(c.iteration_index) as max_iter,
                        COUNT(*) as n_candidates
                    FROM candidates c
                    WHERE c.direct_code_embedding IS NOT NULL
                      AND c.iteration_index IS NOT NULL
                    GROUP BY c.campaign_id
                    HAVING COUNT(*) >= 10
                ),
                early_candidates AS (
                    -- Select candidates in first 25%% of iterations
                    SELECT
                        c.id,
                        c.campaign_id,
                        c.iteration_index,
                        c.direct_code_embedding
                    FROM candidates c
                    JOIN campaign_stats cs ON c.campaign_id = cs.campaign_id
                    WHERE c.direct_code_embedding IS NOT NULL
                      AND c.iteration_index IS NOT NULL
                      AND c.iteration_index <= (cs.max_iter * %(early_fraction)s)::int
                ),
                early_diversity AS (
                    -- Compute pairwise diversity among early candidates
                    SELECT
                        e1.campaign_id,
                        COUNT(*) as n_pairs,
                        AVG(e1.direct_code_embedding <=> e2.direct_code_embedding) as early_diversity
                    FROM early_candidates e1
                    JOIN early_candidates e2 ON e1.campaign_id = e2.campaign_id AND e1.id < e2.id
                    GROUP BY e1.campaign_id
                ),
                early_counts AS (
                    SELECT campaign_id, COUNT(*) as n_early
                    FROM early_candidates
                    GROUP BY campaign_id
                    HAVING COUNT(*) >= 2
                ),
                campaign_scores AS (
                    SELECT
                        c.campaign_id,
                        MAX(m.value) as best_score
                    FROM candidates c
                    JOIN measurements m ON m.candidate_id = c.id AND m.name = 'combined_score'
                    GROUP BY c.campaign_id
                )
                SELECT
                    camp.id as campaign_id,
                    camp.research_question,
                    camp.models_used,
                    camp.algorithm_used,
                    ed.early_diversity,
                    ec.n_early as n_early_candidates,
                    cs_stats.n_candidates as n_total_candidates,
                    cs.best_score
                FROM campaigns camp
                JOIN campaign_stats cs_stats ON cs_stats.campaign_id = camp.id
                JOIN early_diversity ed ON ed.campaign_id = camp.id
                JOIN early_counts ec ON ec.campaign_id = camp.id
                JOIN campaign_scores cs ON cs.campaign_id = camp.id
                WHERE camp.research_question IS NOT NULL
                  AND cs.best_score IS NOT NULL
                ORDER BY camp.id
            """, {"early_fraction": early_fraction})
            rows = cur.fetchall()

    campaign_data = []
    for row in rows:
        problem = _truncate_problem(row["research_question"])
        if problem is None:
            continue

        campaign_data.append({
            "campaign_id": str(row["campaign_id"]),
            "problem": problem,
            "model": normalize_model_name(row["models_used"][0]) if row["models_used"] else "unknown",
            "algorithm": row["algorithm_used"] or "unknown",
            "early_diversity": round(float(row["early_diversity"]), 4),
            "best_score": round(float(row["best_score"]), 4),
            "n_early_candidates": row["n_early_candidates"],
            "n_total_candidates": row["n_total_candidates"],
        })

    return campaign_data


def process_early_diversity_vs_outcome(
    early_fraction: float = 0.25,
    database_url: str | None = None,
) -> dict[str, Any]:
    """Analyze whether diversity in early iterations predicts final outcome."""
    campaign_data = _compute_early_diversity_campaign_data(
        early_fraction=early_fraction, database_url=database_url
    )

    by_problem: dict[str, list[dict]] = {}
    for camp in campaign_data:
        prob = camp["problem"]
        if prob not in by_problem:
            by_problem[prob] = []
        by_problem[prob].append(camp)

    problem_results = []
    for problem in sorted(by_problem.keys()):
        camps = by_problem[problem]
        if len(camps) < 4:
            continue

        early_divs = [c["early_diversity"] for c in camps]
        best_scores = [c["best_score"] for c in camps]

        rho, p_val = stats.spearmanr(early_divs, best_scores)

        problem_results.append({
            "problem": problem,
            "n_campaigns": len(camps),
            "correlation": round(rho, 4) if not np.isnan(rho) else None,
            "spearman_rho": round(rho, 4) if not np.isnan(rho) else None,
            "spearman_p": round(p_val, 4) if not np.isnan(p_val) else None,
            "early_diversity_mean": round(float(np.mean(early_divs)), 4),
            "early_diversity_std": round(float(np.std(early_divs)), 4),
            "best_score_mean": round(float(np.mean(best_scores)), 4),
            "best_score_std": round(float(np.std(best_scores)), 4),
            "campaigns": camps,
        })

    all_early_divs = [c["early_diversity"] for c in campaign_data]
    all_best_scores = [c["best_score"] for c in campaign_data]
    if len(campaign_data) >= 4:
        overall_rho, _ = stats.spearmanr(all_early_divs, all_best_scores)
        overall_corr = float(overall_rho)
    else:
        overall_corr = None

    return {
        "problems": problem_results,
        "early_fraction": early_fraction,
        "summary": {
            "total_problems": len(problem_results),
            "total_campaigns": len(campaign_data),
            "overall_correlation": round(overall_corr, 4) if overall_corr and not np.isnan(overall_corr) else None,
            "problems_with_positive_correlation": len([p for p in problem_results if p["correlation"] and p["correlation"] > 0]),
            "problems_with_negative_correlation": len([p for p in problem_results if p["correlation"] and p["correlation"] < 0]),
        },
    }


def process_early_diversity_scatter(
    early_fraction: float = 0.25,
    group_by: str | None = None,
    problems: list[str] | None = None,
    database_url: str | None = None,
) -> dict[str, Any]:
    """Process early diversity data for scatter plot with grouping."""
    campaign_data = _compute_early_diversity_campaign_data(
        early_fraction=early_fraction, database_url=database_url
    )

    if problems:
        campaign_data = [c for c in campaign_data if c.get("problem") in problems]

    points = []
    for camp in campaign_data:
        if group_by == "model":
            group = camp.get("model") or "unknown"
        elif group_by == "algorithm":
            group = camp.get("algorithm") or "unknown"
        elif group_by == "model_algorithm":
            model = camp.get("model") or "unknown"
            alg = camp.get("algorithm") or "unknown"
            group = f"{model} + {alg}"
        else:
            group = "all"

        points.append({
            "campaign_id": camp["campaign_id"],
            "problem": camp["problem"],
            "early_diversity": camp["early_diversity"],
            "best_score": camp["best_score"],
            "group": group,
        })

    if not points:
        return {
            "group_by": group_by,
            "problems": problems,
            "early_fraction": early_fraction,
            "points": [],
            "group_stats": [],
            "overall_correlation": None,
        }

    all_div = [p["early_diversity"] for p in points]
    all_score = [p["best_score"] for p in points]
    if len(points) >= 4:
        overall_rho, _ = stats.spearmanr(all_div, all_score)
        overall_corr = float(overall_rho)
    else:
        overall_corr = None

    by_group: dict[str, list[dict]] = {}
    for p in points:
        g = p["group"]
        if g not in by_group:
            by_group[g] = []
        by_group[g].append(p)

    group_stats = []
    for group_name in sorted(by_group.keys()):
        group_points = by_group[group_name]
        if len(group_points) < 3:
            continue

        divs = [p["early_diversity"] for p in group_points]
        scores = [p["best_score"] for p in group_points]
        if len(group_points) >= 4:
            rho, p_val = stats.spearmanr(divs, scores)
        else:
            rho, p_val = None, None

        group_stats.append({
            "group": group_name,
            "n": len(group_points),
            "correlation": round(rho, 4) if rho is not None and not np.isnan(rho) else None,
            "spearman_rho": round(rho, 4) if rho is not None and not np.isnan(rho) else None,
            "spearman_p": round(p_val, 4) if p_val is not None and not np.isnan(p_val) else None,
            "early_diversity_mean": round(float(np.mean(divs)), 4),
            "score_mean": round(float(np.mean(scores)), 4),
        })

    return {
        "group_by": group_by,
        "problems": problems,
        "early_fraction": early_fraction,
        "points": points,
        "group_stats": group_stats,
        "overall_correlation": round(overall_corr, 4) if overall_corr and not np.isnan(overall_corr) else None,
        "total_campaigns": len(points),
    }


def generate_early_diversity_scatter_figure(data: dict[str, Any]) -> bytes:
    """Generate scatter plot of early diversity vs score with optional grouping."""
    points = data.get("points", [])
    if not points:
        return _generate_empty_figure("No early diversity data available")

    group_by = data.get("group_by")
    group_stats = data.get("group_stats", [])
    early_fraction = data.get("early_fraction", 0.25)
    problems = data.get("problems")

    if group_by is None:
        fig, ax = plt.subplots(figsize=(6, 4.5))
        x = [p["early_diversity"] for p in points]
        y = [p["best_score"] for p in points]

        ax.scatter(x, y, c=COLOR_PALETTE[0], alpha=0.6, s=50, edgecolors='white', linewidths=0.5)

        if len(x) >= 2:
            z = np.polyfit(x, y, 1)
            p_fit = np.poly1d(z)
            x_line = np.linspace(min(x), max(x), 100)
            ax.plot(x_line, p_fit(x_line), '--', color=COLOR_PALETTE[0], alpha=0.8, linewidth=2)

            rho, p_value = stats.spearmanr(x, y)
            if p_value < 0.001:
                p_str = "p < .001"
            else:
                p_str = f"p = {p_value:.3f}"
            ax.set_title(f"ρ = {rho:.2f}, {p_str}", fontsize=14)

        ax.set_xlabel(f'Early Diversity (first {int(early_fraction * 100)}%)')
        ax.set_ylabel('Best Score')
        ax.grid(True, alpha=0.3)

        # if problems:
        #     ax.set_title(ax.get_title() + f'\nProblems: {", ".join(problems)}', fontsize=10)

    elif group_by == "model_algorithm":
        algorithms = sorted(set(g["group"].rsplit(" + ", 1)[1] for g in group_stats if " + " in g["group"]))
        models = sorted(set(g["group"].rsplit(" + ", 1)[0] for g in group_stats if " + " in g["group"]))

        if not algorithms:
            return _generate_empty_figure("No model+algorithm groups found")

        n_algs = len(algorithms)
        n_models = len(models)
        fig_width = min(20, max(10, 5 * n_algs))
        fig, axes = plt.subplots(1, n_algs, figsize=(fig_width, 6), squeeze=False, sharey=True)
        axes = axes[0]

        if n_models <= 10:
            model_colors = {m: plt.cm.tab10(i / 10) for i, m in enumerate(models)}
        elif n_models <= 20:
            model_colors = {m: plt.cm.tab20(i / 20) for i, m in enumerate(models)}
        else:
            model_colors = {m: plt.cm.viridis(0.1 + 0.8 * i / n_models) for i, m in enumerate(models)}

        model_styles = {m: (LINESTYLE_PALETTE[i % len(LINESTYLE_PALETTE)],
                           MARKER_PALETTE[i % len(MARKER_PALETTE)])
                       for i, m in enumerate(models)}

        for ax_idx, algorithm in enumerate(algorithms):
            ax = axes[ax_idx]

            for model in models:
                group_name = f"{model} + {algorithm}"
                gs = next((g for g in group_stats if g["group"] == group_name), None)
                if not gs:
                    continue

                group_points = [p for p in points if p["group"] == group_name]
                if not group_points:
                    continue

                x = [p["early_diversity"] for p in group_points]
                y = [p["best_score"] for p in group_points]
                color = model_colors[model]
                linestyle, marker = model_styles[model]
                short_name = model[:15] + "..." if len(model) > 15 else model

                ax.scatter(x, y, c=[color], marker=marker, alpha=0.6, s=40,
                          edgecolors='white', linewidths=0.5,
                          label=short_name if ax_idx == 0 else None)

                if len(x) >= 2:
                    z = np.polyfit(x, y, 1)
                    p_fit = np.poly1d(z)
                    x_line = np.linspace(min(x), max(x), 100)
                    ax.plot(x_line, p_fit(x_line), linestyle=linestyle, color=color,
                           alpha=0.7, linewidth=1.8)

            ax.set_xlabel(f'Early Diversity (first {int(early_fraction * 100)}%)')
            if ax_idx == 0:
                ax.set_ylabel('Best Score')
            ax.grid(True, alpha=0.3)
            # if n_algs > 1:
            #     ax.set_title(algorithm, fontweight='bold', fontsize=11)

        # overall_corr = data.get("overall_correlation")
        # corr_str = f"Overall r = {overall_corr:.3f}" if overall_corr is not None else ""
        # prob_str = f' | {", ".join(problems)}' if problems else ""
        # if n_algs == 1:
        #     alg_suffix = f' ({algorithms[0]})'
        # else:
        #     alg_suffix = ''
        # fig.suptitle(f'Early Code Diversity vs Score by Model + Algorithm{alg_suffix}\n{corr_str}, n={len(points)}{prob_str}',
        #              fontweight='bold', fontsize=12, y=1.02)

        plt.tight_layout()

        handles, labels = axes[0].get_legend_handles_labels()
        if handles:
            n_legend_cols = min(6, max(1, n_models))
            fig.legend(handles, labels, loc='upper center', fontsize=9,
                      ncol=n_legend_cols, bbox_to_anchor=(0.5, -0.02))

    else:
        groups = sorted(gs["group"] for gs in group_stats)
        n_groups = len(groups)

        fig_width = max(10, min(16, 2.5 * max(n_groups // 3, 4)))
        fig, ax = plt.subplots(figsize=(fig_width, 7))

        if n_groups <= 10:
            colors = [plt.cm.tab10(i / 10) for i in range(n_groups)]
        elif n_groups <= 20:
            colors = [plt.cm.tab20(i / 20) for i in range(n_groups)]
        else:
            colors = [plt.cm.viridis(0.1 + 0.8 * i / n_groups) for i in range(n_groups)]

        for i, gs in enumerate(group_stats):
            group_name = gs["group"]
            group_points = [p for p in points if p["group"] == group_name]
            if not group_points:
                continue

            x = [p["early_diversity"] for p in group_points]
            y = [p["best_score"] for p in group_points]

            color = colors[i]
            marker = MARKER_PALETTE[i % len(MARKER_PALETTE)]
            linestyle = LINESTYLE_PALETTE[i % len(LINESTYLE_PALETTE)]

            label = group_name
            if len(label) > 20:
                label = label[:18] + "..."
            rho = gs.get("spearman_rho")
            p_val = gs.get("spearman_p")
            if rho is not None:
                if p_val is not None and p_val < 0.001:
                    label += f" (ρ={rho:.2f}, p<.001)"
                elif p_val is not None:
                    label += f" (ρ={rho:.2f}, p={p_val:.3f})"
                else:
                    label += f" (ρ={rho:.2f})"

            ax.scatter(x, y, c=[color], marker=marker, alpha=0.6, s=40,
                      edgecolors='white', linewidths=0.5, label=label)

            if len(x) >= 2:
                z = np.polyfit(x, y, 1)
                p_fit = np.poly1d(z)
                x_line = np.linspace(min(x), max(x), 100)
                ax.plot(x_line, p_fit(x_line), linestyle=linestyle, color=color,
                       alpha=0.7, linewidth=1.8)

        ax.set_xlabel(f'Early Diversity (first {int(early_fraction * 100)}% of iterations)')
        ax.set_ylabel('Best Score')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=10, framealpha=0.9)

    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=300)
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


# ===========================================================================
# TAB 3: Top-K Diversity
# ===========================================================================

def _compute_cross_diversity(embeddings: list[list[float]]) -> float:
    """Compute pairwise diversity among a set of embeddings."""
    if len(embeddings) < 2:
        return 0.0

    emb_array = np.array(embeddings)
    sim_matrix = cosine_similarity(emb_array)
    np.fill_diagonal(sim_matrix, 0)
    n = len(embeddings)
    avg_similarity = sim_matrix.sum() / (n * (n - 1))
    return 1 - avg_similarity


def process_topk_winners_diversity(
    campaigns: list[dict[str, Any]],
    top_pct: float = 0.25,
    group_by: str | None = None,
    problems_filter: list[str] | None = None,
) -> dict[str, Any]:
    """Compute diversity ACROSS top winners vs other winners using direct code embeddings.

    Args:
        campaigns: List of campaign data with best_embedding
        top_pct: Fraction of campaigns to consider "top" (default 0.25 = top 25%)
        group_by: How to group campaigns (model, algorithm, model_algorithm, or None for by problem)
        problems_filter: Optional list of problems to include

    Groups are excluded if either the top or other bucket has fewer than 2 campaigns,
    since we need at least 2 to compute pairwise diversity.
    """
    if problems_filter:
        campaigns = [c for c in campaigns if c["problem"] in problems_filter]

    campaigns = [c for c in campaigns if c["best_score"] > 0]

    if not campaigns:
        return {"groups": [], "summary": {"total_groups": 0, "excluded_groups": 0}}

    def get_group_key(camp: dict) -> str:
        if group_by == "model":
            models = camp.get("models_used")
            return ", ".join(sorted(normalize_model_name(m) for m in models)) if models else "Unknown"
        elif group_by == "algorithm":
            return camp.get("algorithm_used") or "Unknown"
        elif group_by == "model_algorithm":
            models = camp.get("models_used")
            model = ", ".join(sorted(normalize_model_name(m) for m in models)) if models else "Unknown"
            algo = camp.get("algorithm_used") or "Unknown"
            return f"{model} + {algo}"
        else:
            return camp["problem"]

    by_group: dict[str, list[dict]] = {}
    for camp in campaigns:
        key = get_group_key(camp)
        if key not in by_group:
            by_group[key] = []
        by_group[key].append(camp)

    groups = []
    excluded_count = 0
    for group_name, group_camps in sorted(by_group.items()):
        sorted_camps = sorted(group_camps, key=lambda x: -x["best_score"])

        # Use percentage-based split
        n_top = max(1, int(len(sorted_camps) * top_pct))
        top_runs = sorted_camps[:n_top]
        other_runs = sorted_camps[n_top:]

        # Require at least 2 in each bucket to compute diversity
        if len(top_runs) < 2 or len(other_runs) < 2:
            excluded_count += 1
            continue

        top_embeddings = [c["best_embedding"] for c in top_runs]
        top_diversity = _compute_cross_diversity(top_embeddings)

        other_embeddings = [c["best_embedding"] for c in other_runs]
        other_diversity = _compute_cross_diversity(other_embeddings)

        top_best_scores = [c["best_score"] for c in top_runs]
        other_best_scores = [c["best_score"] for c in other_runs]

        groups.append({
            "group": group_name,
            "top_n": len(top_runs),
            "other_n": len(other_runs),
            "total_runs": len(sorted_camps),
            "top_winners_diversity": round(top_diversity, 4),
            "other_winners_diversity": round(other_diversity, 4),
            "diversity_diff": round(top_diversity - other_diversity, 4),
            "top_score_mean": round(float(np.mean(top_best_scores)), 4),
            "top_score_range": [round(min(top_best_scores), 4), round(max(top_best_scores), 4)],
            "other_score_mean": round(float(np.mean(other_best_scores)), 4),
        })

    return {
        "groups": groups,
        "group_by": group_by,
        "top_pct": top_pct,
        "problems_filter": problems_filter,
        "summary": {
            "total_groups": len(groups),
            "excluded_groups": excluded_count,
            "groups_where_top_more_diverse": len([g for g in groups if g["diversity_diff"] > 0.01]),
            "groups_where_top_less_diverse": len([g for g in groups if g["diversity_diff"] < -0.01]),
        },
    }


def generate_topk_winners_diversity_figure(
    data: dict[str, Any],
) -> bytes:
    """Generate bar chart comparing diversity across top winners vs other winners."""
    groups = data.get("groups", [])
    if not groups:
        excluded = data.get("summary", {}).get("excluded_groups", 0)
        msg = "No groups with sufficient data"
        if excluded > 0:
            msg += f"\n({excluded} groups excluded: need ≥2 in both top and other buckets)"
        return _generate_empty_figure(msg)

    group_by = data.get("group_by")
    top_pct = data.get("top_pct", 0.25)
    top_pct_str = f"{int(top_pct * 100)}%"

    if group_by == "model_algorithm":
        by_algorithm: dict[str, dict[str, dict]] = {}
        for g in groups:
            group_name = g["group"]
            if " + " in group_name:
                model, algorithm = group_name.rsplit(" + ", 1)
            else:
                model, algorithm = group_name, "unknown"
            if algorithm not in by_algorithm:
                by_algorithm[algorithm] = {}
            by_algorithm[algorithm][model] = g

        algorithms = sorted(by_algorithm.keys())
        n_algs = len(algorithms)

        if n_algs == 0:
            return _generate_empty_figure("No model+algorithm groups found")

        all_models = sorted(set(
            model for alg_data in by_algorithm.values() for model in alg_data.keys()
        ))
        n_models = len(all_models)

        max_model_len = max(len(m) for m in all_models) if all_models else 10

        width_per_alg = max(4, min(8, 1.5 * max(len(by_algorithm[alg]) for alg in algorithms)))
        fig_width = min(24, max(10, width_per_alg * n_algs))
        fig_height = 6 + (max_model_len * 0.05)
        fig, axes = plt.subplots(1, n_algs, figsize=(fig_width, fig_height), squeeze=False, sharey=True)
        axes = axes[0]

        if n_models <= 10:
            model_colors = {m: plt.cm.tab10(i / 10) for i, m in enumerate(all_models)}
        elif n_models <= 20:
            model_colors = {m: plt.cm.tab20(i / 20) for i, m in enumerate(all_models)}
        else:
            model_colors = {m: plt.cm.viridis(0.1 + 0.8 * i / n_models) for i, m in enumerate(all_models)}

        for ax_idx, algorithm in enumerate(algorithms):
            ax = axes[ax_idx]
            alg_data = by_algorithm[algorithm]

            models_in_alg = sorted(alg_data.keys())
            n_models_in_alg = len(models_in_alg)

            if n_models_in_alg == 0:
                ax.axis("off")
                continue

            x = np.arange(n_models_in_alg)
            width = 0.35

            top_diversity = [alg_data[m]["top_winners_diversity"] for m in models_in_alg]
            other_diversity = [alg_data[m]["other_winners_diversity"] for m in models_in_alg]

            bar_colors = [model_colors[m] for m in models_in_alg]

            ax.bar(x - width/2, top_diversity, width,
                  color=bar_colors, alpha=0.9, edgecolor='white', linewidth=0.5)
            ax.bar(x + width/2, other_diversity, width,
                  color=bar_colors, alpha=0.4, edgecolor='white', linewidth=0.5)

            ax.set_xticks(x)
            ax.set_xticklabels(models_in_alg, rotation=45, ha='right', fontsize=8)
            ax.grid(True, alpha=0.3, axis='y')
            # ax.set_title(algorithm, fontweight='bold', fontsize=11)

            if ax_idx == 0:
                ax.set_ylabel('Best-Candidate Code Diversity')

        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='gray', alpha=0.9, label=f'Top {top_pct_str} Runs'),
            Patch(facecolor='gray', alpha=0.4, label='Other Runs'),
        ]
        fig.legend(handles=legend_elements, loc='upper center', ncol=2, fontsize=9,
                  bbox_to_anchor=(0.5, 0.02))

        # problems_filter = data.get("problems_filter")
        # prob_str = f' | {", ".join(problems_filter)}' if problems_filter else ""
        # fig.suptitle(f'Cross-Winner Code Diversity: Top {top_pct_str} vs Other by Model + Algorithm{prob_str}',
        #             fontweight='bold', fontsize=12, y=1.02)

        plt.tight_layout(rect=[0, 0.05, 1, 0.98])

    else:
        has_grouping = group_by is not None

        group_names = [g["group"] for g in groups]
        max_label_len = max(len(name) for name in group_names) if group_names else 10
        fig_width = max(10, min(16, 1.5 * len(groups) + max_label_len * 0.1))
        fig_height = 7 + (max_label_len * 0.03)
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))

        top_diversity = [g["top_winners_diversity"] for g in groups]
        other_diversity = [g["other_winners_diversity"] for g in groups]

        x = np.arange(len(group_names))
        width = 0.35

        ax.bar(x - width/2, top_diversity, width, label=f'Top {top_pct_str} Runs',
               color=PALETTE[0], alpha=0.8)
        ax.bar(x + width/2, other_diversity, width, label='Other Runs',
               color=PALETTE[1], alpha=0.8)

        ax.set_ylabel('Best-Candidate Code Diversity')
        ax.set_xticks(x)
        ax.set_xticklabels([_abbreviate_problem(n) for n in group_names], rotation=0, ha='center', fontsize=16)
        ax.grid(True, alpha=0.3, axis='y')

        # if group_by:
        #     group_label = {"model": "Model", "algorithm": "Algorithm"}[group_by]
        #     title = f'Cross-Winner Code Diversity: Top {top_pct_str} vs Other\nGrouped by {group_label}'
        # else:
        #     problems_filter = data.get("problems_filter")
        #     if problems_filter and len(problems_filter) < 3:
        #         title = f'Cross-Winner Code Diversity: Top {top_pct_str} vs Other\n{", ".join(problems_filter)}'
        #     else:
        #         title = f'Cross-Winner Code Diversity: Top {top_pct_str} vs Other\nBy Problem'

        # ax.set_title(title, fontweight='bold')

        ax.legend(loc='upper right', fontsize=11, framealpha=0.9)

        plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=300)
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


# ===========================================================================
# TAB 4: Factor Importance
# ===========================================================================

def query_mutation_factors(
    database_url: str | None = None,
) -> list[dict[str, Any]]:
    """Query mutation data with all factors needed for factor analysis using direct code embeddings.

    Computes context diversity in SQL using pgvector to avoid loading all embeddings into memory.

    Results are cached for 5 minutes since mutation data changes infrequently.
    """
    url = database_url or _get_database_url()

    cache_key = f"mutation_factors:{url}"
    cached = _get_cached(cache_key)
    if cached is not None:
        return cached

    with psycopg.connect(url, row_factory=dict_row) as conn:
        with conn.cursor() as cur:
            # Query mutation factors, reading precomputed context_code_diversity from candidates table
            cur.execute("""
                WITH parent_edges AS (
                    SELECT target_candidate_id as child_id, source_candidate_id as parent_id
                    FROM candidate_edges
                    WHERE edge_type = 'parent'
                ),
                context_edges AS (
                    SELECT target_candidate_id as child_id, source_candidate_id as context_id
                    FROM candidate_edges
                    WHERE edge_type = 'context'
                ),
                -- Context stats: count, max score, variance, better/worse counts relative to parent
                context_stats AS (
                    SELECT
                        ce.child_id,
                        COUNT(ctx_m.value) as context_size,
                        MAX(ctx_m.value) as max_context_score,
                        VAR_POP(ctx_m.value::float) as context_score_variance,
                        SUM(CASE WHEN ctx_m.value > parent_m.value THEN 1 ELSE 0 END) as better_count,
                        SUM(CASE WHEN ctx_m.value < parent_m.value THEN 1 ELSE 0 END) as worse_count
                    FROM context_edges ce
                    JOIN parent_edges pe ON ce.child_id = pe.child_id
                    JOIN measurements ctx_m ON ctx_m.candidate_id = ce.context_id AND ctx_m.name = 'combined_score'
                    JOIN measurements parent_m ON parent_m.candidate_id = pe.parent_id AND parent_m.name = 'combined_score'
                    GROUP BY ce.child_id
                ),
                -- Running best score before each candidate
                running_best AS (
                    SELECT
                        c.id as candidate_id,
                        c.campaign_id,
                        MAX(m.value) OVER (
                            PARTITION BY c.campaign_id
                            ORDER BY c.iteration_index
                            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
                        ) as best_before
                    FROM candidates c
                    JOIN measurements m ON m.candidate_id = c.id AND m.name = 'combined_score'
                    WHERE c.iteration_index IS NOT NULL
                )
                SELECT
                    c.id as candidate_id,
                    c.campaign_id,
                    c.iteration_index,
                    camp.research_question,
                    camp.models_used,
                    camp.algorithm_used,
                    child_m.value as child_score,
                    parent_m.value as parent_score,
                    COALESCE(cs.context_size, 0) as context_size,
                    COALESCE(cs.max_context_score, parent_m.value) as max_context_score,
                    COALESCE(cs.context_score_variance, 0) as context_score_variance,
                    COALESCE(cs.better_count, 0) as better_count,
                    COALESCE(cs.worse_count, 0) as worse_count,
                    COALESCE(c.context_code_diversity, 0) as context_diversity,
                    rb.best_before
                FROM candidates c
                JOIN campaigns camp ON c.campaign_id = camp.id
                JOIN parent_edges pe ON pe.child_id = c.id
                JOIN measurements child_m ON child_m.candidate_id = c.id AND child_m.name = 'combined_score'
                JOIN measurements parent_m ON parent_m.candidate_id = pe.parent_id AND parent_m.name = 'combined_score'
                LEFT JOIN context_stats cs ON cs.child_id = c.id
                LEFT JOIN running_best rb ON rb.candidate_id = c.id
                WHERE c.direct_code_embedding IS NOT NULL
                  AND camp.research_question IS NOT NULL
                  AND cs.context_size > 0
                ORDER BY c.campaign_id, c.iteration_index
            """)
            rows = cur.fetchall()

    results = []
    for row in rows:
        problem = _truncate_problem(row["research_question"])
        if problem is None:
            continue

        child_score = float(row["child_score"])
        parent_score = float(row["parent_score"])
        max_context_score = float(row["max_context_score"])
        context_size = row["context_size"]
        context_score_variance = float(row["context_score_variance"])
        better_count = row["better_count"]
        worse_count = row["worse_count"]
        best_before = float(row["best_before"]) if row["best_before"] is not None else None

        score_delta_parent_only = child_score - parent_score
        baseline = max(parent_score, max_context_score)
        score_delta = child_score - baseline

        if best_before is not None:
            score_delta_global = child_score - best_before
        else:
            score_delta_global = child_score if child_score > 0 else 0.0

        better_ratio = better_count / context_size if context_size > 0 else 0
        worse_ratio = worse_count / context_size if context_size > 0 else 0
        score_diversity = context_score_variance

        results.append({
            "candidate_id": str(row["candidate_id"]),
            "campaign_id": str(row["campaign_id"]),
            "problem": problem,
            "model": normalize_model_name(row["models_used"][0]) if row["models_used"] else "unknown",
            "algorithm": row["algorithm_used"] or "unknown",
            "iteration_index": row["iteration_index"],
            "parent_score": round(parent_score, 4),
            "child_score": round(child_score, 4),
            "max_context_score": round(max_context_score, 4),
            "best_score_so_far": round(best_before, 4) if best_before is not None else None,
            "score_delta": round(score_delta, 4),
            "score_delta_parent_only": round(score_delta_parent_only, 4),
            "score_delta_global": round(score_delta_global, 4),
            "context_size": context_size,
            "context_diversity": round(float(row["context_diversity"]), 4),
            "score_diversity": round(score_diversity, 4),
            "better_count": better_count,
            "worse_count": worse_count,
            "better_ratio": round(better_ratio, 4),
            "worse_ratio": round(worse_ratio, 4),
        })

    _set_cached(cache_key, results)
    return results


def query_mutation_factors_problems(
    database_url: str | None = None,
) -> list[str]:
    """Query distinct problems that have mutation factor data (lightweight)."""
    url = database_url or _get_database_url()

    with psycopg.connect(url, row_factory=dict_row) as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT DISTINCT camp.research_question
                FROM candidates c
                JOIN campaigns camp ON c.campaign_id = camp.id
                WHERE c.direct_code_embedding IS NOT NULL
                ORDER BY camp.research_question
            """)
            rows = cur.fetchall()

    return [_truncate_problem(r["research_question"]) for r in rows if r["research_question"]]


def get_mutation_factors_problems(
    database_url: str | None = None,
) -> list[str]:
    """Get list of problems that have mutation factor data."""
    return query_mutation_factors_problems(database_url)


# ---------------------------------------------------------------------------
# Q3a: Factor Importance
# ---------------------------------------------------------------------------

FACTOR_NAMES = {
    "context_diversity": "Code Diversity",
    "score_diversity": "Score Diversity",
    "better_ratio": "Better Ratio",
    "worse_ratio": "Worse Ratio",
}

FACTOR_KEYS = ["score_diversity", "context_diversity", "better_ratio", "worse_ratio"]


def welch_ttest_effect_size(
    group_a: np.ndarray,
    group_b: np.ndarray,
) -> dict[str, float]:
    """Compute Welch's t-test and Cohen's d between two independent groups.

    Args:
        group_a: Values for group A (e.g. improving candidates).
        group_b: Values for group B (e.g. non-improving candidates).

    Returns:
        Dict with keys: t_stat, p_value, cohens_d, mean_diff, ci_lower, ci_upper.
        CI is 95% on the raw mean difference (group_a - group_b).
    """
    group_a = np.asarray(group_a, dtype=float)
    group_b = np.asarray(group_b, dtype=float)

    n1, n2 = len(group_a), len(group_b)
    mean_a, mean_b = float(np.mean(group_a)), float(np.mean(group_b))
    s1 = float(np.std(group_a, ddof=1))
    s2 = float(np.std(group_b, ddof=1))

    t_stat, p_value = stats.ttest_ind(group_a, group_b, equal_var=False)

    pooled_std = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
    cohens_d = (mean_a - mean_b) / pooled_std if pooled_std > 0 else 0.0

    mean_diff = mean_a - mean_b
    se_diff = np.sqrt(s1**2 / n1 + s2**2 / n2)
    if se_diff > 0:
        df_num = (s1**2 / n1 + s2**2 / n2) ** 2
        df_den = (s1**2 / n1)**2 / (n1 - 1) + (s2**2 / n2)**2 / (n2 - 1)
        df = df_num / df_den if df_den > 0 else min(n1, n2) - 1
        t_crit = stats.t.ppf(0.975, df)
        ci_lower = mean_diff - t_crit * se_diff
        ci_upper = mean_diff + t_crit * se_diff
    else:
        ci_lower = mean_diff
        ci_upper = mean_diff

    return {
        "t_stat": round(float(t_stat), 4),
        "p_value": float(p_value),
        "cohens_d": round(cohens_d, 4),
        "mean_diff": round(mean_diff, 6),
        "ci_lower": round(ci_lower, 6),
        "ci_upper": round(ci_upper, 6),
    }


def compute_factor_importance(
    data: list[dict[str, Any]],
    group_by: str | None = None,
    problems_filter: list[str] | None = None,
) -> dict[str, Any]:
    """Compute factor importance for predicting improvement using direct code embeddings.

    For each factor, computes:
    - Standardized mean difference (Cohen's d with pooled std)
    - Welch's t-test p-value (unequal variance t-test)
    - 95% confidence interval on the mean difference
    """
    if problems_filter:
        data = [r for r in data if r["problem"] in problems_filter]

    if not data:
        return {"groups": [], "summary": {"total_mutations": 0}}

    improvements = [r for r in data if r["score_delta_global"] > 0]
    non_improvements = [r for r in data if r["score_delta_global"] <= 0]

    if not improvements or not non_improvements:
        return {"groups": [], "summary": {"total_mutations": len(data)}}

    def get_group_key(r: dict) -> str:
        if group_by == "model":
            return r.get("model") or "unknown"
        elif group_by == "algorithm":
            return r.get("algorithm") or "unknown"
        elif group_by == "model_algorithm":
            model = r.get("model") or "unknown"
            alg = r.get("algorithm") or "unknown"
            return f"{model} + {alg}"
        else:
            return "all"

    imp_by_group: dict[str, list[dict]] = {}
    non_imp_by_group: dict[str, list[dict]] = {}

    for r in improvements:
        key = get_group_key(r)
        if key not in imp_by_group:
            imp_by_group[key] = []
        imp_by_group[key].append(r)

    for r in non_improvements:
        key = get_group_key(r)
        if key not in non_imp_by_group:
            non_imp_by_group[key] = []
        non_imp_by_group[key].append(r)

    groups = []
    all_group_keys = set(imp_by_group.keys()) & set(non_imp_by_group.keys())

    for group_name in sorted(all_group_keys):
        imp_data = imp_by_group.get(group_name, [])
        non_imp_data = non_imp_by_group.get(group_name, [])

        if len(imp_data) < 5 or len(non_imp_data) < 5:
            continue

        correlations = {}
        p_values = {}
        ci_lower = {}
        ci_upper = {}
        cohens_d = {}

        for factor in FACTOR_KEYS:
            imp_values = np.array([r[factor] for r in imp_data], dtype=float)
            non_imp_values = np.array([r[factor] for r in non_imp_data], dtype=float)

            result = welch_ttest_effect_size(imp_values, non_imp_values)
            correlations[factor] = result["cohens_d"]
            p_values[factor] = result["p_value"]
            cohens_d[factor] = result["cohens_d"]
            ci_lower[factor] = result["ci_lower"]
            ci_upper[factor] = result["ci_upper"]

        groups.append({
            "group": group_name,
            "n": len(imp_data) + len(non_imp_data),
            "n_improvements": len(imp_data),
            "n_non_improvements": len(non_imp_data),
            "correlations": correlations,
            "p_values": p_values,
            "cohens_d": cohens_d,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "improvement_rate": round(len(imp_data) / (len(imp_data) + len(non_imp_data)), 4),
        })

    return {
        "group_by": group_by,
        "problems_filter": problems_filter,
        "groups": groups,
        "summary": {
            "total_groups": len(groups),
            "total_mutations": sum(g["n"] for g in groups),
            "total_improvements": sum(g["n_improvements"] for g in groups),
        },
    }


def _significance_stars(p: float | None) -> str:
    """Return significance stars for a p-value."""
    if p is None:
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def _generate_factor_heatmap(
    groups: list[dict[str, Any]],
    group_by: str,
) -> bytes:
    """Generate heatmap for factor importance by model or model+algorithm."""
    if group_by == "model_algorithm":
        by_algorithm: dict[str, dict[str, dict]] = {}
        for g in groups:
            group_name = g["group"]
            if " + " in group_name:
                model, algorithm = group_name.rsplit(" + ", 1)
            else:
                model, algorithm = group_name, "unknown"
            if algorithm not in by_algorithm:
                by_algorithm[algorithm] = {}
            by_algorithm[algorithm][model] = {
                "correlations": g.get("correlations", {}),
                "p_values": g.get("p_values", {}),
                "n": g.get("n", 0),
                "n_improvements": g.get("n_improvements", 0),
            }

        algorithms = sorted(by_algorithm.keys())
        algorithms = [alg for alg in algorithms if by_algorithm[alg]]
        n_algs = len(algorithms)

        if n_algs == 0:
            return _generate_empty_figure("No data for heatmap")

        all_models = sorted(set(
            m for alg_data in by_algorithm.values() for m in alg_data.keys()
            if alg_data[m].get("correlations")
        ))

        if not all_models:
            return _generate_empty_figure("No data for heatmap")

        n_models = len(all_models)
        fig, axes = plt.subplots(
            1, n_algs, figsize=(6 * n_algs, max(6, n_models * 0.65 + 2)),
            squeeze=False)
        axes = axes[0]

        cmap = plt.cm.RdYlGn
        vmin, vmax = -1.0, 1.0

        im = None
        for idx, algorithm in enumerate(algorithms):
            ax = axes[idx]
            alg_data = by_algorithm[algorithm]
            models_present = [m for m in all_models if m in alg_data and alg_data[m].get("correlations")]

            # rows = models, cols = factors
            matrix = np.full((len(models_present), len(FACTOR_KEYS)), np.nan)
            p_matrix = np.full((len(models_present), len(FACTOR_KEYS)), np.nan)
            n_values = []

            for row_idx, model in enumerate(models_present):
                corrs = alg_data[model]["correlations"]
                p_vals = alg_data[model].get("p_values", {})
                n_values.append(alg_data[model].get("n", 0))
                for col_idx, factor in enumerate(FACTOR_KEYS):
                    if factor in corrs:
                        matrix[row_idx, col_idx] = corrs[factor]
                    if factor in p_vals:
                        p_matrix[row_idx, col_idx] = p_vals[factor]

            im = ax.imshow(matrix, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)

            ax.set_xticks(np.arange(len(FACTOR_KEYS)))
            ax.set_xticklabels(
                [FACTOR_NAMES[f] for f in FACTOR_KEYS],
                rotation=30, ha="right", fontsize=13)
            ax.set_yticks(np.arange(len(models_present)))
            y_labels = [f"{m} (n={n_values[i]})" for i, m in enumerate(models_present)]
            if idx == 0:
                ax.set_yticklabels(y_labels, fontsize=12)
            else:
                ax.set_yticklabels(y_labels, fontsize=12)

            for i in range(len(models_present)):
                for j in range(len(FACTOR_KEYS)):
                    val = matrix[i, j]
                    if np.isnan(val):
                        continue
                    p_val = p_matrix[i, j] if not np.isnan(p_matrix[i, j]) else None
                    is_sig = p_val is not None and p_val < 0.05
                    text_color = "white" if abs(val) > 0.6 else "black"
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                           fontsize=12, color=text_color)
                    if not is_sig:
                        ax.add_patch(Rectangle(
                            (j - 0.5, i - 0.5), 1, 1,
                            fill=False, hatch='///', alpha=0.4,
                            edgecolor='gray', linewidth=0.5))

            ax.set_title(algorithm, fontweight="bold", fontsize=16)

        plt.tight_layout()

        if im is not None:
            cbar = fig.colorbar(im, ax=axes.tolist(), orientation="vertical",
                               fraction=0.015, pad=0.06)
            cbar.set_label("Std. Mean Diff. (Cohen's $d$)", fontsize=14)
            cbar.ax.tick_params(labelsize=12)

    else:
        models = [g["group"] for g in groups]

        matrix = np.full((len(FACTOR_KEYS), len(models)), np.nan)
        p_matrix = np.full((len(FACTOR_KEYS), len(models)), np.nan)
        for col_idx, g in enumerate(groups):
            corrs = g.get("correlations", {})
            p_vals = g.get("p_values", {})
            for row_idx, factor in enumerate(FACTOR_KEYS):
                if factor in corrs:
                    matrix[row_idx, col_idx] = corrs[factor]
                if factor in p_vals:
                    p_matrix[row_idx, col_idx] = p_vals[factor]

        fig, ax = plt.subplots(figsize=(max(12, len(models) * 1.5), 6))

        cmap = plt.cm.RdYlGn
        vmin, vmax = -1.0, 1.0

        im = ax.imshow(matrix, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)

        x_labels = [f"{m}\n(n={groups[i].get('n', '?')})" for i, m in enumerate(models)]
        ax.set_xticks(np.arange(len(models)))
        ax.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=14)
        ax.set_yticks(np.arange(len(FACTOR_KEYS)))
        ax.set_yticklabels([FACTOR_NAMES[f] for f in FACTOR_KEYS], fontsize=16)

        for i in range(len(FACTOR_KEYS)):
            for j in range(len(models)):
                val = matrix[i, j]
                if np.isnan(val):
                    continue
                p_val = p_matrix[i, j] if not np.isnan(p_matrix[i, j]) else None
                is_sig = p_val is not None and p_val < 0.05
                text_color = "white" if abs(val) > 0.6 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                       fontsize=14, color=text_color)
                if not is_sig:
                    ax.add_patch(Rectangle(
                        (j - 0.5, i - 0.5), 1, 1,
                        fill=False, hatch='///', alpha=0.4,
                        edgecolor='gray', linewidth=0.5))

        cbar = fig.colorbar(im, ax=ax, orientation="vertical", fraction=0.03, pad=0.04)
        cbar.set_label("Std. Mean Diff. (Cohen's $d$)", fontsize=16)
        cbar.ax.tick_params(labelsize=14)


        plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


def generate_factor_importance_figure(
    data: dict[str, Any],
) -> bytes:
    """Generate visualization for factor importance (correlations)."""
    groups = data.get("groups", [])
    if not groups:
        return _generate_empty_figure("No mutation factor data available")

    group_by = data.get("group_by")
    n_groups = len(groups)

    if group_by is None or n_groups == 1:
        fig, ax = plt.subplots(figsize=(6, 4.5))
        group = groups[0]
        correlations = group.get("correlations", {})
        p_values = group.get("p_values", {})

        factors = []
        values = []
        colors = []
        p_vals_list = []

        for factor in FACTOR_KEYS:
            if factor in correlations:
                factors.append(FACTOR_NAMES[factor])
                val = correlations[factor]
                values.append(val)
                colors.append(COLORS["high"] if val > 0 else COLORS["low"])
                p_vals_list.append(p_values.get(factor))

        if not factors:
            return _generate_empty_figure("Insufficient data to compute correlations")

        x_pos = np.arange(len(factors))
        bars = ax.bar(x_pos, values, color=colors, alpha=0.8, width=0.6)

        # Show p-value labels above/below each bar
        y_range = max(abs(v) for v in values) if values else 0.5
        y_pad = y_range * 0.05
        for i, (val, p_val) in enumerate(zip(values, p_vals_list)):
            if p_val is None:
                continue
            if p_val < 0.001:
                p_str = "p<.001"
            elif p_val < 0.01:
                p_str = f"p={p_val:.3f}"
            else:
                p_str = f"p={p_val:.2f}"
            if val >= 0:
                y_pos = val + y_pad
                va = 'bottom'
            else:
                y_pos = val - y_pad
                va = 'top'
            ax.text(i, y_pos, p_str, ha='center', va=va,
                    fontsize=13, color='0.2', fontweight='bold')

        ax.set_xticks(x_pos)
        ax.set_xticklabels(factors, rotation=30, ha='right')
        ax.set_ylabel("Std. Mean Diff.")
        ax.axhline(y=0, color="gray", linestyle="-", linewidth=0.5)
        ax.grid(True, alpha=0.3, axis="y")

        # Expand y-axis to prevent label clipping
        y_max = max(abs(v) for v in values) if values else 0.5
        ax.set_ylim(-y_max * 1.35, y_max * 1.35)

        n_imp = group.get("n_improvements", 0)
        n_non = group.get("n_non_improvements", 0)
        ax.text(0.02, 0.98, f"$|C^{{\\uparrow}}|={n_imp}$, $|C^{{\\downarrow}}|={n_non}$",
                transform=ax.transAxes, fontsize=12, va='top', ha='left',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

        plt.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=300)
        plt.close(fig)
        buf.seek(0)
        return buf.getvalue()

    elif group_by in ("model", "model_algorithm"):
        return _generate_factor_heatmap(groups, group_by)

    else:
        fig, ax = plt.subplots(figsize=(12, 6))

        x = np.arange(len(FACTOR_KEYS))
        width = 0.8 / n_groups

        for i, group in enumerate(groups):
            correlations = group.get("correlations", {})
            values = [correlations.get(f, 0) for f in FACTOR_KEYS]
            offset = (i - n_groups / 2 + 0.5) * width
            label = group["group"]
            if len(label) > 20:
                label = label[:18] + "..."
            n_imp = group.get('n_improvements', 0)
            n_non = group.get('n_non_improvements', 0)
            ax.bar(x + offset, values, width, label=f"{label} ($|C^\\uparrow|$={n_imp}, $|C^\\downarrow|$={n_non})",
                   color=PALETTE[i % len(PALETTE)], alpha=0.8)

        ax.set_xticks(x)
        ax.set_xticklabels([FACTOR_NAMES[f] for f in FACTOR_KEYS], rotation=15, ha="right")
        ax.set_ylabel("Std. Mean Diff.")
        ax.axhline(y=0, color="gray", linestyle="-", linewidth=0.5)
        ax.grid(True, alpha=0.3, axis="y")
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=10, framealpha=0.9)

        plt.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=300)
        plt.close(fig)
        buf.seek(0)
        return buf.getvalue()


# ===========================================================================
# HIGH-LEVEL API: Tab 1 - Diversity vs Score
# ===========================================================================

def get_diversity_vs_score_scatter(
    database_url: str | None = None,
    group_by: str | None = None,
    problems: list[str] | None = None,
) -> dict[str, Any]:
    """Get diversity vs score scatter data using direct code embeddings."""
    campaigns = query_campaign_diversity_and_scores(database_url)
    return process_diversity_vs_score_scatter(campaigns, group_by, problems)


def get_diversity_vs_score_scatter_figure(
    database_url: str | None = None,
    group_by: str | None = None,
    problems: list[str] | None = None,
) -> bytes:
    """Get diversity vs score scatter figure as PNG using direct code embeddings."""
    data = get_diversity_vs_score_scatter(database_url, group_by, problems)
    return generate_diversity_vs_score_scatter_figure(data)


# ===========================================================================
# HIGH-LEVEL API: Tab 2 - Early Diversity
# ===========================================================================

def get_early_diversity_vs_outcome(
    database_url: str | None = None,
    early_fraction: float = 0.25,
) -> dict[str, Any]:
    """Get early diversity vs final outcome analysis using direct code embeddings."""
    return process_early_diversity_vs_outcome(
        early_fraction=early_fraction, database_url=database_url
    )


def get_early_diversity_scatter(
    database_url: str | None = None,
    early_fraction: float = 0.25,
    group_by: str | None = None,
    problems: list[str] | None = None,
) -> dict[str, Any]:
    """Get early diversity scatter data with grouping using direct code embeddings."""
    return process_early_diversity_scatter(
        early_fraction=early_fraction, group_by=group_by, problems=problems, database_url=database_url
    )


def get_early_diversity_scatter_figure(
    database_url: str | None = None,
    early_fraction: float = 0.25,
    group_by: str | None = None,
    problems: list[str] | None = None,
) -> bytes:
    """Get early diversity scatter figure as PNG using direct code embeddings."""
    data = get_early_diversity_scatter(database_url, early_fraction, group_by, problems)
    return generate_early_diversity_scatter_figure(data)


# ===========================================================================
# HIGH-LEVEL API: Tab 3 - Top-K Diversity
# ===========================================================================

def get_topk_winners_diversity(
    database_url: str | None = None,
    top_pct: float = 0.25,
    group_by: str | None = None,
    problems: list[str] | None = None,
) -> dict[str, Any]:
    """Get top winners diversity analysis using direct code embeddings.

    Args:
        database_url: Database connection string
        top_pct: Fraction of campaigns to consider "top" (default 0.25 = top 25%)
        group_by: How to group campaigns (model, algorithm, model_algorithm, or None)
        problems: Optional list of problems to include
    """
    campaigns = query_campaigns_with_best_embeddings(database_url)
    return process_topk_winners_diversity(campaigns, top_pct, group_by, problems)


def get_topk_winners_diversity_figure(
    database_url: str | None = None,
    top_pct: float = 0.25,
    group_by: str | None = None,
    problems: list[str] | None = None,
) -> bytes:
    """Get top winners diversity figure as PNG using direct code embeddings.

    Args:
        database_url: Database connection string
        top_pct: Fraction of campaigns to consider "top" (default 0.25 = top 25%)
        group_by: How to group campaigns (model, algorithm, model_algorithm, or None)
        problems: Optional list of problems to include
    """
    data = get_topk_winners_diversity(database_url, top_pct, group_by, problems)
    return generate_topk_winners_diversity_figure(data)


# ===========================================================================
# HIGH-LEVEL API: Tab 4 - Factor Importance
# ===========================================================================

def get_factor_importance_figure(
    database_url: str | None = None,
    group_by: str | None = None,
    problems: list[str] | None = None,
) -> bytes:
    """Get factor importance bar chart as PNG using direct code embeddings."""
    raw_data = query_mutation_factors(database_url)
    importance_data = compute_factor_importance(raw_data, group_by, problems)
    return generate_factor_importance_figure(importance_data)


# ===========================================================================
# CLI Entry Point
# ===========================================================================

if __name__ == "__main__":
    import json
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--generate-figures":
        output_dir = sys.argv[2] if len(sys.argv) > 2 else "."
        print(f"Generating figures to {output_dir}...")

        # Generate diversity vs score scatter
        fig_data = get_diversity_vs_score_scatter_figure()
        with open(f"{output_dir}/q3_code_diversity_vs_score.png", "wb") as f:
            f.write(fig_data)
        print("  Generated: q3_code_diversity_vs_score.png")

        # Generate early diversity scatter
        fig_data = get_early_diversity_scatter_figure()
        with open(f"{output_dir}/q3_code_early_diversity.png", "wb") as f:
            f.write(fig_data)
        print("  Generated: q3_code_early_diversity.png")

        # Generate top-K diversity
        fig_data = get_topk_winners_diversity_figure()
        with open(f"{output_dir}/q3_code_topk_diversity.png", "wb") as f:
            f.write(fig_data)
        print("  Generated: q3_code_topk_diversity.png")
    else:
        result = get_diversity_vs_score_scatter()
        print(json.dumps(result, indent=2, default=str))
