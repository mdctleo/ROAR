#!/usr/bin/env python3
"""Q3: Solution Diversity Analytics (Summary-First Approach).

Research questions:
1. Does exploring diverse algorithmic approaches lead to better outcomes?
2. Does early diversity predict better final outcomes?
3. Do different models/algorithms produce more diverse candidates?
4. Are the top-performing candidates diverse, or do they converge to similar approaches?
5. Does showing the generator diverse context lead to more novel output?

File organization:
- Shared utilities and database queries
- Tab 1: Diversity vs Score (scatter plot)
- Tab 2: Early Diversity (early iterations vs outcome)
- Tab 3: Context Diversity (context diversity vs output novelty)
- Tab 4: Top-K Diversity (winners convergence analysis)
- High-level API functions
"""

import io
from typing import Any

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import psycopg
from psycopg.rows import dict_row
from sklearn.metrics.pairwise import cosine_similarity

from analytics.utils import normalize_model_name


# ===========================================================================
# SHARED: Configuration and Constants
# ===========================================================================

DATABASE_URL_DEFAULT = "postgresql://postgres:postgres@localhost:5432/adrs"

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.titlesize': 12,
    'figure.dpi': 150,
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
# Colors chosen for maximum distinguishability (based on ColorBrewer qualitative palettes)
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
# circle, square, triangle-up, diamond, triangle-down, plus, X

LINESTYLE_PALETTE = ['-', '--', '-.', ':', (0, (3, 1, 1, 1))]
# solid, dashed, dash-dot, dotted, densely dash-dotted

PROBLEMS = ["Knapsack", "Palindrome", "Polyomino"]
GROUP_BYS = ["algorithm", "model", "model_algorithm"]


# ===========================================================================
# SHARED: Utility Functions
# ===========================================================================

def _get_database_url() -> str:
    import os
    return os.environ.get("DATABASE_URL", DATABASE_URL_DEFAULT)


def _truncate_problem(rq: str | None) -> str | None:
    """Extract short problem name from research question.

    Also replaces commas with semicolons to avoid breaking comma-delimited
    query parameters in the API.
    """
    if not rq:
        return None
    rq_lower = rq.lower()
    if "knapsack" in rq_lower:
        return "Knapsack"
    elif "polyomino" in rq_lower or "packing" in rq_lower:
        return "Polyomino"
    elif "palindrome" in rq_lower or "hamiltonian" in rq_lower:
        return "Palindrome"
    else:
        truncated = rq[:30] + "..." if len(rq) > 30 else rq
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
# SHARED: Database Queries
# ===========================================================================

def query_campaign_diversity_and_scores(database_url: str | None = None) -> list[dict[str, Any]]:
    """Query campaign-level diversity scores and best scores.

    Overview:
        For each campaign (a single optimization run), computes how diverse
        the candidate solutions are from each other. A high diversity score
        means the run explored many different algorithmic approaches; a low
        score means candidates converged to similar solutions.

    How diversity is calculated:
        1. Each candidate has a "solution summary embedding" - a 1536-dim vector
           (from text-embedding-3-small) representing the semantic meaning of
           an LLM-generated summary of that candidate's algorithmic approach.

        2. For each pair of candidates in a campaign, we compute cosine similarity
           using pgvector's <=> (cosine distance) operator in the database.

        3. Diversity score = 1 - average_similarity = average cosine distance
           Higher diversity (closer to 1) = candidates are dissimilar from each other

    Returns:
        List of dicts with campaign metadata, diversity_score, and best_score.
        Campaigns with <2 embeddings are excluded.
    """
    url = database_url or _get_database_url()

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
                        AVG(c1.solution_summary_embedding <=> c2.solution_summary_embedding) as diversity_score
                    FROM candidates c1
                    JOIN candidates c2 ON c1.campaign_id = c2.campaign_id AND c1.id < c2.id
                    WHERE c1.solution_summary_embedding IS NOT NULL
                      AND c2.solution_summary_embedding IS NOT NULL
                    GROUP BY c1.campaign_id
                ),
                campaign_counts AS (
                    SELECT campaign_id, COUNT(*) as n_candidates
                    FROM candidates
                    WHERE solution_summary_embedding IS NOT NULL
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

    return results


def query_campaigns_with_best_embeddings(
    database_url: str | None = None,
) -> list[dict[str, Any]]:
    """Query campaigns with the embedding of their best-scoring candidate.

    For each campaign, finds the candidate with the highest combined_score
    and returns that candidate's embedding along with campaign metadata.

    This is used for computing diversity ACROSS top winners (not within-run diversity).

    Returns:
        List of dicts with campaign_id, problem, model, algorithm, best_score,
        and best_embedding (the embedding of the top-scoring candidate).
    """
    url = database_url or _get_database_url()

    with psycopg.connect(url, row_factory=dict_row) as conn:
        with conn.cursor() as cur:
            cur.execute("""
                WITH best_candidates AS (
                    SELECT DISTINCT ON (c.campaign_id)
                        c.campaign_id,
                        c.id as candidate_id,
                        c.solution_summary_embedding::text as embedding,
                        m.value as score
                    FROM candidates c
                    JOIN measurements m ON m.candidate_id = c.id AND m.name = 'combined_score'
                    WHERE c.solution_summary_embedding IS NOT NULL
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

    return results


# ===========================================================================
# TAB 1: Diversity vs Score
# ===========================================================================

def process_diversity_vs_score_scatter(
    campaigns: list[dict[str, Any]],
    group_by: str | None = None,
    problems: list[str] | None = None,
) -> dict[str, Any]:
    """Process campaign data for diversity vs score scatter plot.

    Args:
        campaigns: List of campaign data with diversity_score and best_score
        group_by: None (no grouping), "model", "algorithm", or "model_algorithm"
        problems: Optional list of problems to include (None = all)

    Returns:
        Dict with points, group_stats, and overall correlation
    """
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
    overall_corr = float(np.corrcoef(all_div, all_score)[0, 1]) if len(points) >= 4 else None

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
        corr = float(np.corrcoef(divs, scores)[0, 1]) if len(group_points) >= 4 else None

        group_stats.append({
            "group": group_name,
            "n": len(group_points),
            "correlation": round(corr, 4) if corr and not np.isnan(corr) else None,
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
    """Generate scatter plot of diversity vs score with optional grouping.

    Encoding strategy:
    - Color encodes model
    - Marker shape encodes algorithm
    - Line style encodes algorithm
    For model_algorithm grouping, uses faceted layout (one panel per algorithm).
    """
    points = data.get("points", [])
    if not points:
        return _generate_empty_figure("No diversity data available")

    group_by = data.get("group_by")
    group_stats = data.get("group_stats", [])
    problems = data.get("problems")

    if group_by is None:
        # No grouping - single panel, single style
        fig, ax = plt.subplots(figsize=(10, 7))
        x = [p["diversity"] for p in points]
        y = [p["score"] for p in points]

        ax.scatter(x, y, c=COLOR_PALETTE[0], alpha=0.6, s=40, edgecolors='white', linewidths=0.5)

        if len(x) >= 2:
            z = np.polyfit(x, y, 1)
            p_fit = np.poly1d(z)
            x_line = np.linspace(min(x), max(x), 100)
            ax.plot(x_line, p_fit(x_line), '--', color=COLOR_PALETTE[0], alpha=0.8, linewidth=2)

        corr = data.get("overall_correlation")
        corr_str = f"r = {corr:.3f}" if corr is not None else "r = N/A"
        ax.set_title(f'Diversity vs Final Score (All Runs)\n{corr_str}, n={len(points)}',
                     fontweight='bold')
        ax.set_xlabel('Run Diversity (1 - avg cosine similarity)')
        ax.set_ylabel('Best Score')
        ax.grid(True, alpha=0.3)

        if problems:
            ax.set_title(ax.get_title() + f'\nProblems: {", ".join(problems)}', fontsize=10)

    elif group_by == "model_algorithm":
        # Faceted layout: one panel per algorithm
        # Color + marker + line style ALL encode model (same model looks identical across panels)
        algorithms = sorted(set(g["group"].rsplit(" + ", 1)[1] for g in group_stats if " + " in g["group"]))
        models = sorted(set(g["group"].rsplit(" + ", 1)[0] for g in group_stats if " + " in g["group"]))

        if not algorithms:
            return _generate_empty_figure("No model+algorithm groups found")

        n_algs = len(algorithms)
        n_models = len(models)
        fig_width = min(20, max(10, 5 * n_algs))
        fig, axes = plt.subplots(1, n_algs, figsize=(fig_width, 6), squeeze=False, sharey=True)
        axes = axes[0]

        # Create consistent style mapping for models (same across all facets)
        # Use matplotlib's categorical colormaps for better distinction
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

                # Only add label on first subplot (for shared legend)
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
            if n_algs > 1:
                ax.set_title(algorithm, fontweight='bold', fontsize=11)

        # Title
        overall_corr = data.get("overall_correlation")
        corr_str = f"Overall r = {overall_corr:.3f}" if overall_corr is not None else ""
        prob_str = f' | {", ".join(problems)}' if problems else ""
        if n_algs == 1:
            alg_suffix = f' ({algorithms[0]})'
        else:
            alg_suffix = ''
        fig.suptitle(f'Diversity vs Score by Model + Algorithm{alg_suffix}\n{corr_str}, n={len(points)}{prob_str}',
                     fontweight='bold', fontsize=12, y=1.02)

        plt.tight_layout()

        # Single legend below the plot (like Q2)
        handles, labels = axes[0].get_legend_handles_labels()
        if handles:
            n_legend_cols = min(6, max(1, n_models))
            fig.legend(handles, labels, loc='upper center', fontsize=9,
                      ncol=n_legend_cols, bbox_to_anchor=(0.5, -0.02))

    else:
        # Single panel for model or algorithm grouping
        # Each group gets unique color + marker + linestyle (like Q2)
        groups = sorted(gs["group"] for gs in group_stats)
        n_groups = len(groups)

        fig_width = max(10, min(16, 2.5 * max(n_groups // 3, 4)))
        fig, ax = plt.subplots(figsize=(fig_width, 7))

        # Use matplotlib's categorical colormaps for better distinction
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
            corr = gs.get("correlation")
            if corr is not None:
                label += f" (r={corr:.2f})"

            ax.scatter(x, y, c=[color], marker=marker, alpha=0.6, s=40,
                      edgecolors='white', linewidths=0.5, label=label)

            if len(x) >= 2:
                z = np.polyfit(x, y, 1)
                p_fit = np.poly1d(z)
                x_line = np.linspace(min(x), max(x), 100)
                ax.plot(x_line, p_fit(x_line), linestyle=linestyle, color=color,
                       alpha=0.7, linewidth=1.8)

        overall_corr = data.get("overall_correlation")
        corr_str = f"Overall r = {overall_corr:.3f}" if overall_corr is not None else ""
        group_label = "Model" if group_by == "model" else "Algorithm"
        ax.set_title(f'Diversity vs Final Score by {group_label}\n{corr_str}, n={len(points)}',
                     fontweight='bold')

        if problems:
            ax.set_title(ax.get_title() + f'\nProblems: {", ".join(problems)}', fontsize=10)

        ax.set_xlabel('Run Diversity (1 - avg cosine similarity)')
        ax.set_ylabel('Best Score')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=9, ncol=min(3, max(1, n_groups // 4 + 1)))

    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=150)
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
    """Compute early diversity for each campaign.

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
                    WHERE c.solution_summary_embedding IS NOT NULL
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
                        c.solution_summary_embedding
                    FROM candidates c
                    JOIN campaign_stats cs ON c.campaign_id = cs.campaign_id
                    WHERE c.solution_summary_embedding IS NOT NULL
                      AND c.iteration_index IS NOT NULL
                      AND c.iteration_index <= (cs.max_iter * %(early_fraction)s)::int
                ),
                early_diversity AS (
                    -- Compute pairwise diversity among early candidates
                    SELECT
                        e1.campaign_id,
                        COUNT(*) as n_pairs,
                        AVG(e1.solution_summary_embedding <=> e2.solution_summary_embedding) as early_diversity
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
    """Analyze whether diversity in early iterations predicts final outcome.

    For each campaign, computes diversity among candidates in the first N% of
    iterations, then correlates with the campaign's best final score.

    Args:
        early_fraction: Fraction of iterations to consider "early" (default 0.25 = first 25%)
        database_url: Database connection URL
    """
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

        correlation = float(np.corrcoef(early_divs, best_scores)[0, 1])

        problem_results.append({
            "problem": problem,
            "n_campaigns": len(camps),
            "correlation": round(correlation, 4) if not np.isnan(correlation) else None,
            "early_diversity_mean": round(float(np.mean(early_divs)), 4),
            "early_diversity_std": round(float(np.std(early_divs)), 4),
            "best_score_mean": round(float(np.mean(best_scores)), 4),
            "best_score_std": round(float(np.std(best_scores)), 4),
            "campaigns": camps,
        })

    all_early_divs = [c["early_diversity"] for c in campaign_data]
    all_best_scores = [c["best_score"] for c in campaign_data]
    overall_corr = float(np.corrcoef(all_early_divs, all_best_scores)[0, 1]) if len(campaign_data) >= 4 else None

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
    """Process early diversity data for scatter plot with grouping.

    Args:
        early_fraction: Fraction of iterations to consider "early" (default 0.25)
        group_by: None (no grouping), "model", "algorithm", or "model_algorithm"
        problems: Optional list of problems to include (None = all)
        database_url: Database connection URL

    Returns:
        Dict with points, group_stats, and overall correlation
    """
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
    overall_corr = float(np.corrcoef(all_div, all_score)[0, 1]) if len(points) >= 4 else None

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
        corr = float(np.corrcoef(divs, scores)[0, 1]) if len(group_points) >= 4 else None

        group_stats.append({
            "group": group_name,
            "n": len(group_points),
            "correlation": round(corr, 4) if corr and not np.isnan(corr) else None,
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
    """Generate scatter plot of early diversity vs score with optional grouping.

    Encoding strategy:
    - Color encodes model
    - Marker shape encodes algorithm
    - Line style encodes algorithm
    For model_algorithm grouping, uses faceted layout (one panel per algorithm).
    """
    points = data.get("points", [])
    if not points:
        return _generate_empty_figure("No early diversity data available")

    group_by = data.get("group_by")
    group_stats = data.get("group_stats", [])
    early_fraction = data.get("early_fraction", 0.25)
    problems = data.get("problems")

    if group_by is None:
        # No grouping - single panel, single style
        fig, ax = plt.subplots(figsize=(10, 7))
        x = [p["early_diversity"] for p in points]
        y = [p["best_score"] for p in points]

        ax.scatter(x, y, c=COLOR_PALETTE[0], alpha=0.6, s=40, edgecolors='white', linewidths=0.5)

        if len(x) >= 2:
            z = np.polyfit(x, y, 1)
            p_fit = np.poly1d(z)
            x_line = np.linspace(min(x), max(x), 100)
            ax.plot(x_line, p_fit(x_line), '--', color=COLOR_PALETTE[0], alpha=0.8, linewidth=2)

        corr = data.get("overall_correlation")
        corr_str = f"r = {corr:.3f}" if corr is not None else "r = N/A"
        ax.set_title(f'Early Diversity vs Final Score (All Runs)\n{corr_str}, n={len(points)}',
                     fontweight='bold')
        ax.set_xlabel(f'Early Diversity (first {int(early_fraction * 100)}% of iterations)')
        ax.set_ylabel('Best Score')
        ax.grid(True, alpha=0.3)

        if problems:
            ax.set_title(ax.get_title() + f'\nProblems: {", ".join(problems)}', fontsize=10)

    elif group_by == "model_algorithm":
        # Faceted layout: one panel per algorithm
        # Color + marker + line style ALL encode model (same model looks identical across panels)
        algorithms = sorted(set(g["group"].rsplit(" + ", 1)[1] for g in group_stats if " + " in g["group"]))
        models = sorted(set(g["group"].rsplit(" + ", 1)[0] for g in group_stats if " + " in g["group"]))

        if not algorithms:
            return _generate_empty_figure("No model+algorithm groups found")

        n_algs = len(algorithms)
        n_models = len(models)
        fig_width = min(20, max(10, 5 * n_algs))
        fig, axes = plt.subplots(1, n_algs, figsize=(fig_width, 6), squeeze=False, sharey=True)
        axes = axes[0]

        # Create consistent style mapping for models (same across all facets)
        # Use matplotlib's categorical colormaps for better distinction
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

                # Only add label on first subplot (for shared legend)
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
            if n_algs > 1:
                ax.set_title(algorithm, fontweight='bold', fontsize=11)

        # Title
        overall_corr = data.get("overall_correlation")
        corr_str = f"Overall r = {overall_corr:.3f}" if overall_corr is not None else ""
        prob_str = f' | {", ".join(problems)}' if problems else ""
        if n_algs == 1:
            alg_suffix = f' ({algorithms[0]})'
        else:
            alg_suffix = ''
        fig.suptitle(f'Early Diversity vs Score by Model + Algorithm{alg_suffix}\n{corr_str}, n={len(points)}{prob_str}',
                     fontweight='bold', fontsize=12, y=1.02)

        plt.tight_layout()

        # Single legend below the plot (like Q2)
        handles, labels = axes[0].get_legend_handles_labels()
        if handles:
            n_legend_cols = min(6, max(1, n_models))
            fig.legend(handles, labels, loc='upper center', fontsize=9,
                      ncol=n_legend_cols, bbox_to_anchor=(0.5, -0.02))

    else:
        # Single panel for model or algorithm grouping
        # Each group gets unique color + marker + linestyle (like Q2)
        groups = sorted(gs["group"] for gs in group_stats)
        n_groups = len(groups)

        fig_width = max(10, min(16, 2.5 * max(n_groups // 3, 4)))
        fig, ax = plt.subplots(figsize=(fig_width, 7))

        # Use matplotlib's categorical colormaps for better distinction
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
            corr = gs.get("correlation")
            if corr is not None:
                label += f" (r={corr:.2f})"

            ax.scatter(x, y, c=[color], marker=marker, alpha=0.6, s=40,
                      edgecolors='white', linewidths=0.5, label=label)

            if len(x) >= 2:
                z = np.polyfit(x, y, 1)
                p_fit = np.poly1d(z)
                x_line = np.linspace(min(x), max(x), 100)
                ax.plot(x_line, p_fit(x_line), linestyle=linestyle, color=color,
                       alpha=0.7, linewidth=1.8)

        overall_corr = data.get("overall_correlation")
        corr_str = f"Overall r = {overall_corr:.3f}" if overall_corr is not None else ""
        group_label = "Model" if group_by == "model" else "Algorithm"
        ax.set_title(f'Early Diversity vs Final Score by {group_label}\n{corr_str}, n={len(points)}',
                     fontweight='bold')

        if problems:
            ax.set_title(ax.get_title() + f'\nProblems: {", ".join(problems)}', fontsize=10)

        ax.set_xlabel(f'Early Diversity (first {int(early_fraction * 100)}% of iterations)')
        ax.set_ylabel('Best Score')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=9, ncol=min(3, max(1, n_groups // 4 + 1)))

    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=150)
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


# ===========================================================================
# TAB 4: Top-K Diversity
# ===========================================================================

def _compute_cross_diversity(embeddings: list[list[float]]) -> float:
    """Compute pairwise diversity among a set of embeddings.

    Returns 1 - average cosine similarity. Higher = more diverse.
    Returns 0 if fewer than 2 embeddings.
    """
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
    """Compute diversity ACROSS top winners vs other winners.

    Answers: Do the best runs converge to the same winning approach,
    or do different runs find different good solutions?

    For each group:
    1. Filters to only runs where best_score > 0 (beat baseline)
    2. Takes top X% runs by best_score, computes diversity among their winners
    3. Takes remaining runs (bottom 1-X%), computes diversity among their winners
    4. Compares: are top performers more or less convergent than typical runs?

    Groups are excluded if either the top or other bucket has fewer than 2 campaigns,
    since we need at least 2 to compute pairwise diversity.

    Args:
        campaigns: List of campaign data with best_score and best_embedding
        top_pct: Fraction of campaigns to consider "top" (default 0.25 = top 25%)
        group_by: None, "model", "algorithm", or "model_algorithm"
        problems_filter: Optional list of problems to include
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
    """Generate bar chart comparing diversity across top winners vs other winners.

    Shows whether top runs converge to similar solutions or find diverse approaches.
    Only includes runs that beat baseline (score > 0).
    For model_algorithm grouping, uses faceted layout (one panel per algorithm).
    """
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
        # Faceted layout: one panel per algorithm
        # Parse groups into algorithm -> {model -> data}
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

        # Get all unique models across algorithms
        all_models = sorted(set(
            model for alg_data in by_algorithm.values() for model in alg_data.keys()
        ))
        n_models = len(all_models)

        # Calculate max model name length to determine figure sizing
        max_model_len = max(len(m) for m in all_models) if all_models else 10

        # Create figure with one subplot per algorithm
        # Width per algorithm scales with number of models and name length
        width_per_alg = max(4, min(8, 1.5 * max(len(by_algorithm[alg]) for alg in algorithms)))
        fig_width = min(24, max(10, width_per_alg * n_algs))
        fig_height = 6 + (max_model_len * 0.05)  # Extra height for rotated labels
        fig, axes = plt.subplots(1, n_algs, figsize=(fig_width, fig_height), squeeze=False, sharey=True)
        axes = axes[0]

        # Color mapping for models (consistent across facets)
        if n_models <= 10:
            model_colors = {m: plt.cm.tab10(i / 10) for i, m in enumerate(all_models)}
        elif n_models <= 20:
            model_colors = {m: plt.cm.tab20(i / 20) for i, m in enumerate(all_models)}
        else:
            model_colors = {m: plt.cm.viridis(0.1 + 0.8 * i / n_models) for i, m in enumerate(all_models)}

        for ax_idx, algorithm in enumerate(algorithms):
            ax = axes[ax_idx]
            alg_data = by_algorithm[algorithm]

            # Get models present in this algorithm
            models_in_alg = sorted(alg_data.keys())
            n_models_in_alg = len(models_in_alg)

            if n_models_in_alg == 0:
                ax.axis("off")
                continue

            x = np.arange(n_models_in_alg)
            width = 0.35

            top_diversity = [alg_data[m]["top_winners_diversity"] for m in models_in_alg]
            other_diversity = [alg_data[m]["other_winners_diversity"] for m in models_in_alg]

            # Use consistent colors per model
            bar_colors = [model_colors[m] for m in models_in_alg]

            ax.bar(x - width/2, top_diversity, width,
                  color=bar_colors, alpha=0.9, edgecolor='white', linewidth=0.5)
            ax.bar(x + width/2, other_diversity, width,
                  color=bar_colors, alpha=0.4, edgecolor='white', linewidth=0.5)

            ax.set_xticks(x)
            ax.set_xticklabels(models_in_alg, rotation=45, ha='right', fontsize=8)
            ax.grid(True, alpha=0.3, axis='y')
            ax.set_title(algorithm, fontweight='bold', fontsize=11)

            if ax_idx == 0:
                ax.set_ylabel('Cross-Winner Diversity')

        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='gray', alpha=0.9, label=f'Top {top_pct_str} Winners'),
            Patch(facecolor='gray', alpha=0.4, label='Other Winners'),
        ]
        fig.legend(handles=legend_elements, loc='upper center', ncol=2, fontsize=9,
                  bbox_to_anchor=(0.5, 0.02))

        problems_filter = data.get("problems_filter")
        prob_str = f' | {", ".join(problems_filter)}' if problems_filter else ""
        fig.suptitle(f'Cross-Winner Diversity: Top {top_pct_str} vs Other by Model + Algorithm{prob_str}',
                    fontweight='bold', fontsize=12, y=1.02)

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

        ax.bar(x - width/2, top_diversity, width, label=f'Top {top_pct_str} Winners',
               color=PALETTE[0], alpha=0.8)
        ax.bar(x + width/2, other_diversity, width, label='Other Winners',
               color=PALETTE[1], alpha=0.8)

        ax.set_ylabel('Cross-Winner Diversity')
        ax.set_xticks(x)

        # Always rotate labels for readability, use full names
        ax.set_xticklabels(group_names, rotation=45, ha='right', fontsize=9)

        ax.grid(True, alpha=0.3, axis='y')

        if group_by:
            group_label = {"model": "Model", "algorithm": "Algorithm"}[group_by]
            title = f'Cross-Winner Diversity: Top {top_pct_str} vs Other\nGrouped by {group_label}'
        else:
            problems_filter = data.get("problems_filter")
            if problems_filter and len(problems_filter) < 3:
                title = f'Cross-Winner Diversity: Top {top_pct_str} vs Other\n{", ".join(problems_filter)}'
            else:
                title = f'Cross-Winner Diversity: Top {top_pct_str} vs Other\nBy Problem'

        ax.set_title(title, fontweight='bold')

        if has_grouping:
            ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=9, framealpha=0.9)
        else:
            ax.legend(loc='upper right', fontsize=9)

        plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=150)
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


# ===========================================================================
# TAB 5: Mutation Factor Analysis
# ===========================================================================

def query_mutation_factors(
    database_url: str | None = None,
) -> list[dict[str, Any]]:
    """Query mutation data with all factors needed for factor analysis.

    Computes context diversity in SQL using pgvector to avoid loading all embeddings into memory.
    """
    url = database_url or _get_database_url()

    with psycopg.connect(url, row_factory=dict_row) as conn:
        with conn.cursor() as cur:
            # Compute all mutation factors in SQL, including context diversity using pgvector
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
                -- Context stats: count, max score, better/worse counts relative to parent
                context_stats AS (
                    SELECT
                        ce.child_id,
                        COUNT(ctx_m.value) as context_size,
                        MAX(ctx_m.value) as max_context_score,
                        SUM(CASE WHEN ctx_m.value > parent_m.value THEN 1 ELSE 0 END) as better_count,
                        SUM(CASE WHEN ctx_m.value < parent_m.value THEN 1 ELSE 0 END) as worse_count
                    FROM context_edges ce
                    JOIN parent_edges pe ON ce.child_id = pe.child_id
                    JOIN measurements ctx_m ON ctx_m.candidate_id = ce.context_id AND ctx_m.name = 'combined_score'
                    JOIN measurements parent_m ON parent_m.candidate_id = pe.parent_id AND parent_m.name = 'combined_score'
                    GROUP BY ce.child_id
                ),
                -- Context diversity: avg pairwise cosine distance among context candidates
                context_diversity AS (
                    SELECT
                        ce1.child_id,
                        AVG(c1.solution_summary_embedding <=> c2.solution_summary_embedding) as context_diversity
                    FROM context_edges ce1
                    JOIN context_edges ce2 ON ce1.child_id = ce2.child_id AND ce1.context_id < ce2.context_id
                    JOIN candidates c1 ON ce1.context_id = c1.id
                    JOIN candidates c2 ON ce2.context_id = c2.id
                    WHERE c1.solution_summary_embedding IS NOT NULL
                      AND c2.solution_summary_embedding IS NOT NULL
                    GROUP BY ce1.child_id
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
                    COALESCE(cs.better_count, 0) as better_count,
                    COALESCE(cs.worse_count, 0) as worse_count,
                    COALESCE(cd.context_diversity, 0) as context_diversity,
                    rb.best_before
                FROM candidates c
                JOIN campaigns camp ON c.campaign_id = camp.id
                JOIN parent_edges pe ON pe.child_id = c.id
                JOIN measurements child_m ON child_m.candidate_id = c.id AND child_m.name = 'combined_score'
                JOIN measurements parent_m ON parent_m.candidate_id = pe.parent_id AND parent_m.name = 'combined_score'
                LEFT JOIN context_stats cs ON cs.child_id = c.id
                LEFT JOIN context_diversity cd ON cd.child_id = c.id
                LEFT JOIN running_best rb ON rb.candidate_id = c.id
                WHERE c.solution_summary_embedding IS NOT NULL
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
        score_diversity = better_ratio + worse_ratio

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
                WHERE c.solution_summary_embedding IS NOT NULL
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
    "context_diversity": "Context Diversity (Semantic)",
    "score_diversity": "Context Score Diversity",
    "better_ratio": "Better Example Ratio",
    "worse_ratio": "Worse Example Ratio",
    "parent_score": "Parent Score",
}

# With global best improvement definition, parent_score is no longer confounded
FACTOR_KEYS = ["score_diversity", "context_diversity", "better_ratio", "worse_ratio", "parent_score"]


def compute_factor_importance(
    data: list[dict[str, Any]],
    group_by: str | None = None,
    problems_filter: list[str] | None = None,
) -> dict[str, Any]:
    """Compute factor importance for predicting improvement.

    Compares mean factor values between improvements (new global best) and
    non-improvements. Positive difference means the factor is higher when
    improvements occur, suggesting it's associated with success.

    Improvement is defined as: child_score > best_score_so_far (global best in campaign)

    Args:
        data: List of mutation records from query_mutation_factors()
        group_by: None (aggregate), "model", "algorithm", or "model_algorithm"
        problems_filter: Optional list of problems to include

    Returns:
        Dict with factor importance (mean difference) per group.
    """
    if problems_filter:
        data = [r for r in data if r["problem"] in problems_filter]

    if not data:
        return {"groups": [], "summary": {"total_mutations": 0}}

    # Split into improvements vs non-improvements (child > global best so far)
    improvements = [r for r in data if r["score_delta_global"] > 0]
    non_improvements = [r for r in data if r["score_delta_global"] <= 0]

    if not improvements or not non_improvements:
        return {"groups": [], "summary": {"total_mutations": len(data)}}

    # Group data
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

    # Group improvements and non-improvements separately
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

        # Need enough data in both groups
        if len(imp_data) < 5 or len(non_imp_data) < 5:
            continue

        # Compute normalized importance for each factor (z-score difference)
        # This makes factors comparable across different scales
        correlations = {}
        for factor in FACTOR_KEYS:
            imp_values = [r[factor] for r in imp_data]
            non_imp_values = [r[factor] for r in non_imp_data]

            # Combine all values to compute overall mean and std for normalization
            all_values = imp_values + non_imp_values
            overall_mean = float(np.mean(all_values))
            overall_std = float(np.std(all_values))

            if overall_std > 0:
                # Normalize both groups using overall stats, then compute difference
                imp_mean_zscore = (float(np.mean(imp_values)) - overall_mean) / overall_std
                non_imp_mean_zscore = (float(np.mean(non_imp_values)) - overall_mean) / overall_std
                importance = imp_mean_zscore - non_imp_mean_zscore
            else:
                importance = 0.0

            correlations[factor] = round(importance, 4)

        groups.append({
            "group": group_name,
            "n": len(imp_data) + len(non_imp_data),
            "n_improvements": len(imp_data),
            "n_non_improvements": len(non_imp_data),
            "correlations": correlations,
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


def _generate_factor_heatmap(
    groups: list[dict[str, Any]],
    group_by: str,
) -> bytes:
    """Generate heatmap for factor importance by model or model+algorithm.

    For model: single heatmap (factors x models)
    For model_algorithm: faceted heatmaps (one per algorithm)
    """
    if group_by == "model_algorithm":
        # Parse groups into algorithm -> {model -> correlations}
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
                "n": g.get("n", 0),
            }

        algorithms = sorted(by_algorithm.keys())
        n_algs = len(algorithms)

        if n_algs == 0:
            return _generate_empty_figure("No data for heatmap")

        # For each algorithm, only show models that have data for that algorithm
        # (skip models with empty/no correlations)
        alg_models: dict[str, list[str]] = {}
        for alg in algorithms:
            alg_data = by_algorithm[alg]
            # Only include models that have at least one correlation value
            models_with_data = [
                m for m in sorted(alg_data.keys())
                if alg_data[m].get("correlations")
            ]
            alg_models[alg] = models_with_data

        # Filter out algorithms with no models
        algorithms = [alg for alg in algorithms if alg_models[alg]]
        n_algs = len(algorithms)

        if n_algs == 0:
            return _generate_empty_figure("No data for heatmap")

        # Create faceted figure - calculate width based on number of models per algorithm
        max_models = max(len(alg_models[alg]) for alg in algorithms)
        fig_width = max(12, 3 * n_algs + max_models * 0.3)
        fig, axes = plt.subplots(1, n_algs, figsize=(fig_width, 5), squeeze=False)
        axes = axes[0]

        # Diverging colormap: red (negative) -> white (zero) -> green (positive)
        cmap = plt.cm.RdYlGn
        vmin, vmax = -0.5, 0.5

        im = None  # Will hold the last imshow for colorbar
        for idx, algorithm in enumerate(algorithms):
            ax = axes[idx]
            alg_data = by_algorithm[algorithm]
            models = alg_models[algorithm]

            if not models:
                ax.axis("off")
                continue

            # Build matrix: rows=factors, cols=models (only models with data)
            matrix = np.full((len(FACTOR_KEYS), len(models)), np.nan)
            for col_idx, model in enumerate(models):
                corrs = alg_data[model]["correlations"]
                for row_idx, factor in enumerate(FACTOR_KEYS):
                    if factor in corrs:
                        matrix[row_idx, col_idx] = corrs[factor]

            # Plot heatmap
            im = ax.imshow(matrix, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)

            # Labels
            ax.set_xticks(np.arange(len(models)))
            ax.set_xticklabels([m[:12] + "..." if len(m) > 12 else m for m in models],
                              rotation=45, ha="right", fontsize=7)
            ax.set_yticks(np.arange(len(FACTOR_KEYS)))
            if idx == 0:
                ax.set_yticklabels([FACTOR_NAMES[f] for f in FACTOR_KEYS], fontsize=9)
            else:
                ax.set_yticklabels([])

            # Annotate cells
            for i in range(len(FACTOR_KEYS)):
                for j in range(len(models)):
                    val = matrix[i, j]
                    if not np.isnan(val):
                        text_color = "white" if abs(val) > 0.3 else "black"
                        ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                               fontsize=7, color=text_color)

            ax.set_title(f"{algorithm}", fontweight="bold", fontsize=11)

        # Add colorbar to the right side with proper spacing
        if im is not None:
            cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
            fig.colorbar(im, cax=cbar_ax, orientation="vertical", label="Correlation")

        fig.suptitle("Factor Importance by Model + Algorithm", fontweight="bold", fontsize=12, y=0.98)
        plt.tight_layout(rect=[0, 0, 0.90, 0.95])  # Leave space for colorbar and title

    else:
        # Single heatmap for group_by=model
        models = [g["group"] for g in groups]

        # Build matrix: rows=factors, cols=models
        matrix = np.full((len(FACTOR_KEYS), len(models)), np.nan)
        for col_idx, g in enumerate(groups):
            corrs = g.get("correlations", {})
            for row_idx, factor in enumerate(FACTOR_KEYS):
                if factor in corrs:
                    matrix[row_idx, col_idx] = corrs[factor]

        fig, ax = plt.subplots(figsize=(max(8, len(models) * 0.6), 4))

        # Diverging colormap
        cmap = plt.cm.RdYlGn
        vmin, vmax = -0.5, 0.5

        im = ax.imshow(matrix, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)

        # Labels
        ax.set_xticks(np.arange(len(models)))
        ax.set_xticklabels([m[:15] + "..." if len(m) > 15 else m for m in models],
                          rotation=45, ha="right", fontsize=9)
        ax.set_yticks(np.arange(len(FACTOR_KEYS)))
        ax.set_yticklabels([FACTOR_NAMES[f] for f in FACTOR_KEYS], fontsize=10)

        # Annotate cells
        for i in range(len(FACTOR_KEYS)):
            for j in range(len(models)):
                val = matrix[i, j]
                if not np.isnan(val):
                    text_color = "white" if abs(val) > 0.3 else "black"
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                           fontsize=8, color=text_color)

        # Colorbar
        fig.colorbar(im, ax=ax, orientation="vertical", fraction=0.03, pad=0.04,
                    label="Correlation")

        total_n = sum(g.get("n", 0) for g in groups)
        ax.set_title(f"Factor Importance by Model (n={total_n} mutations)",
                    fontweight="bold")
        plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


def generate_factor_importance_figure(
    data: dict[str, Any],
) -> bytes:
    """Generate visualization for factor importance (correlations).

    - Aggregate/single group: horizontal bar chart
    - Group by algorithm: grouped bar chart (3 groups, colors work well)
    - Group by model: heatmap (12+ groups)
    - Group by model+algorithm: faceted heatmaps by algorithm (33+ groups)
    """
    groups = data.get("groups", [])
    if not groups:
        return _generate_empty_figure("No mutation factor data available")

    group_by = data.get("group_by")
    n_groups = len(groups)

    if group_by is None or n_groups == 1:
        # Single aggregate view - horizontal bar chart
        fig, ax = plt.subplots(figsize=(10, 5))
        group = groups[0]
        correlations = group.get("correlations", {})

        factors = []
        values = []
        colors = []

        for factor in FACTOR_KEYS:
            if factor in correlations:
                factors.append(FACTOR_NAMES[factor])
                val = correlations[factor]
                values.append(val)
                colors.append(COLORS["high"] if val > 0 else COLORS["low"])

        if not factors:
            return _generate_empty_figure("Insufficient data to compute correlations")

        y_pos = np.arange(len(factors))
        bars = ax.barh(y_pos, values, color=colors, alpha=0.8)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(factors)
        ax.set_xlabel("Correlation with Score Improvement")
        ax.axvline(x=0, color="gray", linestyle="-", linewidth=0.5)
        ax.grid(True, alpha=0.3, axis="x")

        # Add value labels
        for bar, val in zip(bars, values):
            x = bar.get_width()
            ha = "left" if x >= 0 else "right"
            offset = 0.02 if x >= 0 else -0.02
            ax.annotate(f"{val:.3f}",
                        xy=(x + offset, bar.get_y() + bar.get_height() / 2),
                        va="center", ha=ha, fontsize=9)

        ax.set_title(f"Factor Importance (n={group['n']} mutations)",
                     fontweight="bold")

        plt.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
        plt.close(fig)
        buf.seek(0)
        return buf.getvalue()

    elif group_by in ("model", "model_algorithm"):
        # Heatmap for high-cardinality groupings
        return _generate_factor_heatmap(groups, group_by)

    else:
        # Group by algorithm - grouped bar chart (small number of groups)
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
            ax.bar(x + offset, values, width, label=f"{label} (n={group['n']})",
                   color=PALETTE[i % len(PALETTE)], alpha=0.8)

        ax.set_xticks(x)
        ax.set_xticklabels([FACTOR_NAMES[f] for f in FACTOR_KEYS], rotation=15, ha="right")
        ax.set_ylabel("Correlation with Score Improvement")
        ax.axhline(y=0, color="gray", linestyle="-", linewidth=0.5)
        ax.grid(True, alpha=0.3, axis="y")
        ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8)

        ax.set_title("Factor Importance by Algorithm", fontweight="bold")

        plt.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
        plt.close(fig)
        buf.seek(0)
        return buf.getvalue()


# ---------------------------------------------------------------------------
# Q3 Factor Analysis: High-Level API
# ---------------------------------------------------------------------------

def get_factor_importance_figure(
    database_url: str | None = None,
    group_by: str | None = None,
    problems: list[str] | None = None,
) -> bytes:
    """Get factor importance bar chart as PNG."""
    raw_data = query_mutation_factors(database_url)
    importance_data = compute_factor_importance(raw_data, group_by, problems)
    return generate_factor_importance_figure(importance_data)


# ===========================================================================
# HIGH-LEVEL API: Tab 1 - Diversity vs Score
# ===========================================================================

def get_diversity_vs_score_scatter(
    database_url: str | None = None,
    group_by: str | None = None,
    problems: list[str] | None = None,
) -> dict[str, Any]:
    """Get diversity vs score scatter data.

    Args:
        database_url: Database connection URL
        group_by: None, "model", "algorithm", or "model_algorithm"
        problems: Optional list of problems to filter

    Returns:
        Scatter plot data with points, group stats, and correlation
    """
    campaigns = query_campaign_diversity_and_scores(database_url)
    return process_diversity_vs_score_scatter(campaigns, group_by, problems)


def get_diversity_vs_score_scatter_figure(
    database_url: str | None = None,
    group_by: str | None = None,
    problems: list[str] | None = None,
) -> bytes:
    """Get diversity vs score scatter figure as PNG.

    Args:
        database_url: Database connection URL
        group_by: None, "model", "algorithm", or "model_algorithm"
        problems: Optional list of problems to filter
    """
    data = get_diversity_vs_score_scatter(database_url, group_by, problems)
    return generate_diversity_vs_score_scatter_figure(data)


# ===========================================================================
# HIGH-LEVEL API: Tab 2 - Early Diversity
# ===========================================================================

def get_early_diversity_vs_outcome(
    database_url: str | None = None,
    early_fraction: float = 0.25,
) -> dict[str, Any]:
    """Get early diversity vs final outcome analysis.

    Args:
        database_url: Database connection URL
        early_fraction: Fraction of iterations to consider "early" (default 0.25)
    """
    return process_early_diversity_vs_outcome(
        early_fraction=early_fraction, database_url=database_url
    )


def get_early_diversity_scatter(
    database_url: str | None = None,
    early_fraction: float = 0.25,
    group_by: str | None = None,
    problems: list[str] | None = None,
) -> dict[str, Any]:
    """Get early diversity scatter data with grouping.

    Args:
        database_url: Database connection URL
        early_fraction: Fraction of iterations to consider "early" (default 0.25)
        group_by: None, "model", "algorithm", or "model_algorithm"
        problems: Optional list of problems to filter

    Returns:
        Scatter plot data with points, group stats, and correlation
    """
    return process_early_diversity_scatter(
        early_fraction=early_fraction, group_by=group_by, problems=problems, database_url=database_url
    )


def get_early_diversity_scatter_figure(
    database_url: str | None = None,
    early_fraction: float = 0.25,
    group_by: str | None = None,
    problems: list[str] | None = None,
) -> bytes:
    """Get early diversity scatter figure as PNG.

    Args:
        database_url: Database connection URL
        early_fraction: Fraction of iterations to consider "early" (default 0.25)
        group_by: None, "model", "algorithm", or "model_algorithm"
        problems: Optional list of problems to filter
    """
    data = get_early_diversity_scatter(database_url, early_fraction, group_by, problems)
    return generate_early_diversity_scatter_figure(data)


# ===========================================================================
# HIGH-LEVEL API: Tab 4 - Top-K Diversity
# ===========================================================================

def get_topk_winners_diversity(
    database_url: str | None = None,
    top_pct: float = 0.25,
    group_by: str | None = None,
    problems: list[str] | None = None,
) -> dict[str, Any]:
    """Get top winners diversity analysis.

    Computes diversity ACROSS the best solutions from top runs, answering:
    "Do top runs converge to similar approaches or find diverse solutions?"

    Args:
        database_url: Database connection URL
        top_pct: Fraction of campaigns to consider "top" (default 0.25 = top 25%)
        group_by: None, "model", "algorithm", or "model_algorithm"
        problems: Optional list of problems to filter
    """
    campaigns = query_campaigns_with_best_embeddings(database_url)
    return process_topk_winners_diversity(campaigns, top_pct, group_by, problems)


def get_topk_winners_diversity_figure(
    database_url: str | None = None,
    top_pct: float = 0.25,
    group_by: str | None = None,
    problems: list[str] | None = None,
) -> bytes:
    """Get top winners diversity figure as PNG.

    Args:
        database_url: Database connection URL
        top_pct: Fraction of campaigns to consider "top" (default 0.25 = top 25%)
        group_by: None, "model", "algorithm", or "model_algorithm"
        problems: Optional list of problems to filter
    """
    data = get_topk_winners_diversity(database_url, top_pct, group_by, problems)
    return generate_topk_winners_diversity_figure(data)


# ===========================================================================
# CLI Entry Point
# ===========================================================================

if __name__ == "__main__":
    import json
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--generate-figures":
        output_dir = sys.argv[2] if len(sys.argv) > 2 else "."
        get_all_diversity_by_group_figures(output_dir=output_dir)
    else:
        result = get_diversity_vs_score_scatter()
        print(json.dumps(result, indent=2, default=str))
