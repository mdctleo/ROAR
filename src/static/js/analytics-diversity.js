// Diversity analytics visualization (Q3: Solution Diversity)

let diversityTab = 'scatter';  // 'scatter', 'early', 'importance', 'topk'
let scatterGroupBy = '';  // '' means no grouping (aggregated)
let scatterProblems = [];  // [] means all problems
let earlyGroupBy = '';  // '' means no grouping for early diversity tab
let earlyProblems = [];  // [] means all problems for early diversity tab
let topkGroupBy = '';  // '' means no grouping for top-k tab
let topkProblems = [];  // [] means all problems for top-k tab

// New factor analysis state
let factorsGroupBy = '';  // '' means aggregate
let factorsProblems = [];  // [] means no problems selected (show empty state)
let availableFactorsProblems = [];

// Dirty state tracking for Apply buttons
let scatterDirty = false;
let earlyDirty = false;
let factorsDirty = false;
let topkDirty = false;

function markScatterDirty() {
    scatterDirty = true;
    updateScatterApplyButton();
}

function updateScatterApplyButton() {
    const btn = document.getElementById('scatterApplyBtn');
    if (btn) {
        btn.className = scatterDirty ? 'btn-apply dirty' : 'btn-apply';
        btn.textContent = scatterDirty ? 'Apply *' : 'Apply';
    }
}

function markEarlyDirty() {
    earlyDirty = true;
    updateEarlyApplyButton();
}

function updateEarlyApplyButton() {
    const btn = document.getElementById('earlyApplyBtn');
    if (btn) {
        btn.className = earlyDirty ? 'btn-apply dirty' : 'btn-apply';
        btn.textContent = earlyDirty ? 'Apply *' : 'Apply';
    }
}

function markFactorsDirty() {
    factorsDirty = true;
    updateFactorsApplyButton();
}

function updateFactorsApplyButton() {
    const btn = document.getElementById('factorsApplyBtn');
    if (btn) {
        btn.className = factorsDirty ? 'btn-apply dirty' : 'btn-apply';
        btn.textContent = factorsDirty ? 'Apply *' : 'Apply';
    }
}

function markTopkDirty() {
    topkDirty = true;
    updateTopkApplyButton();
}

function updateTopkApplyButton() {
    const btn = document.getElementById('topkApplyBtn');
    if (btn) {
        btn.className = topkDirty ? 'btn-apply dirty' : 'btn-apply';
        btn.textContent = topkDirty ? 'Apply *' : 'Apply';
    }
}

function renderDiversity() {
    const analyticsContent = document.getElementById('analyticsContent');

    let html = `
        <div class="diversity-description">
            <p><strong>Q3: Solution Diversity Analysis</strong></p>
            <p>Measures algorithmic diversity using LLM-generated summaries of each candidate's approach.</p>
        </div>

        <div class="diversity-tabs">
            <button class="tab-btn ${diversityTab === 'scatter' ? 'active' : ''}" onclick="switchDiversityTab('scatter')">
                Diversity vs Score
            </button>
            <button class="tab-btn ${diversityTab === 'early' ? 'active' : ''}" onclick="switchDiversityTab('early')">
                Early Diversity
            </button>
            <button class="tab-btn ${diversityTab === 'importance' ? 'active' : ''}" onclick="switchDiversityTab('importance')">
                Factor Importance
            </button>
            <button class="tab-btn ${diversityTab === 'topk' ? 'active' : ''}" onclick="switchDiversityTab('topk')">
                Top-K Diversity
            </button>
        </div>

        <div id="diversityContent">
            <div class="loading">Loading...</div>
        </div>
    `;

    analyticsContent.innerHTML = html;
    loadDiversityTab(diversityTab);
}

async function switchDiversityTab(tab) {
    diversityTab = tab;

    // Update tab buttons
    document.querySelectorAll('.diversity-tabs .tab-btn').forEach(btn => {
        btn.classList.remove('active');
    });
    event.target.classList.add('active');

    loadDiversityTab(tab);
}

async function loadDiversityTab(tab) {
    const content = document.getElementById('diversityContent');
    content.innerHTML = '<div class="loading">Loading...</div>';

    try {
        if (tab === 'scatter') {
            renderScatterTab();
        } else if (tab === 'early') {
            renderEarlyTab();
        } else if (tab === 'importance') {
            await loadFactorsProblems();
        } else if (tab === 'topk') {
            renderTopKTab();
        }
    } catch (error) {
        content.innerHTML = `<div class="message error" style="display:block;">Error: ${error.message}</div>`;
    }
}

// ===========================================================================
// Diversity vs Score Tab
// ===========================================================================

function renderScatterTab() {
    const content = document.getElementById('diversityContent');
    const figureUrl = getScatterFigureUrl();

    content.innerHTML = `
        <h3>Does Diversity Lead to Better Outcomes?</h3>
        <p class="section-description">Scatter plot showing correlation between run diversity and final score. Color encodes model, marker shape and line style encode algorithm. Model+Algorithm grouping uses a faceted layout with one panel per algorithm.</p>

        <div class="variance-figure-container">
            <div class="figure-controls">
                <select id="scatterGroupBySelect" onchange="markScatterDirty()">
                    <option value="" ${scatterGroupBy === '' ? 'selected' : ''}>No Grouping (Aggregated)</option>
                    <option value="algorithm" ${scatterGroupBy === 'algorithm' ? 'selected' : ''}>Group by Algorithm</option>
                    <option value="model" ${scatterGroupBy === 'model' ? 'selected' : ''}>Group by Model</option>
                    <option value="model_algorithm" ${scatterGroupBy === 'model_algorithm' ? 'selected' : ''}>Group by Model + Algorithm</option>
                </select>
                <button id="scatterApplyBtn" class="btn-apply" onclick="applyScatterChanges()">Apply</button>
                <button class="btn-secondary" onclick="downloadScatterFigure()">Download</button>
            </div>
            <div class="problem-checkboxes" style="margin: 10px 0;">
                <label style="margin-right: 15px; font-weight: bold;">Problems:</label>
                <label style="margin-right: 10px;">
                    <input type="checkbox" id="prob_Knapsack" onchange="markScatterDirty()" ${scatterProblems.includes('Knapsack') ? 'checked' : ''}>
                    Knapsack
                </label>
                <label style="margin-right: 10px;">
                    <input type="checkbox" id="prob_Palindrome" onchange="markScatterDirty()" ${scatterProblems.includes('Palindrome') ? 'checked' : ''}>
                    Palindrome
                </label>
                <label style="margin-right: 10px;">
                    <input type="checkbox" id="prob_Polyomino" onchange="markScatterDirty()" ${scatterProblems.includes('Polyomino') ? 'checked' : ''}>
                    Polyomino
                </label>
            </div>
            <div class="figure-wrapper" id="scatterFigureWrapper">
                <div class="empty-state">Select one or more problems above and click Apply to view diversity analysis.</div>
            </div>
        </div>
    `;
}

function getScatterFigureUrl() {
    let url = '/analytics/diversity/scatter/figure';
    const params = [];

    if (scatterGroupBy) {
        params.push(`group_by=${encodeURIComponent(scatterGroupBy)}`);
    }

    const selectedProblems = getSelectedProblems();
    if (selectedProblems.length > 0 && selectedProblems.length < 3) {
        params.push(`problems=${encodeURIComponent(selectedProblems.join(','))}`);
    }

    if (params.length > 0) {
        url += '?' + params.join('&');
    }
    return url;
}

function getSelectedProblems() {
    const problems = [];
    const knapsack = document.getElementById('prob_Knapsack');
    const palindrome = document.getElementById('prob_Palindrome');
    const polyomino = document.getElementById('prob_Polyomino');

    if (knapsack && knapsack.checked) problems.push('Knapsack');
    if (palindrome && palindrome.checked) problems.push('Palindrome');
    if (polyomino && polyomino.checked) problems.push('Polyomino');

    return problems;
}

function applyScatterChanges() {
    scatterGroupBy = document.getElementById('scatterGroupBySelect').value;
    scatterProblems = getSelectedProblems();
    scatterDirty = false;
    updateScatterApplyButton();

    const wrapper = document.getElementById('scatterFigureWrapper');
    if (!wrapper) return;

    if (scatterProblems.length === 0) {
        wrapper.innerHTML = '<div class="empty-state">Select one or more problems above and click Apply to view diversity analysis.</div>';
        return;
    }

    const figUrl = getScatterFigureUrl();
    wrapper.innerHTML = `<img id="scatterFigure" src="${figUrl}" alt="Diversity vs Score Scatter" />`;
}

function downloadScatterFigure() {
    const groupSuffix = scatterGroupBy ? `_by_${scatterGroupBy}` : '';
    const selectedProblems = getSelectedProblems();
    const problemSuffix = selectedProblems.length > 0 && selectedProblems.length < 3
        ? `_${selectedProblems.join('_').toLowerCase()}`
        : '';
    const filename = `diversity_vs_score${groupSuffix}${problemSuffix}.png`;
    downloadFigure(getScatterFigureUrl(), filename);
}

// ===========================================================================
// Early Diversity Tab
// ===========================================================================

function renderEarlyTab() {
    const content = document.getElementById('diversityContent');
    const figureUrl = getEarlyFigureUrl();

    content.innerHTML = `
        <h3>Does Early Diversity Predict Final Outcome?</h3>
        <p class="section-description">Tests whether diversity in early iterations (first 25%) correlates with better final scores. Color encodes model, marker shape and line style encode algorithm. Model+Algorithm grouping uses a faceted layout with one panel per algorithm.</p>

        <div class="variance-figure-container">
            <div class="figure-controls">
                <select id="earlyGroupBySelect" onchange="markEarlyDirty()">
                    <option value="" ${earlyGroupBy === '' ? 'selected' : ''}>No Grouping (Aggregated)</option>
                    <option value="algorithm" ${earlyGroupBy === 'algorithm' ? 'selected' : ''}>Group by Algorithm</option>
                    <option value="model" ${earlyGroupBy === 'model' ? 'selected' : ''}>Group by Model</option>
                    <option value="model_algorithm" ${earlyGroupBy === 'model_algorithm' ? 'selected' : ''}>Group by Model + Algorithm</option>
                </select>
                <button id="earlyApplyBtn" class="btn-apply" onclick="applyEarlyChanges()">Apply</button>
                <button class="btn-secondary" onclick="downloadEarlyFigure()">Download</button>
            </div>
            <div class="problem-checkboxes" style="margin: 10px 0;">
                <label style="margin-right: 15px; font-weight: bold;">Problems:</label>
                <label style="margin-right: 10px;">
                    <input type="checkbox" id="early_prob_Knapsack" onchange="markEarlyDirty()" ${earlyProblems.includes('Knapsack') ? 'checked' : ''}>
                    Knapsack
                </label>
                <label style="margin-right: 10px;">
                    <input type="checkbox" id="early_prob_Palindrome" onchange="markEarlyDirty()" ${earlyProblems.includes('Palindrome') ? 'checked' : ''}>
                    Palindrome
                </label>
                <label style="margin-right: 10px;">
                    <input type="checkbox" id="early_prob_Polyomino" onchange="markEarlyDirty()" ${earlyProblems.includes('Polyomino') ? 'checked' : ''}>
                    Polyomino
                </label>
            </div>
            <div class="figure-wrapper" id="earlyFigureWrapper">
                <div class="empty-state">Select one or more problems above and click Apply to view early diversity analysis.</div>
            </div>
        </div>
    `;
}

function getEarlyFigureUrl() {
    let url = '/analytics/diversity/early/figure';
    const params = [];

    if (earlyGroupBy) {
        params.push(`group_by=${encodeURIComponent(earlyGroupBy)}`);
    }

    const selectedProblems = getSelectedEarlyProblems();
    if (selectedProblems.length > 0 && selectedProblems.length < 3) {
        params.push(`problems=${encodeURIComponent(selectedProblems.join(','))}`);
    }

    if (params.length > 0) {
        url += '?' + params.join('&');
    }
    return url;
}

function getSelectedEarlyProblems() {
    const problems = [];
    const knapsack = document.getElementById('early_prob_Knapsack');
    const palindrome = document.getElementById('early_prob_Palindrome');
    const polyomino = document.getElementById('early_prob_Polyomino');

    if (knapsack && knapsack.checked) problems.push('Knapsack');
    if (palindrome && palindrome.checked) problems.push('Palindrome');
    if (polyomino && polyomino.checked) problems.push('Polyomino');

    return problems;
}

function applyEarlyChanges() {
    earlyGroupBy = document.getElementById('earlyGroupBySelect').value;
    earlyProblems = getSelectedEarlyProblems();
    earlyDirty = false;
    updateEarlyApplyButton();

    const wrapper = document.getElementById('earlyFigureWrapper');
    if (!wrapper) return;

    if (earlyProblems.length === 0) {
        wrapper.innerHTML = '<div class="empty-state">Select one or more problems above and click Apply to view early diversity analysis.</div>';
        return;
    }

    const figUrl = getEarlyFigureUrl();
    wrapper.innerHTML = `<img id="earlyFigure" src="${figUrl}" alt="Early Diversity vs Outcome" />`;
}

function downloadEarlyFigure() {
    const groupSuffix = earlyGroupBy ? `_by_${earlyGroupBy}` : '';
    const selectedProblems = getSelectedEarlyProblems();
    const problemSuffix = selectedProblems.length > 0 && selectedProblems.length < 3
        ? `_${selectedProblems.join('_').toLowerCase()}`
        : '';
    const filename = `early_diversity_vs_outcome${groupSuffix}${problemSuffix}.png`;
    downloadFigure(getEarlyFigureUrl(), filename);
}

// ===========================================================================
// Factor Analysis Tabs (Importance, Deep Dive, Bad Example)
// ===========================================================================

async function loadFactorsProblems() {
    try {
        const res = await fetch('/analytics/factors/problems');
        if (!res.ok) throw new Error('Failed to fetch problems');
        const data = await res.json();
        availableFactorsProblems = data.problems || [];
        renderFactorsControls();
    } catch (error) {
        const content = document.getElementById('diversityContent');
        content.innerHTML = `<div class="message error" style="display:block;">Error loading problems: ${error.message}</div>`;
    }
}

function renderFactorsControls() {
    const content = document.getElementById('diversityContent');

    let tabTitle = '';
    let tabDescription = '';
    let extraControls = '';

    if (diversityTab === 'importance') {
        tabTitle = 'Which Factors Matter Most for Score Improvement?';
        tabDescription = 'Bar chart showing normalized factor importance for predicting new global bests. Values are z-score differences between improvement and non-improvement iterations.';
    }

    content.innerHTML = `
        <h3>${tabTitle}</h3>
        <div class="variance-figure-container">
            <div class="figure-controls">
                <select id="factorsGroupBySelect" onchange="markFactorsDirty()">
                    <option value="" ${factorsGroupBy === '' ? 'selected' : ''}>Aggregate</option>
                    <option value="model" ${factorsGroupBy === 'model' ? 'selected' : ''}>Group by Model</option>
                    <option value="algorithm" ${factorsGroupBy === 'algorithm' ? 'selected' : ''}>Group by Algorithm</option>
                    <option value="model_algorithm" ${factorsGroupBy === 'model_algorithm' ? 'selected' : ''}>Group by Model + Algorithm</option>
                </select>
                ${extraControls}
                <button id="factorsApplyBtn" class="btn-apply" onclick="applyFactorsChanges()">Apply</button>
                <button class="btn-secondary" onclick="downloadFactorsFigure()">Download</button>
            </div>
            <p class="section-description">${tabDescription}</p>
            <div class="problem-checkboxes" style="margin: 10px 0;">
                <label style="margin-right: 15px; font-weight: bold;">Problems:</label>
                ${availableFactorsProblems.map(p => `
                    <label style="margin-right: 10px;">
                        <input type="checkbox" id="factors_prob_${p.replace(/[^a-zA-Z0-9]/g, '_')}"
                               onchange="markFactorsDirty()"
                               ${factorsProblems.includes(p) ? 'checked' : ''}>
                        ${escapeHtml(p)}
                    </label>
                `).join('')}
            </div>
            <div class="figure-wrapper" id="factorsFigureWrapper">
                <div class="empty-state">Select one or more problems above and click Apply to view factor analysis.</div>
            </div>
        </div>
    `;
}

function getSelectedFactorsProblems() {
    const problems = [];
    for (const p of availableFactorsProblems) {
        const checkbox = document.getElementById(`factors_prob_${p.replace(/[^a-zA-Z0-9]/g, '_')}`);
        if (checkbox && checkbox.checked) {
            problems.push(p);
        }
    }
    return problems;
}

function applyFactorsChanges() {
    factorsGroupBy = document.getElementById('factorsGroupBySelect')?.value || '';
    factorsProblems = getSelectedFactorsProblems();

    factorsDirty = false;
    updateFactorsApplyButton();

    const wrapper = document.getElementById('factorsFigureWrapper');
    if (!wrapper) return;

    if (factorsProblems.length === 0) {
        wrapper.innerHTML = '<div class="empty-state">Select one or more problems above and click Apply to view factor analysis.</div>';
        return;
    }

    const figUrl = getFactorsFigureUrl();
    wrapper.innerHTML = `
        <img id="factorsFigure" src="${figUrl}" alt="Factor Analysis"
             onerror="this.parentElement.innerHTML='<div class=\\'empty-state\\'>Failed to load figure</div>'" />
    `;
}

function getFactorsFigureUrl() {
    const params = [];

    if (factorsGroupBy) {
        params.push(`group_by=${encodeURIComponent(factorsGroupBy)}`);
    }

    if (factorsProblems.length > 0) {
        params.push(`problems=${encodeURIComponent(factorsProblems.join(','))}`);
    }

    let baseUrl = '/analytics/factors/importance/figure';

    if (params.length > 0) {
        return baseUrl + '?' + params.join('&');
    }
    return baseUrl;
}

function downloadFactorsFigure() {
    if (factorsProblems.length === 0) return;

    const link = document.createElement('a');
    link.href = getFactorsFigureUrl();

    const groupSuffix = factorsGroupBy ? `_by_${factorsGroupBy}` : '';
    const problemSuffix = factorsProblems.length < 3
        ? `_${factorsProblems.join('_').replace(/[^a-zA-Z0-9_]/g, '').substring(0, 30)}`
        : '';

    link.download = `q3_factor_importance${groupSuffix}${problemSuffix}.png`;
    link.click();
}

// ===========================================================================
// Top-K Diversity Tab
// ===========================================================================

function renderTopKTab() {
    const content = document.getElementById('diversityContent');
    const figureUrl = getTopKFigureUrl();

    content.innerHTML = `
        <h3>Are Top Winners Diverse or Converged?</h3>
        <p class="section-description">Compares diversity <strong>across</strong> winning solutions from top-10 runs vs other runs (excluding top-10). Only includes runs that beat baseline (score > 0). Measures whether top performers converge to similar approaches (low diversity) or find diverse solutions (high diversity).</p>

        <div class="variance-figure-container">
            <div class="figure-controls">
                <select id="topkGroupBySelect" onchange="markTopkDirty()">
                    <option value="" ${topkGroupBy === '' ? 'selected' : ''}>No Grouping (By Problem)</option>
                    <option value="algorithm" ${topkGroupBy === 'algorithm' ? 'selected' : ''}>Group by Algorithm</option>
                    <option value="model" ${topkGroupBy === 'model' ? 'selected' : ''}>Group by Model</option>
                    <option value="model_algorithm" ${topkGroupBy === 'model_algorithm' ? 'selected' : ''}>Group by Model + Algorithm</option>
                </select>
                <button id="topkApplyBtn" class="btn-apply" onclick="applyTopkChanges()">Apply</button>
                <button class="btn-secondary" onclick="downloadTopKFigure()">Download</button>
            </div>
            <div class="problem-checkboxes" style="margin: 10px 0;">
                <label style="margin-right: 15px; font-weight: bold;">Problems:</label>
                <label style="margin-right: 10px;">
                    <input type="checkbox" id="topk_prob_Knapsack" onchange="markTopkDirty()" ${topkProblems.includes('Knapsack') ? 'checked' : ''}>
                    Knapsack
                </label>
                <label style="margin-right: 10px;">
                    <input type="checkbox" id="topk_prob_Palindrome" onchange="markTopkDirty()" ${topkProblems.includes('Palindrome') ? 'checked' : ''}>
                    Palindrome
                </label>
                <label style="margin-right: 10px;">
                    <input type="checkbox" id="topk_prob_Polyomino" onchange="markTopkDirty()" ${topkProblems.includes('Polyomino') ? 'checked' : ''}>
                    Polyomino
                </label>
            </div>
            <div class="figure-wrapper" id="topkFigureWrapper">
                <div class="empty-state">Select one or more problems above and click Apply to view top-K diversity analysis.</div>
            </div>
        </div>
    `;
}

function getTopKFigureUrl() {
    let url = '/analytics/diversity/topk/figure';
    const params = [];

    if (topkGroupBy) {
        params.push(`group_by=${encodeURIComponent(topkGroupBy)}`);
    }

    const selectedProblems = getSelectedTopKProblems();
    if (selectedProblems.length > 0 && selectedProblems.length < 3) {
        params.push(`problems=${encodeURIComponent(selectedProblems.join(','))}`);
    }

    if (params.length > 0) {
        url += '?' + params.join('&');
    }
    return url;
}

function getSelectedTopKProblems() {
    const problems = [];
    const knapsack = document.getElementById('topk_prob_Knapsack');
    const palindrome = document.getElementById('topk_prob_Palindrome');
    const polyomino = document.getElementById('topk_prob_Polyomino');

    if (knapsack && knapsack.checked) problems.push('Knapsack');
    if (palindrome && palindrome.checked) problems.push('Palindrome');
    if (polyomino && polyomino.checked) problems.push('Polyomino');

    return problems;
}

function applyTopkChanges() {
    topkGroupBy = document.getElementById('topkGroupBySelect').value;
    topkProblems = getSelectedTopKProblems();
    topkDirty = false;
    updateTopkApplyButton();

    const wrapper = document.getElementById('topkFigureWrapper');
    if (!wrapper) return;

    if (topkProblems.length === 0) {
        wrapper.innerHTML = '<div class="empty-state">Select one or more problems above and click Apply to view top-K diversity analysis.</div>';
        return;
    }

    const figUrl = getTopKFigureUrl();
    wrapper.innerHTML = `<img id="topkFigure" src="${figUrl}" alt="Top-K Winners Diversity" />`;
}

function downloadTopKFigure() {
    const groupSuffix = topkGroupBy ? `_by_${topkGroupBy}` : '';
    const selectedProblems = getSelectedTopKProblems();
    const problemSuffix = selectedProblems.length > 0 && selectedProblems.length < 3
        ? `_${selectedProblems.join('_').toLowerCase()}`
        : '';
    const filename = `topk_winners_diversity${groupSuffix}${problemSuffix}.png`;
    downloadFigure(getTopKFigureUrl(), filename);
}

// ===========================================================================
// Utility Functions
// ===========================================================================

function downloadFigure(url, filename) {
    const link = document.createElement('a');
    link.href = url;
    link.download = filename;
    link.click();
}
