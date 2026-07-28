// Code Diversity analytics visualization (Q3: Code Diversity - Direct Code Embeddings)

let codeDiversityTab = 'scatter';  // 'scatter', 'early', 'importance', 'topk'
let codeScatterGroupBy = '';  // '' means no grouping (aggregated)
let codeScatterProblems = [];  // [] means all problems
let codeEarlyGroupBy = '';  // '' means no grouping for early diversity tab
let codeEarlyProblems = [];  // [] means all problems for early diversity tab
let codeTopkGroupBy = '';  // '' means no grouping for top-k tab
let codeTopkProblems = [];  // [] means all problems for top-k tab

// Factor analysis state
let codeFactorsGroupBy = '';  // '' means aggregate
let codeFactorsProblems = [];  // [] means no problems selected (show empty state)

// Available problems (fetched from API)
let availableCodeDiversityProblems = [];
let availableCodeFactorsProblems = [];

// Dirty state tracking for Apply buttons
let codeScatterDirty = false;
let codeEarlyDirty = false;
let codeFactorsDirty = false;
let codeTopkDirty = false;

function markCodeScatterDirty() {
    codeScatterDirty = true;
    updateCodeScatterApplyButton();
}

function updateCodeScatterApplyButton() {
    const btn = document.getElementById('codeScatterApplyBtn');
    if (btn) {
        btn.className = codeScatterDirty ? 'btn-apply dirty' : 'btn-apply';
        btn.textContent = codeScatterDirty ? 'Apply *' : 'Apply';
    }
}

function markCodeEarlyDirty() {
    codeEarlyDirty = true;
    updateCodeEarlyApplyButton();
}

function updateCodeEarlyApplyButton() {
    const btn = document.getElementById('codeEarlyApplyBtn');
    if (btn) {
        btn.className = codeEarlyDirty ? 'btn-apply dirty' : 'btn-apply';
        btn.textContent = codeEarlyDirty ? 'Apply *' : 'Apply';
    }
}

function markCodeFactorsDirty() {
    codeFactorsDirty = true;
    updateCodeFactorsApplyButton();
}

function updateCodeFactorsApplyButton() {
    const btn = document.getElementById('codeFactorsApplyBtn');
    if (btn) {
        btn.className = codeFactorsDirty ? 'btn-apply dirty' : 'btn-apply';
        btn.textContent = codeFactorsDirty ? 'Apply *' : 'Apply';
    }
}

function markCodeTopkDirty() {
    codeTopkDirty = true;
    updateCodeTopkApplyButton();
}

function updateCodeTopkApplyButton() {
    const btn = document.getElementById('codeTopkApplyBtn');
    if (btn) {
        btn.className = codeTopkDirty ? 'btn-apply dirty' : 'btn-apply';
        btn.textContent = codeTopkDirty ? 'Apply *' : 'Apply';
    }
}

function renderCodeDiversity() {
    const analyticsContent = document.getElementById('analyticsContent');

    let html = `
        <div class="diversity-description">
            <p><strong>Q3: Code Diversity Analysis (Direct Code Embeddings)</strong></p>
            <p>Measures code diversity using direct embeddings from jina-embeddings-v2-base-code (768-dim) applied to raw candidate code.</p>
        </div>

        <div class="diversity-tabs">
            <button class="tab-btn ${codeDiversityTab === 'scatter' ? 'active' : ''}" onclick="switchCodeDiversityTab('scatter')">
                Diversity vs Score
            </button>
            <button class="tab-btn ${codeDiversityTab === 'early' ? 'active' : ''}" onclick="switchCodeDiversityTab('early')">
                Early Diversity
            </button>
            <button class="tab-btn ${codeDiversityTab === 'importance' ? 'active' : ''}" onclick="switchCodeDiversityTab('importance')">
                Factor Importance
            </button>
            <button class="tab-btn ${codeDiversityTab === 'topk' ? 'active' : ''}" onclick="switchCodeDiversityTab('topk')">
                Top-K Diversity
            </button>
        </div>

        <div id="codeDiversityContent">
            <div class="loading">Loading...</div>
        </div>
    `;

    analyticsContent.innerHTML = html;
    loadCodeDiversityTab(codeDiversityTab);
}

async function switchCodeDiversityTab(tab) {
    codeDiversityTab = tab;

    // Update tab buttons
    document.querySelectorAll('.diversity-tabs .tab-btn').forEach(btn => {
        btn.classList.remove('active');
    });
    event.target.classList.add('active');

    loadCodeDiversityTab(tab);
}

async function loadCodeDiversityTab(tab) {
    const content = document.getElementById('codeDiversityContent');
    content.innerHTML = '<div class="loading">Loading...</div>';

    try {
        // Fetch available problems if not already loaded
        if (availableCodeDiversityProblems.length === 0) {
            await loadCodeDiversityProblems();
        }

        if (tab === 'scatter') {
            renderCodeScatterTab();
        } else if (tab === 'early') {
            renderCodeEarlyTab();
        } else if (tab === 'importance') {
            await loadCodeFactorsProblems();
        } else if (tab === 'topk') {
            renderCodeTopKTab();
        }
    } catch (error) {
        content.innerHTML = `<div class="message error" style="display:block;">Error: ${error.message}</div>`;
    }
}

async function loadCodeDiversityProblems() {
    try {
        const res = await fetch('/analytics/code-diversity/problems');
        if (!res.ok) throw new Error('Failed to fetch problems');
        const data = await res.json();
        availableCodeDiversityProblems = data.problems || [];
    } catch (error) {
        console.error('Error loading code diversity problems:', error);
        availableCodeDiversityProblems = [];
    }
}

// ===========================================================================
// Code Diversity vs Score Tab
// ===========================================================================

function renderCodeScatterTab() {
    const content = document.getElementById('codeDiversityContent');

    const problemCheckboxes = availableCodeDiversityProblems.map(p => {
        const safeId = p.replace(/[^a-zA-Z0-9]/g, '_');
        return `
            <label style="margin-right: 10px;">
                <input type="checkbox" id="code_prob_${safeId}" onchange="markCodeScatterDirty()" ${codeScatterProblems.includes(p) ? 'checked' : ''}>
                ${escapeHtml(p)}
            </label>
        `;
    }).join('');

    content.innerHTML = `
        <h3>Does Code Diversity Lead to Better Outcomes?</h3>
        <p class="section-description">Scatter plot showing correlation between run code diversity and final score. Color encodes model, marker shape and line style encode algorithm. Model+Algorithm grouping uses a faceted layout with one panel per algorithm.</p>

        <div class="variance-figure-container">
            <div class="figure-controls">
                <select id="codeScatterGroupBySelect" onchange="markCodeScatterDirty()">
                    <option value="" ${codeScatterGroupBy === '' ? 'selected' : ''}>No Grouping (Aggregated)</option>
                    <option value="algorithm" ${codeScatterGroupBy === 'algorithm' ? 'selected' : ''}>Group by Algorithm</option>
                    <option value="model" ${codeScatterGroupBy === 'model' ? 'selected' : ''}>Group by Model</option>
                    <option value="model_algorithm" ${codeScatterGroupBy === 'model_algorithm' ? 'selected' : ''}>Group by Model + Algorithm</option>
                </select>
                <button id="codeScatterApplyBtn" class="btn-apply" onclick="applyCodeScatterChanges()">Apply</button>
                <button class="btn-secondary" onclick="downloadCodeScatterFigure()">Download</button>
            </div>
            <div class="problem-checkboxes" style="margin: 10px 0;">
                <label style="margin-right: 15px; font-weight: bold;">Problems:</label>
                ${problemCheckboxes}
            </div>
            <div class="figure-wrapper" id="codeScatterFigureWrapper">
                <div class="empty-state">Select one or more problems above and click Apply to view code diversity analysis.</div>
            </div>
        </div>
    `;
}

function getCodeScatterFigureUrl() {
    let url = '/analytics/code-diversity/scatter/figure';
    const params = [];

    if (codeScatterGroupBy) {
        params.push(`group_by=${encodeURIComponent(codeScatterGroupBy)}`);
    }

    const selectedProblems = getSelectedCodeProblems();
    // Only pass problems param if filtering (not all selected)
    if (selectedProblems.length > 0 && selectedProblems.length < availableCodeDiversityProblems.length) {
        params.push(`problems=${encodeURIComponent(selectedProblems.join(','))}`);
    }

    if (params.length > 0) {
        url += '?' + params.join('&');
    }
    return url;
}

function getSelectedCodeProblems() {
    const problems = [];
    for (const p of availableCodeDiversityProblems) {
        const safeId = p.replace(/[^a-zA-Z0-9]/g, '_');
        const checkbox = document.getElementById(`code_prob_${safeId}`);
        if (checkbox && checkbox.checked) {
            problems.push(p);
        }
    }
    return problems;
}

function applyCodeScatterChanges() {
    codeScatterGroupBy = document.getElementById('codeScatterGroupBySelect').value;
    codeScatterProblems = getSelectedCodeProblems();
    codeScatterDirty = false;
    updateCodeScatterApplyButton();

    const wrapper = document.getElementById('codeScatterFigureWrapper');
    if (!wrapper) return;

    if (codeScatterProblems.length === 0) {
        wrapper.innerHTML = '<div class="empty-state">Select one or more problems above and click Apply to view code diversity analysis.</div>';
        return;
    }

    const figUrl = getCodeScatterFigureUrl();
    wrapper.innerHTML = `<img id="codeScatterFigure" src="${figUrl}" alt="Code Diversity vs Score Scatter" />`;
}

function downloadCodeScatterFigure() {
    const groupSuffix = codeScatterGroupBy ? `_by_${codeScatterGroupBy}` : '';
    const selectedProblems = getSelectedCodeProblems();
    const problemSuffix = selectedProblems.length > 0 && selectedProblems.length < 3
        ? `_${selectedProblems.join('_').toLowerCase()}`
        : '';
    const filename = `code_diversity_vs_score${groupSuffix}${problemSuffix}.png`;
    downloadCodeFigure(getCodeScatterFigureUrl(), filename);
}

// ===========================================================================
// Code Early Diversity Tab
// ===========================================================================

function renderCodeEarlyTab() {
    const content = document.getElementById('codeDiversityContent');

    const problemCheckboxes = availableCodeDiversityProblems.map(p => {
        const safeId = p.replace(/[^a-zA-Z0-9]/g, '_');
        return `
            <label style="margin-right: 10px;">
                <input type="checkbox" id="code_early_prob_${safeId}" onchange="markCodeEarlyDirty()" ${codeEarlyProblems.includes(p) ? 'checked' : ''}>
                ${escapeHtml(p)}
            </label>
        `;
    }).join('');

    content.innerHTML = `
        <h3>Does Early Code Diversity Predict Final Outcome?</h3>
        <p class="section-description">Tests whether code diversity in early iterations (first 25%) correlates with better final scores. Color encodes model, marker shape and line style encode algorithm. Model+Algorithm grouping uses a faceted layout with one panel per algorithm.</p>

        <div class="variance-figure-container">
            <div class="figure-controls">
                <select id="codeEarlyGroupBySelect" onchange="markCodeEarlyDirty()">
                    <option value="" ${codeEarlyGroupBy === '' ? 'selected' : ''}>No Grouping (Aggregated)</option>
                    <option value="algorithm" ${codeEarlyGroupBy === 'algorithm' ? 'selected' : ''}>Group by Algorithm</option>
                    <option value="model" ${codeEarlyGroupBy === 'model' ? 'selected' : ''}>Group by Model</option>
                    <option value="model_algorithm" ${codeEarlyGroupBy === 'model_algorithm' ? 'selected' : ''}>Group by Model + Algorithm</option>
                </select>
                <button id="codeEarlyApplyBtn" class="btn-apply" onclick="applyCodeEarlyChanges()">Apply</button>
                <button class="btn-secondary" onclick="downloadCodeEarlyFigure()">Download</button>
            </div>
            <div class="problem-checkboxes" style="margin: 10px 0;">
                <label style="margin-right: 15px; font-weight: bold;">Problems:</label>
                ${problemCheckboxes}
            </div>
            <div class="figure-wrapper" id="codeEarlyFigureWrapper">
                <div class="empty-state">Select one or more problems above and click Apply to view early code diversity analysis.</div>
            </div>
        </div>
    `;
}

function getCodeEarlyFigureUrl() {
    let url = '/analytics/code-diversity/early/figure';
    const params = [];

    if (codeEarlyGroupBy) {
        params.push(`group_by=${encodeURIComponent(codeEarlyGroupBy)}`);
    }

    const selectedProblems = getSelectedCodeEarlyProblems();
    // Only pass problems param if filtering (not all selected)
    if (selectedProblems.length > 0 && selectedProblems.length < availableCodeDiversityProblems.length) {
        params.push(`problems=${encodeURIComponent(selectedProblems.join(','))}`);
    }

    if (params.length > 0) {
        url += '?' + params.join('&');
    }
    return url;
}

function getSelectedCodeEarlyProblems() {
    const problems = [];
    for (const p of availableCodeDiversityProblems) {
        const safeId = p.replace(/[^a-zA-Z0-9]/g, '_');
        const checkbox = document.getElementById(`code_early_prob_${safeId}`);
        if (checkbox && checkbox.checked) {
            problems.push(p);
        }
    }
    return problems;
}

function applyCodeEarlyChanges() {
    codeEarlyGroupBy = document.getElementById('codeEarlyGroupBySelect').value;
    codeEarlyProblems = getSelectedCodeEarlyProblems();
    codeEarlyDirty = false;
    updateCodeEarlyApplyButton();

    const wrapper = document.getElementById('codeEarlyFigureWrapper');
    if (!wrapper) return;

    if (codeEarlyProblems.length === 0) {
        wrapper.innerHTML = '<div class="empty-state">Select one or more problems above and click Apply to view early code diversity analysis.</div>';
        return;
    }

    const figUrl = getCodeEarlyFigureUrl();
    wrapper.innerHTML = `<img id="codeEarlyFigure" src="${figUrl}" alt="Early Code Diversity vs Outcome" />`;
}

function downloadCodeEarlyFigure() {
    const groupSuffix = codeEarlyGroupBy ? `_by_${codeEarlyGroupBy}` : '';
    const selectedProblems = getSelectedCodeEarlyProblems();
    const problemSuffix = selectedProblems.length > 0 && selectedProblems.length < 3
        ? `_${selectedProblems.join('_').toLowerCase()}`
        : '';
    const filename = `early_code_diversity_vs_outcome${groupSuffix}${problemSuffix}.png`;
    downloadCodeFigure(getCodeEarlyFigureUrl(), filename);
}

// ===========================================================================
// Code Factor Analysis Tab (Importance)
// ===========================================================================

async function loadCodeFactorsProblems() {
    try {
        const res = await fetch('/analytics/code-factors/problems');
        if (!res.ok) throw new Error('Failed to fetch problems');
        const data = await res.json();
        availableCodeFactorsProblems = data.problems || [];
        renderCodeFactorsControls();
    } catch (error) {
        const content = document.getElementById('codeDiversityContent');
        content.innerHTML = `<div class="message error" style="display:block;">Error loading problems: ${error.message}</div>`;
    }
}

function renderCodeFactorsControls() {
    const content = document.getElementById('codeDiversityContent');

    let tabTitle = 'Which Factors Matter Most for Score Improvement?';
    let tabDescription = 'Bar chart showing normalized factor importance for predicting new global bests (using code embeddings). Values are z-score differences between improvement and non-improvement iterations.';

    content.innerHTML = `
        <h3>${tabTitle}</h3>
        <div class="variance-figure-container">
            <div class="figure-controls">
                <select id="codeFactorsGroupBySelect" onchange="markCodeFactorsDirty()">
                    <option value="" ${codeFactorsGroupBy === '' ? 'selected' : ''}>Aggregate</option>
                    <option value="model" ${codeFactorsGroupBy === 'model' ? 'selected' : ''}>Group by Model</option>
                    <option value="algorithm" ${codeFactorsGroupBy === 'algorithm' ? 'selected' : ''}>Group by Algorithm</option>
                    <option value="model_algorithm" ${codeFactorsGroupBy === 'model_algorithm' ? 'selected' : ''}>Group by Model + Algorithm</option>
                </select>
                <button id="codeFactorsApplyBtn" class="btn-apply" onclick="applyCodeFactorsChanges()">Apply</button>
                <button class="btn-secondary" onclick="downloadCodeFactorsFigure()">Download</button>
            </div>
            <p class="section-description">${tabDescription}</p>
            <div class="problem-checkboxes" style="margin: 10px 0;">
                <label style="margin-right: 15px; font-weight: bold;">Problems:</label>
                ${availableCodeFactorsProblems.map(p => `
                    <label style="margin-right: 10px;">
                        <input type="checkbox" id="code_factors_prob_${p.replace(/[^a-zA-Z0-9]/g, '_')}"
                               onchange="markCodeFactorsDirty()"
                               ${codeFactorsProblems.includes(p) ? 'checked' : ''}>
                        ${escapeHtml(p)}
                    </label>
                `).join('')}
            </div>
            <div class="figure-wrapper" id="codeFactorsFigureWrapper">
                <div class="empty-state">Select one or more problems above and click Apply to view factor analysis.</div>
            </div>
        </div>
    `;
}

function getSelectedCodeFactorsProblems() {
    const problems = [];
    for (const p of availableCodeFactorsProblems) {
        const checkbox = document.getElementById(`code_factors_prob_${p.replace(/[^a-zA-Z0-9]/g, '_')}`);
        if (checkbox && checkbox.checked) {
            problems.push(p);
        }
    }
    return problems;
}

function applyCodeFactorsChanges() {
    codeFactorsGroupBy = document.getElementById('codeFactorsGroupBySelect')?.value || '';
    codeFactorsProblems = getSelectedCodeFactorsProblems();

    codeFactorsDirty = false;
    updateCodeFactorsApplyButton();

    const wrapper = document.getElementById('codeFactorsFigureWrapper');
    if (!wrapper) return;

    if (codeFactorsProblems.length === 0) {
        wrapper.innerHTML = '<div class="empty-state">Select one or more problems above and click Apply to view factor analysis.</div>';
        return;
    }

    const figUrl = getCodeFactorsFigureUrl();
    wrapper.innerHTML = `
        <img id="codeFactorsFigure" src="${figUrl}" alt="Code Factor Analysis"
             onerror="this.parentElement.innerHTML='<div class=\\'empty-state\\'>Failed to load figure</div>'" />
    `;
}

function getCodeFactorsFigureUrl() {
    const params = [];

    if (codeFactorsGroupBy) {
        params.push(`group_by=${encodeURIComponent(codeFactorsGroupBy)}`);
    }

    if (codeFactorsProblems.length > 0) {
        params.push(`problems=${encodeURIComponent(codeFactorsProblems.join(','))}`);
    }

    let baseUrl = '/analytics/code-factors/importance/figure';

    if (params.length > 0) {
        return baseUrl + '?' + params.join('&');
    }
    return baseUrl;
}

function downloadCodeFactorsFigure() {
    if (codeFactorsProblems.length === 0) return;

    const link = document.createElement('a');
    link.href = getCodeFactorsFigureUrl();

    const groupSuffix = codeFactorsGroupBy ? `_by_${codeFactorsGroupBy}` : '';
    const problemSuffix = codeFactorsProblems.length < 3
        ? `_${codeFactorsProblems.join('_').replace(/[^a-zA-Z0-9_]/g, '').substring(0, 30)}`
        : '';

    link.download = `q3_code_factor_importance${groupSuffix}${problemSuffix}.png`;
    link.click();
}

// ===========================================================================
// Code Top-K Diversity Tab
// ===========================================================================

function renderCodeTopKTab() {
    const content = document.getElementById('codeDiversityContent');

    const problemCheckboxes = availableCodeDiversityProblems.map(p => {
        const safeId = p.replace(/[^a-zA-Z0-9]/g, '_');
        return `
            <label style="margin-right: 10px;">
                <input type="checkbox" id="code_topk_prob_${safeId}" onchange="markCodeTopkDirty()" ${codeTopkProblems.includes(p) ? 'checked' : ''}>
                ${escapeHtml(p)}
            </label>
        `;
    }).join('');

    content.innerHTML = `
        <h3>Are Top Winners Diverse or Converged? (Code Embeddings)</h3>
        <p class="section-description">Compares code diversity <strong>across</strong> winning solutions from top-10% runs vs other runs. Only includes runs that beat baseline (score > 0). Measures whether top performers converge to similar code (low diversity) or find diverse solutions (high diversity).</p>

        <div class="variance-figure-container">
            <div class="figure-controls">
                <select id="codeTopkGroupBySelect" onchange="markCodeTopkDirty()">
                    <option value="" ${codeTopkGroupBy === '' ? 'selected' : ''}>No Grouping (By Problem)</option>
                    <option value="algorithm" ${codeTopkGroupBy === 'algorithm' ? 'selected' : ''}>Group by Algorithm</option>
                    <option value="model" ${codeTopkGroupBy === 'model' ? 'selected' : ''}>Group by Model</option>
                    <option value="model_algorithm" ${codeTopkGroupBy === 'model_algorithm' ? 'selected' : ''}>Group by Model + Algorithm</option>
                </select>
                <button id="codeTopkApplyBtn" class="btn-apply" onclick="applyCodeTopkChanges()">Apply</button>
                <button class="btn-secondary" onclick="downloadCodeTopKFigure()">Download</button>
            </div>
            <div class="problem-checkboxes" style="margin: 10px 0;">
                <label style="margin-right: 15px; font-weight: bold;">Problems:</label>
                ${problemCheckboxes}
            </div>
            <div class="figure-wrapper" id="codeTopkFigureWrapper">
                <div class="empty-state">Select one or more problems above and click Apply to view top-K code diversity analysis.</div>
            </div>
        </div>
    `;
}

function getCodeTopKFigureUrl() {
    let url = '/analytics/code-diversity/topk/figure';
    const params = ['top_pct=0.10'];

    if (codeTopkGroupBy) {
        params.push(`group_by=${encodeURIComponent(codeTopkGroupBy)}`);
    }

    const selectedProblems = getSelectedCodeTopKProblems();
    // Only pass problems param if filtering (not all selected)
    if (selectedProblems.length > 0 && selectedProblems.length < availableCodeDiversityProblems.length) {
        params.push(`problems=${encodeURIComponent(selectedProblems.join(','))}`);
    }

    url += '?' + params.join('&');
    return url;
}

function getSelectedCodeTopKProblems() {
    const problems = [];
    for (const p of availableCodeDiversityProblems) {
        const safeId = p.replace(/[^a-zA-Z0-9]/g, '_');
        const checkbox = document.getElementById(`code_topk_prob_${safeId}`);
        if (checkbox && checkbox.checked) {
            problems.push(p);
        }
    }
    return problems;
}

function applyCodeTopkChanges() {
    codeTopkGroupBy = document.getElementById('codeTopkGroupBySelect').value;
    codeTopkProblems = getSelectedCodeTopKProblems();
    codeTopkDirty = false;
    updateCodeTopkApplyButton();

    const wrapper = document.getElementById('codeTopkFigureWrapper');
    if (!wrapper) return;

    if (codeTopkProblems.length === 0) {
        wrapper.innerHTML = '<div class="empty-state">Select one or more problems above and click Apply to view top-K code diversity analysis.</div>';
        return;
    }

    const figUrl = getCodeTopKFigureUrl();
    wrapper.innerHTML = `<img id="codeTopkFigure" src="${figUrl}" alt="Top-K Winners Code Diversity" />`;
}

function downloadCodeTopKFigure() {
    const groupSuffix = codeTopkGroupBy ? `_by_${codeTopkGroupBy}` : '';
    const selectedProblems = getSelectedCodeTopKProblems();
    const problemSuffix = selectedProblems.length > 0 && selectedProblems.length < 3
        ? `_${selectedProblems.join('_').toLowerCase()}`
        : '';
    const filename = `topk_winners_code_diversity${groupSuffix}${problemSuffix}.png`;
    downloadCodeFigure(getCodeTopKFigureUrl(), filename);
}

// ===========================================================================
// Utility Functions
// ===========================================================================

function downloadCodeFigure(url, filename) {
    const link = document.createElement('a');
    link.href = url;
    link.download = filename;
    link.click();
}
