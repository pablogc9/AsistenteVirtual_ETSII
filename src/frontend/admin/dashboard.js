const token = localStorage.getItem('admin_token');
if (!token) window.location.href = 'login.html';

let intentChart = null;

async function fetchStats() {
    const response = await fetch('http://127.0.0.1:8000/admin/stats', {
        headers: { 'Authorization': `Bearer ${token}` }
    });

    if (response.status === 401) {
        logout();
        return;
    }

    const stats = await response.json();

    document.getElementById('total-interactions').innerText = stats.total_interactions;
    document.getElementById('total-tokens').innerText       = stats.total_tokens;

    const feedbackText = stats.avg_feedback !== null
        ? (stats.avg_feedback * 100).toFixed(1) + '%'
        : 'Sin datos';
    document.getElementById('avg-feedback').innerText    = feedbackText;
    document.getElementById('safe-percentage').innerText = stats.safe_percentage + '%';

    if (intentChart) intentChart.destroy();

    const INTENT_COLORS = {
        academica: '#0072bc',
        saludo:    '#ffc107',
        malicioso: '#dc3545',
    };

    const labels = Object.keys(stats.intent_distribution);
    const ctx = document.getElementById('intentChart').getContext('2d');
    intentChart = new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels,
            datasets: [{
                data:            labels.map(l => stats.intent_distribution[l]),
                backgroundColor: labels.map(l => INTENT_COLORS[l] || '#adb5bd'),
            }]
        },
        options: {
            plugins: {
                legend: { position: 'bottom' }
            },
            layout: { padding: 8 },
            maintainAspectRatio: true,
            aspectRatio: 1.8,
        }
    });
}

function logout() {
    localStorage.removeItem('admin_token');
    window.location.href = 'login.html';
}

// ---------------------------------------------------------------------------
// Estado de la tabla de logs
// ---------------------------------------------------------------------------

const logsState = {
    page:     1,
    pageSize: 10,
    intent:   '',
    feedback: '',
    isSafe:   '',
};

// ---------------------------------------------------------------------------
// Helpers de renderizado (independientes del estado)
// ---------------------------------------------------------------------------

const INTENT_COLORS = { academica: '#0072bc', saludo: '#ffc107', malicioso: '#dc3545' };

function intentBadge(intent) {
    const color     = INTENT_COLORS[intent] || '#adb5bd';
    const textColor = intent === 'saludo' ? '#000' : '#fff';
    return `<span class="badge" style="background-color:${color};color:${textColor}">${intent}</span>`;
}

function typeBadge(type) {
    return type === 'usuario'
        ? `<span class="badge bg-secondary">Usuario</span>`
        : `<span class="badge bg-dark">Asistente</span>`;
}

function feedbackIcon(f) {
    if (f === 1) return '<span class="text-success">👍</span>';
    if (f === 0) return '<span class="text-danger">👎</span>';
    return '<span class="text-muted">—</span>';
}

function stripMarkdown(text) {
    if (!text) return text;
    return text
        .replace(/\*\*(.*?)\*\*/g, '$1')
        .replace(/\*(.*?)\*/g, '$1')
        .replace(/`(.*?)`/g, '$1');
}

function truncate(text, max = 120) {
    const clean = stripMarkdown(text);
    return clean && clean.length > max ? clean.slice(0, max) + '…' : (clean ?? '—');
}

// ---------------------------------------------------------------------------
// Renderizado de la tabla
// ---------------------------------------------------------------------------

function renderLogs(logs) {
    const tbody = document.getElementById('logs-table-body');
    tbody.innerHTML = '';

    if (logs.length === 0) {
        tbody.innerHTML = `
            <tr>
                <td colspan="7" class="text-center text-muted py-4">
                    No hay registros que coincidan con los filtros aplicados.
                </td>
            </tr>`;
        return;
    }

    for (const log of logs) {
        const date = new Date(log.timestamp).toLocaleString('es-ES');

        const scoreCell = log.rerank_score !== null && log.rerank_score !== undefined
            ? `<span class="score-badge" title="CrossEncoder score">${log.rerank_score.toFixed(2)}</span>`
            : '<span class="text-muted">—</span>';

        tbody.insertAdjacentHTML('beforeend', `
            <tr class="log-question-row">
                <td class="text-nowrap text-muted small" rowspan="2">${date}</td>
                <td>${typeBadge('usuario')}</td>
                <td>${truncate(log.question)}</td>
                <td>${intentBadge(log.intent)}</td>
                <td class="text-muted">—</td>
                <td class="text-muted">—</td>
                <td class="text-muted">—</td>
            </tr>
            <tr class="log-answer-row">
                <td>${typeBadge('asistente')}</td>
                <td class="text-muted fst-italic">${truncate(log.answer)}</td>
                <td class="text-muted">—</td>
                <td>${log.tokens_used ?? '—'}</td>
                <td>${scoreCell}</td>
                <td>${feedbackIcon(log.feedback)}</td>
            </tr>
        `);
    }
}

// ---------------------------------------------------------------------------
// Renderizado de la paginación
// ---------------------------------------------------------------------------

function renderPagination(total, totalPages, currentPage, pageSize) {
    const from = total === 0 ? 0 : (currentPage - 1) * pageSize + 1;
    const to   = Math.min(currentPage * pageSize, total);

    document.getElementById('pagination-info').textContent = total > 0
        ? `Mostrando ${from}–${to} de ${total} registros`
        : 'Sin resultados';

    const ul = document.getElementById('pagination-controls');
    ul.innerHTML = '';
    if (totalPages <= 1) return;

    const addBtn = (label, page, disabled = false, active = false) => {
        const li = document.createElement('li');
        li.className = `page-item${disabled ? ' disabled' : ''}${active ? ' active' : ''}`;
        li.innerHTML = disabled
            ? `<span class="page-link">${label}</span>`
            : `<button class="page-link" onclick="goToPage(${page})">${label}</button>`;
        ul.appendChild(li);
    };

    const addEllipsis = () => {
        const li = document.createElement('li');
        li.className = 'page-item disabled';
        li.innerHTML = '<span class="page-link">…</span>';
        ul.appendChild(li);
    };

    const delta = 2;
    const rangeStart = Math.max(1, currentPage - delta);
    const rangeEnd   = Math.min(totalPages, currentPage + delta);

    addBtn('‹', currentPage - 1, currentPage <= 1);

    if (rangeStart > 1) {
        addBtn(1, 1);
        if (rangeStart > 2) addEllipsis();
    }

    for (let i = rangeStart; i <= rangeEnd; i++) {
        addBtn(i, i, false, i === currentPage);
    }

    if (rangeEnd < totalPages) {
        if (rangeEnd < totalPages - 1) addEllipsis();
        addBtn(totalPages, totalPages);
    }

    addBtn('›', currentPage + 1, currentPage >= totalPages);
}

// ---------------------------------------------------------------------------
// Fetch principal (lee de logsState)
// ---------------------------------------------------------------------------

async function fetchLogs() {
    const params = new URLSearchParams({
        page:      logsState.page,
        page_size: logsState.pageSize,
    });
    if (logsState.intent)   params.set('intent',   logsState.intent);
    if (logsState.feedback) params.set('feedback',  logsState.feedback);
    if (logsState.isSafe)   params.set('is_safe',   logsState.isSafe);

    const response = await fetch(`http://127.0.0.1:8000/admin/logs?${params}`, {
        headers: { 'Authorization': `Bearer ${token}` }
    });

    if (response.status === 401) { logout(); return; }

    const data = await response.json();
    renderLogs(data.items);
    renderPagination(data.total, data.total_pages, data.page, data.page_size);
}

// ---------------------------------------------------------------------------
// Control de filtros y navegación de páginas
// ---------------------------------------------------------------------------

function applyFilters() {
    logsState.page     = 1;
    logsState.intent   = document.getElementById('filter-intent').value;
    logsState.feedback = document.getElementById('filter-feedback').value;
    logsState.isSafe   = document.getElementById('filter-safety').value;
    fetchLogs();
}

function resetFilters() {
    logsState.page = logsState.intent = logsState.feedback = logsState.isSafe = '';
    logsState.page = 1;
    document.getElementById('filter-intent').value   = '';
    document.getElementById('filter-feedback').value = '';
    document.getElementById('filter-safety').value   = '';
    fetchLogs();
}

function goToPage(n) {
    if (n < 1) return;
    logsState.page = n;
    fetchLogs();
}

// ---------------------------------------------------------------------------
// Navegación por secciones (sidebar)
// ---------------------------------------------------------------------------

function showSection(name, linkEl) {
    // Ocultar todas las secciones
    document.querySelectorAll('[id^="section-"]').forEach(el => el.classList.add('d-none'));
    // Mostrar la seleccionada
    document.getElementById(`section-${name}`).classList.remove('d-none');
    // Actualizar estilos del sidebar
    document.querySelectorAll('.sidebar .nav-link').forEach(a => a.classList.remove('active'));
    linkEl.classList.add('active');

    // Cargar datos según la sección
    if (name === 'config') loadConfig();
    if (name === 'dashboard') { fetchStats(); fetchLogs(); }
}


// ---------------------------------------------------------------------------
// Configuración dinámica
// ---------------------------------------------------------------------------

const API_HEADERS = () => ({
    'Authorization': `Bearer ${token}`,
    'Content-Type':  'application/json',
});

async function loadConfig() {
    const alertEl = document.getElementById('config-alert');
    alertEl.className = 'alert d-none';

    try {
        const response = await fetch('http://127.0.0.1:8000/admin/config', {
            headers: { 'Authorization': `Bearer ${token}` }
        });

        if (response.status === 401) { logout(); return; }

        const config = await response.json();

        // Seleccionar el modelo en el <select>; si no coincide con ninguna opción,
        // añadimos una opción dinámica para no perder el valor actual
        const modelSelect = document.getElementById('input-model');
        const exists = [...modelSelect.options].some(o => o.value === config.model_name);
        if (!exists) {
            const opt = new Option(`${config.model_name} (actual)`, config.model_name);
            modelSelect.add(opt);
        }
        modelSelect.value = config.model_name;

        document.getElementById('input-prompt').value = config.system_prompt;

    } catch (err) {
        showConfigAlert('danger', `Error al cargar la configuración: ${err.message}`);
    }
}

async function saveConfig() {
    const alertEl   = document.getElementById('config-alert');
    const saveBtn   = document.querySelector('[onclick="saveConfig()"]');
    const modelName = document.getElementById('input-model').value.trim();
    const prompt    = document.getElementById('input-prompt').value.trim();

    if (!modelName || !prompt) {
        showConfigAlert('warning', 'El nombre del modelo y el prompt no pueden estar vacíos.');
        return;
    }

    saveBtn.disabled    = true;
    saveBtn.textContent = 'Guardando…';

    try {
        const response = await fetch('http://127.0.0.1:8000/admin/config', {
            method:  'PUT',
            headers: API_HEADERS(),
            body:    JSON.stringify({ system_prompt: prompt, model_name: modelName }),
        });

        if (response.status === 401) { logout(); return; }

        if (!response.ok) {
            const err = await response.json();
            throw new Error(err.detail || 'Error desconocido');
        }

        showConfigAlert('success', 'Configuración guardada. Se aplicará en la próxima pregunta.');

    } catch (err) {
        showConfigAlert('danger', `Error al guardar: ${err.message}`);
    } finally {
        saveBtn.disabled    = false;
        saveBtn.textContent = 'Guardar cambios';
    }
}

function showConfigAlert(type, message) {
    const alertEl = document.getElementById('config-alert');
    alertEl.className   = `alert alert-${type}`;
    alertEl.textContent = message;
}


// ---------------------------------------------------------------------------
// Inicialización
// ---------------------------------------------------------------------------

fetchStats();
fetchLogs();
