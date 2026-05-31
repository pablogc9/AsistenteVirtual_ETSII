/* ============================================================================
 *  Asistente Virtual ETSI Informática (UMA) — Widget embebible
 * ----------------------------------------------------------------------------
 *  Inyecta un botón flotante en la esquina de cualquier página web. Al pulsarlo
 *  se despliega un panel de chat superpuesto, sin ocupar toda la pantalla.
 *
 *  Integración en cualquier web (una sola línea):
 *      <script src="http://TU_SERVIDOR/static/widget/embed.js"></script>
 *
 *  Configuración opcional (definir ANTES de cargar este script):
 *      <script>window.ETSI_WIDGET_API = "http://127.0.0.1:8000";</script>
 *
 *  Aislamiento: todo el widget vive dentro de un Shadow DOM, y sus estilos se
 *  cargan desde widget.css (en la misma carpeta), por lo que NO afectan a la
 *  página anfitriona ni se ven afectados por ella.
 * ========================================================================== */
(function () {
    "use strict";

    // Capturar la URL de este propio script ANTES de cualquier await/async,
    // para poder derivar la ruta del CSS del widget (widget.css en la misma carpeta).
    const SCRIPT_SRC = (document.currentScript && document.currentScript.src) || "";

    // Evitar doble inyección si el script se carga dos veces
    if (window.__etsiWidgetLoaded) return;
    window.__etsiWidgetLoaded = true;

    // ── Configuración ───────────────────────────────────────────────────────
    const API_BASE = (window.ETSI_WIDGET_API || "http://127.0.0.1:8000").replace(/\/$/, "");

    // Rutas a los recursos del widget (CSS y HTML): misma carpeta que embed.js.
    // Si no se puede derivar (p.ej. inyección manual), se usa la ruta estándar.
    const WIDGET_CSS = SCRIPT_SRC
        ? SCRIPT_SRC.replace(/embed\.js(\?.*)?$/, "widget.css")
        : `${API_BASE}/static/widget/widget.css`;
    const WIDGET_HTML = SCRIPT_SRC
        ? SCRIPT_SRC.replace(/embed\.js(\?.*)?$/, "widget.html")
        : `${API_BASE}/static/widget/widget.html`;

    // ── Historial de conversación ─────────────────────────────────────────────
    const historial = []; // { role: "user"|"assistant", content: "..." }

    // ── Cargar marked.js (markdown) desde CDN si no está presente ─────────────
    function ensureMarked(callback) {
        if (window.marked) return callback();
        const s = document.createElement("script");
        s.src = "https://cdn.jsdelivr.net/npm/marked/marked.min.js";
        s.onload = callback;
        s.onerror = callback; // si falla, seguimos sin markdown
        document.head.appendChild(s);
    }

    function renderMarkdown(text) {
        if (window.marked && typeof window.marked.parse === "function") {
            return window.marked.parse(text);
        }
        // Fallback: escapar HTML y respetar saltos de línea
        const div = document.createElement("div");
        div.textContent = text;
        return div.innerHTML.replace(/\n/g, "<br>");
    }

    // ── Construcción del widget en Shadow DOM ─────────────────────────────────
    async function buildWidget() {
        // Cargar la fuente IBM Plex en el documento anfitrión (las @font-face
        // no cruzan el shadow boundary, pero las fuentes ya cargadas sí se usan)
        if (!document.getElementById("etsi-widget-font")) {
            const font = document.createElement("link");
            font.id = "etsi-widget-font";
            font.rel = "stylesheet";
            font.href = "https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap";
            document.head.appendChild(font);
        }

        // Cargar el marcado HTML del widget desde widget.html
        let markup;
        try {
            const resp = await fetch(WIDGET_HTML);
            markup = await resp.text();
        } catch (e) {
            console.error("[ETSI Widget] No se pudo cargar widget.html:", e);
            return;
        }

        const host = document.createElement("div");
        host.id = "etsi-widget-host";
        document.body.appendChild(host);

        const root = host.attachShadow({ mode: "open" });

        // Los estilos se cargan desde widget.css mediante un <link> DENTRO del
        // shadow root, manteniendo el aislamiento respecto a la página anfitriona.
        const linkEl = document.createElement("link");
        linkEl.rel = "stylesheet";
        linkEl.href = WIDGET_CSS;
        root.appendChild(linkEl);

        const container = document.createElement("div");
        container.innerHTML = markup;
        root.appendChild(container);

        wireEvents(root);
    }

    // ── Lógica de eventos e interacción ───────────────────────────────────────
    function wireEvents(root) {
        const launcher    = root.getElementById("launcher");
        const panel       = root.getElementById("panel");
        const headerClose = root.getElementById("header-close");
        const input       = root.getElementById("user-input");
        const sendBtn     = root.getElementById("send-btn");

        function togglePanel() {
            const isOpen = panel.classList.toggle("open");
            launcher.classList.toggle("open", isOpen);
            if (isOpen) setTimeout(() => input.focus(), 150);
        }
        function closePanel() {
            panel.classList.remove("open");
            launcher.classList.remove("open");
        }

        launcher.addEventListener("click", togglePanel);
        headerClose.addEventListener("click", closePanel);
        sendBtn.addEventListener("click", () => sendMessage(root));
        input.addEventListener("keypress", (e) => {
            if (e.key === "Enter") sendMessage(root);
        });
    }

    // ── Envío de mensajes ─────────────────────────────────────────────────────
    async function sendMessage(root) {
        const input   = root.getElementById("user-input");
        const sendBtn = root.getElementById("send-btn");
        const chat    = root.getElementById("chat");
        const question = input.value.trim();
        if (!question) return;

        input.disabled = true;
        sendBtn.disabled = true;

        appendMessage(root, "user", question);
        input.value = "";

        const loading = document.createElement("div");
        loading.className = "message-wrapper bot";
        loading.innerHTML = `
            <div class="sender-label">Asistente</div>
            <div class="message"><div class="loading-dots"><span></span><span></span><span></span></div></div>`;
        chat.appendChild(loading);
        chat.scrollTop = chat.scrollHeight;

        try {
            historial.push({ role: "user", content: question });
            if (historial.length > 6) historial.splice(0, 2);

            const response = await fetch(`${API_BASE}/ask`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ question, historial }),
            });
            const data = await response.json();

            historial.push({ role: "assistant", content: data.answer });
            chat.removeChild(loading);
            appendMessage(root, "bot", data.answer, data.sources, data.log_id);
        } catch (error) {
            console.error(error);
            chat.removeChild(loading);
            appendMessage(root, "bot", "Lo siento, ha ocurrido un error al conectar con el servidor.");
        } finally {
            input.disabled = false;
            sendBtn.disabled = false;
            input.focus();
        }
    }

    // ── Feedback ──────────────────────────────────────────────────────────────
    async function sendFeedback(root, logId, score, btn) {
        const bar     = btn.closest(".feedback-bar");
        const buttons = bar.querySelectorAll(".feedback-btn");
        buttons.forEach((b) => (b.disabled = true));
        try {
            await fetch(`${API_BASE}/feedback`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ log_id: logId, score }),
            });
            btn.classList.add("feedback-btn--selected");
            buttons.forEach((b) => { if (b !== btn) b.style.display = "none"; });
            bar.querySelector(".feedback-label").textContent = "¡Gracias!";
        } catch {
            buttons.forEach((b) => (b.disabled = false));
        }
    }

    // ── Render de un mensaje ────────────────────────────────────────────────────
    function appendMessage(root, sender, text, sources = [], logId = null) {
        const chat = root.getElementById("chat");
        const wrapper = document.createElement("div");
        wrapper.className = `message-wrapper ${sender}`;
        const label = sender === "user" ? "Tú" : "Asistente";

        let html = `
            <div class="sender-label">${label}</div>
            <div class="message">${renderMarkdown(text)}</div>`;

        if (sender === "bot" && sources && sources.length > 0) {
            const items = sources
                .filter((s) => s.title || s.url)
                .map((s) => {
                    const hasUrl = s.url && s.url.length > 0;
                    const isPdf  = hasUrl && s.url.endsWith(".pdf");
                    const icon   = !hasUrl ? "🗂️" : (isPdf ? "📄" : "🌐");
                    const title  = s.title || (isPdf ? "Documento PDF" : "Página web");
                    const short  = !hasUrl ? "" : (isPdf ? s.url.split("/").pop() : s.url.replace(/^https?:\/\//, ""));
                    if (hasUrl) {
                        return `
                            <a class="source-item" href="${s.url}" target="_blank" rel="noopener">
                                <span class="source-icon">${icon}</span>
                                <span class="source-label">
                                    <span class="source-title">${title}</span>
                                    ${short ? `<span class="source-url">${short}</span>` : ""}
                                </span>
                            </a>`;
                    }
                    return `
                        <div class="source-item source-item--nolink">
                            <span class="source-icon">${icon}</span>
                            <span class="source-label"><span class="source-title">${title}</span></span>
                        </div>`;
                })
                .join("");

            html += `
                <details class="sources">
                    <summary>Fuentes consultadas</summary>
                    <div class="sources-list">${items}</div>
                </details>`;
        }

        if (sender === "bot" && logId) {
            html += `
                <div class="feedback-bar">
                    <span class="feedback-label">¿Fue útil?</span>
                    <button class="feedback-btn" data-score="1" title="Sí, fue útil">
                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                            <path d="M14 9V5a3 3 0 0 0-3-3l-4 9v11h11.28a2 2 0 0 0 2-1.7l1.38-9a2 2 0 0 0-2-2.3H14z"/>
                            <path d="M7 22H4a2 2 0 0 1-2-2v-7a2 2 0 0 1 2-2h3"/>
                        </svg>
                    </button>
                    <button class="feedback-btn" data-score="0" title="No fue útil">
                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                            <path d="M10 15v4a3 3 0 0 0 3 3l4-9V2H5.72a2 2 0 0 0-2 1.7l-1.38 9a2 2 0 0 0 2 2.3H10z"/>
                            <path d="M17 2h2.67A2.31 2.31 0 0 1 22 4v7a2.31 2.31 0 0 1-2.33 2H17"/>
                        </svg>
                    </button>
                </div>`;
        }

        wrapper.innerHTML = html;

        // Conectar los botones de feedback (sin onclick inline por el Shadow DOM)
        if (sender === "bot" && logId) {
            wrapper.querySelectorAll(".feedback-btn").forEach((btn) => {
                btn.addEventListener("click", () =>
                    sendFeedback(root, logId, parseInt(btn.dataset.score, 10), btn)
                );
            });
        }

        chat.appendChild(wrapper);
        chat.scrollTop = chat.scrollHeight;
    }

    // ── Arranque ────────────────────────────────────────────────────────────────
    function init() {
        ensureMarked(buildWidget);
    }
    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", init);
    } else {
        init();
    }
})();
