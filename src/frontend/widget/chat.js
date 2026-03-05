const historial = []; // Cada elemento: { role: "user"|"assistant", content: "..." }

async function sendMessage() {
    const input     = document.getElementById('user-input');
    const sendBtn   = document.getElementById('send-btn');
    const container = document.getElementById('chat-container');
    const question  = input.value.trim();

    if (!question) return;

    // Bloquear la UI mientras se espera la respuesta
    input.disabled  = true;
    sendBtn.disabled = true;

    // Mostrar mensaje del usuario
    appendMessage('user', question);
    input.value = '';

    // Mostrar indicador de carga animado mientras esperamos respuesta
    const loadingWrapper = document.createElement('div');
    loadingWrapper.className = 'message-wrapper bot';
    loadingWrapper.innerHTML = `
        <div class="sender-label">Asistente</div>
        <div class="message">
            <div class="loading-dots">
                <span></span><span></span><span></span>
            </div>
        </div>`;
    container.appendChild(loadingWrapper);
    container.scrollTop = container.scrollHeight;

    try {
        // Guardar pregunta en el historial
        historial.push({ role: "user", content: question });

        // Mantener solo las últimas 3 interacciones (3 preguntas + 3 respuestas = 6 mensajes)
        if (historial.length > 6) historial.splice(0, 2);

        // Llamada POST a la API de FastAPI
        const response = await fetch('http://127.0.0.1:8000/ask', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ question, historial })
        });

        const data = await response.json();

        // Guardar respuesta en el historial
        historial.push({ role: "assistant", content: data.answer });

        // Reemplazar el loading por la respuesta real del asistente
        container.removeChild(loadingWrapper);
        appendMessage('bot', data.answer, data.sources);

    } catch (error) {
        console.error(error);
        container.removeChild(loadingWrapper);
        appendMessage('bot', 'Lo siento, ha ocurrido un error al conectar con el servidor.');
    } finally {
        // Restaurar la UI siempre, tanto si hay éxito como si hay error
        input.disabled   = false;
        sendBtn.disabled = false;
        input.focus();
    }
}

function appendMessage(sender, text, sources = []) {
    const container = document.getElementById('chat-container');

    const wrapper = document.createElement('div');
    wrapper.className = `message-wrapper ${sender}`;

    const label = sender === 'user' ? 'Tú' : 'Asistente';

    // Burbuja principal con el texto de la respuesta
    let html = `
        <div class="sender-label">${label}</div>
        <div class="message">${marked.parse(text)}</div>`;

    // Bloque de fuentes: solo para mensajes del bot y si hay fuentes
    if (sender === 'bot' && sources && sources.length > 0) {

        // Cada fuente es un objeto { title, url } devuelto por el backend
        const items = sources
            .filter(s => s.url)
            .map(s => {
                const isPdf  = s.url.endsWith('.pdf');
                const icon   = isPdf ? '📄' : '🌐';
                // Título legible procedente del metadata del chunk
                const title  = s.title || (isPdf ? 'Documento PDF' : 'Página web');
                // Texto corto para la segunda línea: nombre de archivo o dominio
                const short  = isPdf
                    ? s.url.split('/').pop()
                    : s.url.replace(/^https?:\/\//, '');

                return `
                    <a class="source-item" href="${s.url}" target="_blank" rel="noopener">
                        <span class="source-icon">${icon}</span>
                        <span class="source-label">
                            <span class="source-title">${title}</span>
                            <span class="source-url">${short}</span>
                        </span>
                    </a>`;
            })
            .join('');

        // Desplegable cerrado por defecto: el usuario pulsa para ver las fuentes
        html += `
            <details class="sources">
                <summary>
                    <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5">
                        <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
                        <polyline points="14 2 14 8 20 8"/>
                    </svg>
                    Fuentes consultadas
                </summary>
                <div class="sources-list">${items}</div>
            </details>`;
    }

    wrapper.innerHTML = html;
    container.appendChild(wrapper);
    container.scrollTop = container.scrollHeight;
}
