if (localStorage.getItem('admin_token')) {
    window.location.href = 'dashboard.html';
}

document.getElementById('login-form').addEventListener('submit', async (e) => {
    e.preventDefault();

    const btn      = document.getElementById('submit-btn');
    const errorDiv = document.getElementById('error-msg');
    const username = document.getElementById('username').value.trim();
    const password = document.getElementById('password').value;

    btn.disabled    = true;
    btn.textContent = 'Iniciando sesión...';
    errorDiv.classList.add('d-none');

    try {
        // OAuth2PasswordRequestForm requiere application/x-www-form-urlencoded, NO JSON
        const body = new URLSearchParams({ username, password });

        const response = await fetch('http://127.0.0.1:8000/login', {
            method:  'POST',
            headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
            body,
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.detail || 'Error desconocido');
        }

        localStorage.setItem('admin_token', data.access_token);
        window.location.href = 'dashboard.html';

    } catch (err) {
        errorDiv.textContent = err.message;
        errorDiv.classList.remove('d-none');
    } finally {
        btn.disabled    = false;
        btn.textContent = 'Iniciar sesión';
    }
});
