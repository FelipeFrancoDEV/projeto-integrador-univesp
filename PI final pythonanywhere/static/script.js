const container = document.getElementById('container');
const registerBtn = document.getElementById('register');
const loginBtn = document.getElementById('login');

registerBtn.addEventListener('click', () => {
    container.classList.add("active");
});

loginBtn.addEventListener('click', () => {
    container.classList.remove("active");
});
document.addEventListener('DOMContentLoaded', function() {
    const form = document.querySelector('form[action="/cadastro"]');

    // Adiciona estilos dinâmicos
    const style = document.createElement('style');
    style.textContent = `
        .error {
            border: 1px solid red !important;
        }
        .error-message {
            color: red;
            font-size: 12px;
            margin-top: 5px;
            display: none;
        }
    `;
    document.head.appendChild(style);

    // Cria mensagens de erro para cada campo
    const campos = [
        { input: form.querySelector('input[name="usuario"]'), message: 'Usuário é obrigatório' },
        { input: form.querySelector('input[name="email"]'), message: 'Email é obrigatório', email: true },
        { input: form.querySelector('input[name="senha"]'), message: 'Senha é obrigatória' }
    ];

    campos.forEach(campo => {
        const errorMessage = document.createElement('div');
        errorMessage.className = 'error-message';
        errorMessage.textContent = campo.message;
        campo.input.parentNode.insertBefore(errorMessage, campo.input.previousSibling);

        // Validação em tempo real
        campo.input.addEventListener('input', function() {
            if (this.value.trim() !== '') {
                this.classList.remove('error');
                errorMessage.style.display = 'none';
            }
        });
    });

    // Validação no submit
    form.addEventListener('submit', function(event) {
        let isValid = true;

        campos.forEach(campo => {
            if (campo.input.value.trim() === '') {
                campo.input.classList.add('error');
                campo.input.previousElementSibling.style.display = 'block';
                if (isValid) campo.input.focus();
                isValid = false;
            }
            else if (campo.email && !isValidEmail(campo.input.value)) {
                campo.input.nextElementSibling.textContent = 'Por favor, insira um email válido';
                campo.input.classList.add('error');
                campo.input.nextElementSibling.style.display = 'block';
                if (isValid) campo.input.focus();
                isValid = false;
            }
        });

        if (!isValid) {
            event.preventDefault();
        }
    });

    function isValidEmail(email) {
        const re = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
        return re.test(email);
    }

});