document.addEventListener('DOMContentLoaded', () => {

    // URL da sua API Python (Flask/FastAPI)
    const API_URL = 'http://127.0.0.1:5000';

    // Variável global para armazenar os dados tratados
    let treatedData = null;
    let chartInstance = null; // Para guardar a instância do gráfico

    // Seleção de todos os elementos da interface
    const sections = document.querySelectorAll('main > section');
    const navInicio = document.getElementById('nav-inicio');
    
    // Elementos da Página Inicial
    const btnIniciarTratamento = document.getElementById('btn-iniciar-tratamento');

    // Elementos da Página de Tratamento
    const uploadArea = document.getElementById('upload-area');
    const fileInput = document.getElementById('file-input');
    const btnSelecionarArquivo = document.getElementById('btn-selecionar-arquivo');
    const fileNameDisplay = document.getElementById('file-name');
    const opcoesTratamento = document.getElementById('opcoes-tratamento');
    const btnExecutarTratamento = document.getElementById('btn-executar-tratamento');
    const tratamentoSpinner = document.getElementById('tratamento-spinner');
    const resultadoTratamento = document.getElementById('resultado-tratamento');
    const tabelaHead = document.getElementById('tabela-head');
    const tabelaBody = document.getElementById('tabela-body');
    const btnIrParaAnalise = document.getElementById('btn-ir-para-analise');

    // Elementos da Página de Resultados
    const selectTarget = document.getElementById('select-target');
    const selectInputs = document.getElementById('select-inputs');
    const btnExecutarAnalise = document.getElementById('btn-executar-analise');
    const analiseSpinner = document.getElementById('analise-spinner');
    const dashboardResultados = document.getElementById('dashboard-resultados');
    const metricR2 = document.getElementById('metric-r2');
    const metricMse = document.getElementById('metric-mse');
    const graficoCanvas = document.getElementById('grafico-dispersao');

    // Função para navegar entre as "páginas" (seções)
    const showPage = (pageId) => {
        sections.forEach(section => {
            section.classList.remove('active');
        });
        document.getElementById(pageId).classList.add('active');
    };

    // --- LÓGICA DA PÁGINA DE TRATAMENTO ---

    // Acionar input de arquivo ao clicar no botão
    btnSelecionarArquivo.addEventListener('click', () => fileInput.click());
    uploadArea.addEventListener('click', () => fileInput.click());

    // Eventos de Drag & Drop
    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.classList.add('drag-over');
    });
    uploadArea.addEventListener('dragleave', () => {
        uploadArea.classList.remove('drag-over');
    });
    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.classList.remove('drag-over');
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            fileInput.files = files;
            handleFileSelect(files[0]);
        }
    });

    // Evento quando um arquivo é selecionado
    fileInput.addEventListener('change', () => {
        if (fileInput.files.length > 0) {
            handleFileSelect(fileInput.files[0]);
        }
    });

    const handleFileSelect = (file) => {
        fileNameDisplay.textContent = `Arquivo selecionado: ${file.name}`;
        opcoesTratamento.classList.remove('d-none');
    };
    
    // Executar o tratamento dos dados
    btnExecutarTratamento.addEventListener('click', async () => {
        const file = fileInput.files[0];
        if (!file) {
            alert('Por favor, selecione um arquivo CSV primeiro.');
            return;
        }

        const formData = new FormData();
        formData.append('dataFile', file);
        // Futuramente, você pode enviar as opções de tratamento aqui também
        // formData.append('removerNulos', document.getElementById('remover-nulos').checked);

        tratamentoSpinner.classList.remove('d-none');
        btnExecutarTratamento.disabled = true;

        try {
            const response = await fetch(`${API_URL}/tratar-dados`, {
                method: 'POST',
                body: formData,
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.error || 'Erro no servidor');
            }

            const result = await response.json();
            treatedData = result.data; // Salva os dados tratados globalmente
            
            displayTreatedData(treatedData);
            populateSelects(treatedData);

        } catch (error) {
            alert(`Erro ao tratar dados: ${error.message}`);
            resultadoTratamento.classList.add('d-none');
        } finally {
            tratamentoSpinner.classList.add('d-none');
            btnExecutarTratamento.disabled = false;
        }
    });

    const displayTreatedData = (data) => {
        tabelaHead.innerHTML = '';
        tabelaBody.innerHTML = '';

        if (data.length === 0) return;

        // Cria o cabeçalho
        const headers = Object.keys(data[0]);
        const headerRow = document.createElement('tr');
        headers.forEach(headerText => {
            const th = document.createElement('th');
            th.textContent = headerText;
            headerRow.appendChild(th);
        });
        tabelaHead.appendChild(headerRow);

        // Cria as linhas de dados (apenas as 10 primeiras)
        data.slice(0, 10).forEach(rowData => {
            const row = document.createElement('tr');
            headers.forEach(header => {
                const td = document.createElement('td');
                td.textContent = rowData[header];
                row.appendChild(td);
            });
            tabelaBody.appendChild(row);
        });

        resultadoTratamento.classList.remove('d-none');
    };
    
    const populateSelects = (data) => {
        if (data.length === 0) return;
        
        selectTarget.innerHTML = '';
        selectInputs.innerHTML = '';
        const columns = Object.keys(data[0]);

        columns.forEach(col => {
            // Adiciona ao seletor de Target
            const optionTarget = document.createElement('option');
            optionTarget.value = col;
            optionTarget.textContent = col;
            selectTarget.appendChild(optionTarget);

            // Adiciona ao seletor de Inputs
            const optionInput = document.createElement('option');
            optionInput.value = col;
            optionInput.textContent = col;
            selectInputs.appendChild(optionInput);
        });
    };

    // --- LÓGICA DA PÁGINA DE ANÁLISE ---

    btnExecutarAnalise.addEventListener('click', async () => {
        const target = selectTarget.value;
        const inputs = Array.from(selectInputs.selectedOptions).map(opt => opt.value);

        if (!target || inputs.length === 0) {
            alert('Por favor, selecione a variável target e pelo menos uma variável de entrada.');
            return;
        }
        
        if (inputs.includes(target)) {
            alert('A variável target não pode ser uma variável de entrada.');
            return;
        }

        analiseSpinner.classList.remove('d-none');
        btnExecutarAnalise.disabled = true;

        try {
            const response = await fetch(`${API_URL}/executar-analise`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    data: treatedData,
                    target: target,
                    inputs: inputs
                })
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.error || 'Erro no servidor');
            }
            
            const results = await response.json();
            displayAnalysisResults(results);

        } catch (error) {
            alert(`Erro ao executar análise: ${error.message}`);
        } finally {
            analiseSpinner.classList.add('d-none');
            btnExecutarAnalise.disabled = false;
        }
    });

    const displayAnalysisResults = (results) => {
        metricR2.textContent = results.r2_score.toFixed(4);
        metricMse.textContent = results.mse.toFixed(2);
        
        const dataPoints = results.predictions.map(p => ({
            x: p.real,
            y: p.predicted
        }));

        renderChart(dataPoints);

        dashboardResultados.classList.remove('d-none');
    };
    
    const renderChart = (dataPoints) => {
        if(chartInstance) {
            chartInstance.destroy(); // Destroi o gráfico anterior para criar um novo
        }
        chartInstance = new Chart(graficoCanvas, {
            type: 'scatter',
            data: {
                datasets: [{
                    label: 'Previsto vs. Real',
                    data: dataPoints,
                    backgroundColor: 'rgba(0, 123, 255, 0.6)'
                }]
            },
            options: {
                scales: {
                    x: { title: { display: true, text: 'Valores Reais' } },
                    y: { title: { display: true, text: 'Valores Previstos' } }
                }
            }
        });
    };


    // --- NAVEGAÇÃO INICIAL ---

    navInicio.addEventListener('click', (e) => {
        e.preventDefault();
        showPage('pagina-inicial');
    });

    btnIniciarTratamento.addEventListener('click', () => {
        showPage('pagina-tratamento');
    });
    
    btnIrParaAnalise.addEventListener('click', () => {
        showPage('pagina-resultados');
    });

    // Inicia na página inicial
    showPage('pagina-inicial'); 
});