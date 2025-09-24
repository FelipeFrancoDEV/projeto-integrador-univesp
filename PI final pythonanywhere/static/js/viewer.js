function initImageViewer() {
    const thumbnails = document.querySelectorAll('.graph-thumbnail');
    const imageViewer = document.getElementById('imageViewer');
    const expandedImg = document.getElementById('expandedImage');
    const closeBtn = document.querySelector('.close-viewer');
    const prevBtn = document.getElementById('prevBtn');
    const nextBtn = document.getElementById('nextBtn');
    const imageCounter = document.getElementById('imageCounter');
    
    // Converte NodeList para Array
    const imagesArray = Array.from(thumbnails);
    let currentIndex = 0;
    
    // Função para mostrar imagem
    function showImage(index) {
        // Verifica limites do array
        if (index < 0) index = imagesArray.length - 1;
        if (index >= imagesArray.length) index = 0;
        
        currentIndex = index;
        expandedImg.src = imagesArray[currentIndex].src;
        updateCounter();
    }
    
    // Atualiza contador
    function updateCounter() {
        imageCounter.textContent = `${currentIndex + 1}/${imagesArray.length}`;
    }
    
    // Fecha o visualizador
    function closeViewer() {
        imageViewer.style.display = 'none';
        document.body.style.overflow = 'auto';
    }
    
    // Eventos de clique nas miniaturas
    thumbnails.forEach((thumbnail, index) => {
        thumbnail.addEventListener('click', function() {
            currentIndex = index;
            showImage(currentIndex);
            imageViewer.style.display = 'block';
            document.body.style.overflow = 'hidden';
        });
    });
    
    // Eventos de navegação
    prevBtn.addEventListener('click', () => showImage(currentIndex - 1));
    nextBtn.addEventListener('click', () => showImage(currentIndex + 1));
    
    // Eventos de teclado
    document.addEventListener('keydown', function(e) {
        if (imageViewer.style.display !== 'block') return;
        
        switch(e.key) {
            case 'Escape':
                closeViewer();
                break;
            case 'ArrowLeft':
                showImage(currentIndex - 1);
                break;
            case 'ArrowRight':
                showImage(currentIndex + 1);
                break;
        }
    });
    
    // Fecha ao clicar no X ou fora da imagem
    closeBtn.addEventListener('click', closeViewer);
    imageViewer.addEventListener('click', function(e) {
        if (e.target === imageViewer) closeViewer();
    });
}

// Inicializa quando o DOM estiver pronto
document.addEventListener('DOMContentLoaded', initImageViewer);
