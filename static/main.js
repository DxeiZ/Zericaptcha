async function fetchCaptcha() {
    const response = await fetch('/generate_captcha');
    const data = await response.json();
    const captchaGrid = document.getElementById('captcha-grid');
    const targetClassElement = document.getElementById('target-class');
    targetClassElement.textContent = data.target_class;
    captchaGrid.innerHTML = '';

    data.images.forEach((img, index) => {
        const imgElement = document.createElement('img');
        imgElement.src = `data:image/jpeg;base64,${img}`;
        imgElement.dataset.index = index;
        imgElement.dataset.label = data.labels[index];
        imgElement.classList.add('col', 'p-0', 'img-thumbnail');
        imgElement.draggable = false;
        imgElement.addEventListener('click', () => {
            imgElement.classList.toggle('selected');
        });
        captchaGrid.appendChild(imgElement);
    });
}

function redirectToTurkhackteam() {
    setTimeout(function() {
        window.location.href = 'https://www.turkhackteam.org';
    }, 5000);
}

document.getElementById('submit-button').addEventListener('click', async () => {
    const selectedImages = document.querySelectorAll('#captcha-grid .selected');
    const selectedIndices = Array.from(selectedImages).map(img => parseInt(img.dataset.index));
    const gridLabels = Array.from(document.getElementById('captcha-grid').children).map(img => img.dataset.label);
    const targetClass = document.getElementById('target-class').textContent;

    if (selectedIndices.length !== 3) {
        document.getElementById('result').textContent = 'Lütfen tam olarak 3 resim seçin.';
        return;
    }

    const response = await fetch('/validate_captcha', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            selected_indices: selectedIndices,
            grid_labels: gridLabels,
            target_class: targetClass
        })
    });
    const result = await response.json();
    if (result.result == 'true') {
        document.getElementById('result').textContent = 'Başarılı, 5 saniye sonra yönlendiriliyorunuz ..';
        document.getElementById('result').onload = redirectToTurkhackteam();
    }
    if (result.result == 'false') {
        document.getElementById('result').textContent = 'Yanlış! Lütfen belirtilen sınıftan 3 resim seçin.';
        fetchCaptcha();
    }
});

document.getElementById('fooRel').addEventListener('click', async () => {
    fetchCaptcha();
})

fetchCaptcha();
