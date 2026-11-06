document.addEventListener("DOMContentLoaded", function () {
    const dropZone = document.getElementById("dropZone");
    const fileInput = document.getElementById("fileInput");
    const dropZonePrompt = document.getElementById("dropZonePrompt");
    const previewContainer = document.getElementById("previewContainer");
    const imagePreview = document.getElementById("imagePreview");
    const removeImageBtn = document.getElementById("removeImageBtn");
    const fileDetails = document.getElementById("fileDetails");
    const submitBtn = document.getElementById("submitBtn");
    const uploadForm = document.getElementById("uploadForm");
    const btnText = document.getElementById("btnText");
    const btnSpinner = document.getElementById("btnSpinner");

    if (!dropZone || !fileInput) return;

    // Prevent default drag behaviors
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, preventDefaults, false);
        document.body.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    // Highlight drop zone on drag
    ['dragenter', 'dragover'].forEach(eventName => {
        dropZone.addEventListener(eventName, () => dropZone.classList.add('drag-over'), false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, () => dropZone.classList.remove('drag-over'), false);
    });

    // Handle dropped files
    dropZone.addEventListener('drop', (e) => {
        const dt = e.dataTransfer;
        const files = dt.files;
        if (files.length > 0) {
            handleFile(files[0]);
        }
    });

    // Handle selected file via file picker
    fileInput.addEventListener('change', function () {
        if (this.files.length > 0) {
            handleFile(this.files[0]);
        }
    });

    function handleFile(file) {
        const validTypes = ['image/jpeg', 'image/png', 'image/webp', 'image/bmp'];
        if (!validTypes.includes(file.type)) {
            alert('Please select a valid image file (JPG, PNG, WEBP, or BMP).');
            return;
        }

        const maxSize = 16 * 1024 * 1024; // 16MB
        if (file.size > maxSize) {
            alert('The selected file exceeds 16MB. Please choose a smaller image.');
            return;
        }

        // Preview file
        const reader = new FileReader();
        reader.onload = function (e) {
            imagePreview.src = e.target.result;
            dropZonePrompt.classList.add('d-none');
            previewContainer.classList.remove('d-none');
            
            const sizeInKb = (file.size / 1024).toFixed(1);
            fileDetails.textContent = `${file.name} (${sizeInKb} KB)`;
            
            submitBtn.disabled = false;
        };
        reader.readAsDataURL(file);
    }

    // Remove image button
    if (removeImageBtn) {
        removeImageBtn.addEventListener('click', function (e) {
            e.stopPropagation();
            fileInput.value = '';
            imagePreview.src = '#';
            dropZonePrompt.classList.remove('d-none');
            previewContainer.classList.add('d-none');
            submitBtn.disabled = true;
        });
    }

    // Handle form submission animation
    if (uploadForm) {
        uploadForm.addEventListener('submit', function () {
            submitBtn.disabled = true;
            btnText.textContent = "Analyzing Image...";
            btnSpinner.classList.remove('d-none');
        });
    }
});
