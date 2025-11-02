// File upload preview
document.addEventListener('DOMContentLoaded', function() {
    const fileInput = document.getElementById('resumes');
    if (fileInput) {
        fileInput.addEventListener('change', function(e) {
            const files = e.target.files;
            console.log(`Selected ${files.length} file(s)`);
        });
    }
});
