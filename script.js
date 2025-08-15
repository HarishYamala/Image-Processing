const form = document.getElementById('uploadForm');
const resultDiv = document.getElementById('result');
const imageInput = document.getElementById('imageInput');
const preview = document.getElementById('preview');

imageInput.addEventListener('change', () => {
  const file = imageInput.files[0];
  if (file) {
    preview.src = URL.createObjectURL(file);
    preview.style.display = 'block';
  }
});

form.addEventListener('submit', async function (e) {
  e.preventDefault();
  const formData = new FormData(form);
  resultDiv.textContent = "Evaluating...";
  
  try {
    const response = await fetch('/predict', {
      method: 'POST',
      body: formData
    });

    if (!response.ok) {
      throw new Error("Server error: " + response.statusText);
    }

    const data = await response.json();
    resultDiv.innerHTML = `
      <p>Quality Grade: <strong>${data.quality}</strong></p>
      <p>Rice Type: <strong>${data.type}</strong></p>
    `;
  } catch (error) {
    resultDiv.innerHTML = `<p style="color: red;">Error: ${error.message}</p>`;
  }
});