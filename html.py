Python 3.13.3 (tags/v3.13.3:6280bb5, Apr  8 2025, 14:47:33) [MSC v.1943 64 bit (AMD64)] on win32
Enter "help" below or click "Help" above for more information.
>>> <!DOCTYPE html>
... <html lang="en">
... <head>
...   <meta charset="UTF-8">
...   <title>Rice Image Quality Evaluation</title>
...   <link rel="stylesheet" href="{{ url_for('static', filename='style.css') }}">
... </head>
... <body>
...   <div class="container">
...     <h1>Rice Image Quality Evaluation</h1>
...     <form id="uploadForm" enctype="multipart/form-data">
...       <input type="file" name="image" id="imageInput" accept="image/*" required><br>
...       <button type="submit">Evaluate</button>
...     </form>
...     <img id="preview" alt="Image Preview">
...     <div id="result"></div>
...   </div>
... 
...   <script src="{{ url_for('static', filename='script.js') }}"></script>
