from app.api import app
import os
from flask import send_from_directory

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_FOLDER = os.path.join(BASE_DIR, 'frontend')
@app.route('/')
def index():
    return send_from_directory(FRONTEND_FOLDER, 'index.html')

@app.route('/<path:path>')
def static_proxy(path):
    # serve static files
    return send_from_directory(FRONTEND_FOLDER, path)

if __name__ == "__main__":
    app.run(debug=True, port=8000)