from app.api import app
import os

if __name__ == "__main__":
    # By default, Flask will serve static files from 'static', so we need to map frontend/
    import os
    from flask import send_from_directory

    # Attach static route for frontend
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    FRONTEND_FOLDER = os.path.join(BASE_DIR, 'frontend')
    @app.route('/')
    def index():
        return send_from_directory(FRONTEND_FOLDER, 'index.html')

    @app.route('/<path:path>')
    def static_proxy(path):
        # serve static files
        return send_from_directory(FRONTEND_FOLDER, path)

    app.run(debug=True)