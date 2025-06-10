from app.api import app

if __name__ == "__main__":
    # By default, Flask will serve static files from 'static', so we need to map frontend/
    import os
    from flask import send_from_directory

    # Attach static route for frontend
    @app.route('/')
    def index():
        return send_from_directory('frontend', 'index.html')

    @app.route('/<path:path>')
    def static_proxy(path):
        # serve static files
        return send_from_directory('frontend', path)

    app.run(debug=True)