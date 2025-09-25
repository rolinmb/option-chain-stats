from consts import PORT
import http.server
import socketserver
import os

BASE_DIR = os.path.dirname(__file__)
PUBLIC_DIR = os.path.join(BASE_DIR, "..", "public")
DATA_DIR = os.path.join(BASE_DIR, "..", "data")
IMG_DIR = os.path.join(BASE_DIR, "..", "img")

class MyHandler(http.server.SimpleHTTPRequestHandler):
    def translate_path(self, path):
        # Route /data/* → DATA_DIR
        if path.startswith("/data/"):
            rel = path[len("/data/"):]
            return os.path.join(DATA_DIR, rel)
        # Route /img/* → IMG_DIR
        if path.startswith("/img/"):
            rel = path[len("/img/"):]
            return os.path.join(IMG_DIR, rel)
        # Default → PUBLIC_DIR
        return super().translate_path(path)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=PUBLIC_DIR, **kwargs)

if __name__ == "__main__":
    with socketserver.TCPServer(("", PORT), MyHandler) as httpd:
        print(f"Serving public/, plus /data/ and /img/ at http://localhost:{PORT}")
        httpd.serve_forever()