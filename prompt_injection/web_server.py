import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from prompt_injection.predict import Predictor

WEB_DIR = Path(__file__).resolve().parent / "web"
predictor = None

class WebUIHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            html_path = WEB_DIR / "index.html"
            with open(html_path, "r", encoding="utf-8") as f:
                self.wfile.write(f.read().encode("utf-8"))
        elif self.path == "/style.css":
            self.send_response(200)
            self.send_header("Content-Type", "text/css; charset=utf-8")
            self.end_headers()
            css_path = WEB_DIR / "style.css"
            with open(css_path, "r", encoding="utf-8") as f:
                self.wfile.write(f.read().encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b"404 Not Found")

    def do_POST(self):
        if self.path == "/api/predict":
            content_length = int(self.headers.get("Content-Length", 0))
            post_data = self.rfile.read(content_length)
            try:
                body = json.loads(post_data.decode("utf-8"))
                text = body.get("text", "").strip()
                if not text:
                    self.send_response(400)
                    self.end_headers()
                    self.wfile.write(b"Missing text parameter")
                    return

                global predictor
                if predictor is None:
                    print("Initializing predictor model on GET request (lazy load)...")
                    predictor = Predictor()

                result = predictor.predict(text)
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(result).encode("utf-8"))
            except Exception as e:
                self.send_response(500)
                self.end_headers()
                self.wfile.write(f"Server error: {e}".encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b"404 Not Found")

def run(port=8000):
    server_address = ("127.0.0.1", port)
    httpd = HTTPServer(server_address, WebUIHandler)
    print(f"\n==================================================", flush=True)
    print(f"  Prompt Injection Guardrail Dashboard", flush=True)
    print(f"  Open in your browser: http://127.0.0.1:{port}", flush=True)
    print(f"  Press Ctrl+C to stop the server.", flush=True)
    print(f"==================================================\n", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server...", flush=True)

if __name__ == "__main__":
    run()
