from http.server import BaseHTTPRequestHandler, HTTPServer
import json
from clone_detector import detect_clones

HOST = "127.0.0.1"
PORT = 5000


class CloneServer(BaseHTTPRequestHandler):
    def _set_headers(self):
        self.send_response(200)
        self.send_header("Content-type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")  # allow frontend access
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_OPTIONS(self):
        self._set_headers()

    def do_POST(self):
        try:
            content_length = int(self.headers["Content-Length"])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode("utf-8"))

            code1 = data.get("code1", "")
            code2 = data.get("code2", "")

            if not code1 or not code2:
                self.send_response(400)
                self.end_headers()
                self.wfile.write(b'{"error":"Missing input code"}')
                return

            # ✅ Run clone detection
            result = detect_clones(code1, code2)

            # ✅ Send headers and output
            self._set_headers()
            self.wfile.write(json.dumps(result).encode("utf-8"))

        except Exception as e:
            self.send_response(500)
            self.end_headers()
            self.wfile.write(json.dumps({"error": str(e)}).encode("utf-8"))


def run():
    print(f"🚀 Starting backend server at http://{HOST}:{PORT}")
    server = HTTPServer((HOST, PORT), CloneServer)
    print("✅ Backend is running. Waiting for requests...")
    server.serve_forever()


if __name__ == "__main__":
    run()
