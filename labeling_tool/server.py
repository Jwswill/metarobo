"""YOLO 바운딩박스 라벨링 툴 (외부 의존성 없이 표준 라이브러리만 사용).

사용법:
    python labeling_tool/server.py
    -> http://localhost:8765 접속
"""
import json
import mimetypes
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CAP_IMAGE_DIR = os.path.join(ROOT, "cap_image")
STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
CLASSES = ["safety", "danger", "corn"]  # index == YOLO class_id
IMAGE_EXTS = (".jpg", ".jpeg", ".png")


def safe_dir_name(name: str) -> str:
    """디렉터리 순회 공격 방지: 경로 구분자가 없는 단순 이름만 허용."""
    if not name or "/" in name or "\\" in name or ".." in name:
        raise ValueError("invalid dir name")
    return name


def list_video_dirs():
    result = []
    if not os.path.isdir(CAP_IMAGE_DIR):
        return result
    for name in sorted(os.listdir(CAP_IMAGE_DIR)):
        image_dir = os.path.join(CAP_IMAGE_DIR, name, "image")
        json_dir = os.path.join(CAP_IMAGE_DIR, name, "json")
        if not os.path.isdir(image_dir):
            continue
        images = [f for f in os.listdir(image_dir) if f.lower().endswith(IMAGE_EXTS)]
        labeled = 0
        if os.path.isdir(json_dir):
            json_names = {os.path.splitext(f)[0] for f in os.listdir(json_dir) if f.endswith(".json")}
            labeled = sum(1 for f in images if os.path.splitext(f)[0] in json_names)
        result.append({"name": name, "total": len(images), "labeled": labeled})
    return result


def list_images(dir_name: str):
    image_dir = os.path.join(CAP_IMAGE_DIR, dir_name, "image")
    json_dir = os.path.join(CAP_IMAGE_DIR, dir_name, "json")
    if not os.path.isdir(image_dir):
        return []
    images = sorted(f for f in os.listdir(image_dir) if f.lower().endswith(IMAGE_EXTS))
    out = []
    for f in images:
        stem = os.path.splitext(f)[0]
        json_path = os.path.join(json_dir, stem + ".json")
        box_count = 0
        if os.path.exists(json_path):
            try:
                with open(json_path, "r", encoding="utf-8") as fp:
                    data = json.load(fp)
                box_count = len(data.get("annotations", []))
            except (OSError, json.JSONDecodeError):
                box_count = 0
        out.append({"name": f, "labeled": os.path.exists(json_path), "box_count": box_count})
    return out


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass  # 콘솔 소음 억제

    def _send_json(self, payload, status=200):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_error_json(self, message, status=400):
        self._send_json({"error": message}, status)

    def _send_file(self, path, content_type=None):
        if not os.path.isfile(path):
            self._send_error_json("not found", 404)
            return
        if content_type is None:
            content_type = mimetypes.guess_type(path)[0] or "application/octet-stream"
        with open(path, "rb") as f:
            data = f.read()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        qs = parse_qs(parsed.query)

        try:
            if path == "/api/classes":
                self._send_json({"classes": CLASSES})
            elif path == "/api/dirs":
                self._send_json({"dirs": list_video_dirs()})
            elif path == "/api/images":
                dir_name = safe_dir_name(qs.get("dir", [""])[0])
                self._send_json({"images": list_images(dir_name)})
            elif path == "/api/image":
                dir_name = safe_dir_name(qs.get("dir", [""])[0])
                name = qs.get("name", [""])[0]
                if "/" in name or "\\" in name or ".." in name:
                    raise ValueError("invalid image name")
                img_path = os.path.join(CAP_IMAGE_DIR, dir_name, "image", name)
                self._send_file(img_path)
            elif path == "/api/annotation":
                dir_name = safe_dir_name(qs.get("dir", [""])[0])
                name = qs.get("name", [""])[0]
                stem = os.path.splitext(name)[0]
                json_path = os.path.join(CAP_IMAGE_DIR, dir_name, "json", stem + ".json")
                if os.path.exists(json_path):
                    with open(json_path, "r", encoding="utf-8") as f:
                        self._send_json(json.load(f))
                else:
                    self._send_json({"image": name, "width": None, "height": None, "annotations": []})
            elif path.startswith("/api/"):
                self._send_error_json("unknown endpoint", 404)
            else:
                self._serve_static(path)
        except ValueError as e:
            self._send_error_json(str(e), 400)
        except Exception as e:  # noqa: BLE001
            self._send_error_json(str(e), 500)

    def _serve_static(self, url_path):
        if url_path == "/":
            url_path = "/index.html"
        rel = url_path.lstrip("/")
        full = os.path.normpath(os.path.join(STATIC_DIR, rel))
        if not full.startswith(STATIC_DIR):
            self._send_error_json("forbidden", 403)
            return
        self._send_file(full)

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path != "/api/annotation":
            self._send_error_json("unknown endpoint", 404)
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
            raw = self.rfile.read(length)
            payload = json.loads(raw.decode("utf-8"))

            dir_name = safe_dir_name(payload.get("dir", ""))
            image_name = payload.get("image", "")
            if not image_name or "/" in image_name or "\\" in image_name:
                raise ValueError("invalid image name")

            annotations = payload.get("annotations", [])
            for ann in annotations:
                if ann.get("class_id") not in range(len(CLASSES)):
                    raise ValueError("invalid class_id")

            json_dir = os.path.join(CAP_IMAGE_DIR, dir_name, "json")
            os.makedirs(json_dir, exist_ok=True)
            stem = os.path.splitext(image_name)[0]
            out_path = os.path.join(json_dir, stem + ".json")

            record = {
                "image": image_name,
                "width": payload.get("width"),
                "height": payload.get("height"),
                "annotations": annotations,
            }
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(record, f, ensure_ascii=False, indent=2)

            self._send_json({"ok": True})
        except ValueError as e:
            self._send_error_json(str(e), 400)
        except Exception as e:  # noqa: BLE001
            self._send_error_json(str(e), 500)


def main():
    port = int(os.environ.get("PORT", 8765))
    server = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    print(f"라벨링 툴 실행 중: http://localhost:{port}")
    print(f"cap_image 경로: {CAP_IMAGE_DIR}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
