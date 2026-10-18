"""One local interface for preparation, live observation, and durable replay."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import re
import threading
from urllib.parse import parse_qs, urlparse

from .recorder import TraceReader, json_text
from .workflow import RunManager, RunConflict

STATIC = Path(__file__).parent / "web"


class ViewerHandler(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def send_data(self, value, mime="application/json; charset=utf-8", status=200):
        body = value.encode("utf-8") if isinstance(value, str) else value
        self.send_response(status)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def send_file(self, path, mime):
        with Path(path).open("rb") as handle:
            self.send_response(200)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(Path(path).stat().st_size))
            self.end_headers()
            while chunk := handle.read(65536):
                self.wfile.write(chunk)

    def trace_path(self, run_id):
        return self.server.manager.trace(self.server.manager.get(run_id)) if run_id else self.server.trace_dir

    def do_GET(self):
        request = urlparse(self.path)
        query = parse_qs(request.query)
        value = lambda name, default=None: query.get(name, [default])[0]
        path = request.path
        try:
            if path in {"/", "/app.js", "/style.css"}:
                filename, mime = {"/": ("index.html", "text/html; charset=utf-8"),
                                  "/app.js": ("app.js", "text/javascript; charset=utf-8"),
                                  "/style.css": ("style.css", "text/css; charset=utf-8")}[path]
                self.send_data((STATIC / filename).read_bytes(), mime)
                return
            if path == "/api/context":
                result = dict(fixed=bool(self.server.trace_dir))
                if self.server.manager:
                    result.update(self.server.manager.metadata())
            elif path == "/api/runs":
                if not self.server.manager:
                    raise ValueError("固定记录模式没有运行列表")
                result = dict(runs=self.server.manager.list())
            elif match := re.fullmatch(r"/api/runs/([a-f0-9]{16})(?:/(log|replay|video|final|svg|pdf))?", path):
                run_id, kind = match.groups()
                if kind in {"replay", "video", "final", "svg", "pdf"}:
                    output = Path(self.server.manager.get(run_id)["output_dir"])
                    names = {"replay": ("replay.html", "text/html; charset=utf-8"),
                             "video": ("process.mp4", "video/mp4"), "final": ("final.json", "application/json"),
                             "svg": ("publication_figures/final_molecule.svg", "image/svg+xml"),
                             "pdf": ("publication_figures/final_molecule.pdf", "application/pdf")}
                    filename, mime = names[kind]
                    if kind in {"svg", "pdf"} and not (output / filename).is_file():
                        filename = f"final_molecule.{kind}"
                    self.send_file(output / filename, mime)
                    return
                result = dict(log=self.server.manager.log_tail(run_id)) if kind == "log" else self.server.manager.info(run_id)
            elif path.startswith("/api/trace/"):
                trace = self.trace_path(value("run"))
                endpoint = path.removeprefix("/api/trace/")
                if endpoint == "stream":
                    self.stream(trace, int(value("after", self.headers.get("Last-Event-ID", "0"))))
                    return
                with TraceReader(trace) as reader:
                    if endpoint == "manifest":
                        result = dict(reader.manifest(), latest_sequence=reader.latest_sequence())
                        if value("run") and self.server.manager:
                            info = self.server.manager.info(value("run"))
                            result["status"] = info["status"]
                            if info.get("error"):
                                result["error"] = info["error"]
                    elif endpoint == "resources":
                        result = reader.resources()
                    elif endpoint == "phases":
                        result = reader.phases()
                    elif endpoint == "events":
                        result = dict(events=reader.events(after=int(value("after", 0)), limit=int(value("limit", 250)),
                                      mode=value("mode", "live"), before=value("before")), latest_sequence=reader.latest_sequence(), phases=reader.phases())
                    elif endpoint == "position":
                        result = reader.position(int(value("at", 0)), value("mode", "live"))
                    elif endpoint.startswith("state/"):
                        result = reader.state(int(endpoint.split("/")[1]))
                    else:
                        self.send_data(json_text(dict(error="接口不存在")), status=404)
                        return
            else:
                self.send_data(json_text(dict(error="接口不存在")), status=404)
                return
            self.send_data(json_text(result))
        except RunConflict as exc:
            self.send_data(json_text(dict(error=str(exc))), status=409)
        except (FileNotFoundError, KeyError) as exc:
            self.send_data(json_text(dict(error=str(exc))), status=404)
        except (ValueError, TypeError) as exc:
            self.send_data(json_text(dict(error=str(exc))), status=400)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_POST(self):
        try:
            origin = self.headers.get("Origin")
            if origin and urlparse(origin).netloc != self.headers.get("Host"):
                self.send_data(json_text(dict(error="请求来源与当前界面不一致")), status=403)
                return
            if not self.server.manager:
                raise RunConflict("已有记录查看模式不启动计算，请使用 ui 入口")
            if self.headers.get_content_type() != "application/json":
                raise ValueError("请求必须使用 application/json")
            length = int(self.headers.get("Content-Length", 0))
            if not 0 < length <= 1024 * 1024:
                raise ValueError("参数大小无效")
            def reject(value):
                raise ValueError(f"无效数值: {value}")
            payload = json.loads(self.rfile.read(length), parse_constant=reject)
            if not isinstance(payload, dict):
                raise ValueError("参数必须是对象")
            path = urlparse(self.path).path
            status = 200
            if path == "/api/prepare":
                result, status = self.server.manager.prepare(payload), 201
            elif path == "/api/open":
                result = self.server.manager.reopen(payload["path"])
            elif match := re.fullmatch(r"/api/runs/([a-f0-9]{16})/(start|export)", path):
                run_id, kind = match.groups()
                if kind == "start":
                    result = self.server.manager.start(run_id, payload)
                else:
                    info = self.server.manager.info(run_id)
                    if info["status"] in {"running", "starting", "preparing", "ready"}:
                        raise RunConflict("组装结束后再导出完整过程")
                    from .export import export_html, export_frames, export_video, export_molecule
                    spec = self.server.manager.get(run_id)
                    trace, output = self.server.manager.trace(spec), Path(spec["output_dir"])
                    with self.server.manager.export_lock:
                        export_html(trace, output / "replay.html")
                        files = export_frames(trace, output / "process_figures")
                        final_figures = export_molecule(output / "final_molecule.json")
                        if payload.get("video"):
                            export_video(trace, output / "process.mp4", scope=payload.get("mode", "formal"))
                    result = dict(html_url=f"/api/runs/{run_id}/replay", figure_dir=str(output / "process_figures"),
                                  figure_count=len(files), final_url=f"/api/runs/{run_id}/final")
                    result.update(svg_url=f"/api/runs/{run_id}/svg", pdf_url=f"/api/runs/{run_id}/pdf",
                                  final_figures=final_figures)
                    if payload.get("video"):
                        result["video_url"] = f"/api/runs/{run_id}/video"
            else:
                self.send_data(json_text(dict(error="接口不存在")), status=404)
                return
            self.send_data(json_text(result), status=status)
        except RunConflict as exc:
            self.send_data(json_text(dict(error=str(exc))), status=409)
        except (KeyError, FileNotFoundError, TypeError, ValueError) as exc:
            self.send_data(json_text(dict(error=str(exc))), status=400)
        except RuntimeError as exc:
            self.send_data(json_text(dict(error=str(exc))), status=422)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def stream(self, trace, after):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        try:
            while not self.server.stopping.is_set():
                with TraceReader(trace) as reader:
                    seq, status = reader.latest_sequence(), reader.manifest()["status"]
                if seq > after:
                    self.wfile.write(f"id: {seq}\ndata: {json_text(dict(latest_sequence=seq,status=status))}\n\n".encode())
                    after = seq
                else:
                    self.wfile.write(b": heartbeat\n\n")
                if status not in {"running", "preparing"}:
                    self.wfile.write(b"event: finished\ndata: {}\n\n")
                    self.wfile.flush()
                    return
                self.wfile.flush()
                self.server.stopping.wait(.5)
        except (BrokenPipeError, ConnectionResetError):
            pass


class ViewerServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, *, project=None, defaults=None, trace_dir=None, host="127.0.0.1", port=8765):
        self.trace_dir = Path(trace_dir).resolve() if trace_dir else None
        if self.trace_dir:
            with TraceReader(self.trace_dir):
                pass
        self.manager = None if trace_dir else RunManager(project, defaults)
        self.stopping = threading.Event()
        super().__init__((host, port), ViewerHandler)

    @property
    def url(self):
        host, port = self.server_address
        return f"http://{host}:{port}"

    def server_close(self):
        self.stopping.set()
        super().server_close()


def start_ui(project=None, *, config=None, host="127.0.0.1", port=8765):
    from dataclasses import asdict
    server = ViewerServer(project=project, defaults=asdict(config) if config else None, host=host, port=port)
    threading.Thread(target=server.serve_forever, daemon=True, name="assembly-interface").start()
    return server


def open_workbench(sample="YM-3", *, config=None, project=None, port=0):
    """Notebook entry: open the same UI; reading and computation require clicks."""
    from urllib.parse import urlencode
    server = start_ui(project, config=config, port=port)
    url = server.url + "/?" + urlencode(dict(sample=sample))
    print(f"装配界面: {url}")
    try:
        from IPython.display import Markdown, display
        display(Markdown(f"[打开拓扑读取、实时组装和回放界面]({url})"))
    except ImportError:
        pass
    return server


def start_server(trace_dir, host="127.0.0.1", port=8765):
    """Read-only server for a specific trace, using the same single-canvas page."""
    server = ViewerServer(trace_dir=trace_dir, host=host, port=port)
    threading.Thread(target=server.serve_forever, daemon=True, name="assembly-replay").start()
    return server


def serve(*, project=None, trace_dir=None, host="127.0.0.1", port=8765):
    with ViewerServer(project=project, trace_dir=trace_dir, host=host, port=port) as server:
        print(f"装配界面: {server.url}", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
