# -*- coding: utf-8 -*-
"""memex HTTP 接入层：只绑 127.0.0.1:8901，给面板（M7）与别的 agent 直连用。

接口（与 §3.6 一致，只多 core 三个）：
    GET  /healthz              → {"ok":true,...}
    POST /recall    {q,k,scope} → 同 MCP 的 recall
    POST /remember  {text,scope,kind,importance,source,meta}
    GET|POST /stats            → 同 MCP 的 stats

实现复用 server_mcp 的 op_*（一份逻辑两个入口，别各写一遍）。
安全：bind 只到 127.0.0.1；非本机 Host 头直接 403；body 上限 1MB；无鉴权是刻意的（本机回环）。

用法：
    $py -m memex.http               # 默认 127.0.0.1:8901
    $py -m memex.http --port 8903 --db data/memex.db
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from urllib.parse import unquote
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from . import server_mcp as S  # noqa: E402

ROOT_DATA = Path(__file__).resolve().parent.parent / "data"

BIND = "127.0.0.1"
PORT = 8901
# ---- M7 面板（只读）：静态目录 + 类型表。加在 8901 的四个接口旁边，互不影响。----
PANEL_DIR = Path(__file__).resolve().parent.parent / "panel"
PANEL_TYPES = {".html": "text/html; charset=utf-8", ".css": "text/css; charset=utf-8",
               ".js": "application/javascript; charset=utf-8", ".json": "application/json; charset=utf-8",
               ".svg": "image/svg+xml", ".png": "image/png", ".ico": "image/x-icon"}
MAX_BODY = 1 << 20
ALLOW_HOSTS = {"127.0.0.1", "localhost"}


def _json_body(h: BaseHTTPRequestHandler) -> dict:
    """读 JSON body。超限抛 BodyTooLarge（由调用方回 413 并断开连接），坏 JSON 抛 ValueError。"""
    n = int(h.headers.get("Content-Length") or 0)
    if n <= 0:
        return {}
    if n > MAX_BODY:
        raise BodyTooLarge(f"body 太大：{n} 字节 > 上限 {MAX_BODY} 字节（1MB）")
    raw = h.rfile.read(n)
    try:
        return json.loads(raw.decode("utf-8") or "{}")
    except UnicodeDecodeError:
        raise ValueError("body 不是 UTF-8")
    except json.JSONDecodeError as e:
        raise ValueError(f"body 不是合法 JSON：{e}")


class BodyTooLarge(Exception):
    """请求体超过 MAX_BODY。"""


def _latest_baseline() -> dict:
    """最新一份 dev 口径报表（自动找 data/*.json，优先带 scope=none 的）。

    面板最容易骗人的地方就是「拿旧数字做决定」，所以基线不写死：
    每次读盘找最新评测产物，并把文件名、时间、口径一起显示出来。
    """
    best = None
    for f in sorted(ROOT_DATA.glob("*.json")):
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(d, dict) or d.get("set") not in ("dev", None):
            continue
        allm = d.get("all") or {}
        if not isinstance(allm, dict) or allm.get("top1") is None:
            continue
        official = f.name.startswith("memex-eval-") or f.name.startswith("p3-")
        cand = {"file": f.name, "official": official,
                "run_at": d.get("run_at") or d.get("at") or "",
                "set": d.get("set") or "?", "n": allm.get("n") or d.get("n"),
                "top1": allm.get("top1"), "r5": allm.get("r5"),
                "p95_ms": allm.get("p95_ms"), "scope": d.get("scope"),
                "arms": d.get("arms"),
                "policy": {k: v for k, v in (d.get("policy") or {}).items()
                           if k.startswith("w_") or k in ("use_rerank", "rerank_backend")}}
        # 优先官方 eval_run 产物（memex-eval-*/p3-*），再优先带 scope=none 的；同一档里取最新
        key = (1 if official else 0, 1 if d.get("scope") == "none" else 0, str(cand["run_at"]))
        if best is None or key > best[0]:
            best = (key, cand)
    return best[1] if best else {}


def panel_payload(node: str = "", days: int = 60) -> dict:
    """面板所需的全部只读数据：策略来源 / 库规模 / 基线 / 衰减分布 / 选中节点的边。

    用 `mode=ro` 打开库：面板永远不该写生产库（连 -wal 也不该碰）。
    """
    import math
    import sqlite3

    from .policy_io import ACTIVE_PATH, active_path, read_active_meta
    from .recall import DEFAULT_POLICY

    path = S._db_path()
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    now = int(time.time())
    try:
        counts = {
            "active": con.execute("SELECT count(*) c FROM memories WHERE status='active'").fetchone()["c"],
            "all": con.execute("SELECT count(*) c FROM memories").fetchone()["c"],
            "edges": con.execute("SELECT count(*) c FROM edges").fetchone()["c"],
            "edge_pairs": con.execute(
                "SELECT count(*) c FROM (SELECT DISTINCT min(a,b) x, max(a,b) y FROM edges)").fetchone()["c"],
            "scopes": {r["scope"]: r["c"] for r in con.execute(
                "SELECT scope, count(*) c FROM memories WHERE status='active' GROUP BY scope")},
        }
        rows = con.execute("SELECT importance, stability, decay_tier, last_access, created_at "
                           "FROM memories WHERE status='active'").fetchall()
        buckets: dict[int, list[float]] = {}
        for r in rows:
            age = max(0.0, (now - (r["last_access"] or r["created_at"] or now)) / 86400.0)
            st = float(r["stability"] or 30.0) or 30.0
            val = float(r["importance"] or 0.5) * math.exp(-age / st)
            buckets.setdefault(min(int(days), int(age)), []).append(val)
        series = [{"day": d, "avg": round(sum(v) / len(v), 4), "n": len(v)}
                  for d, v in sorted(buckets.items())]

        tiers: dict[int, dict] = {}
        for r in rows:
            t = int(r["decay_tier"] or 0)
            e = tiers.setdefault(t, {"tier": t, "stability": float(r["stability"] or 0), "n": 0})
            e["n"] += 1
        curves = []
        for t in sorted(tiers):
            st = tiers[t]["stability"] or 30.0
            curves.append({"tier": t, "stability": st, "n": tiers[t]["n"],
                           "points": [round(math.exp(-d / st), 4)
                                      for d in range(0, days + 1, max(1, days // 20))]})

        meta = read_active_meta() or {}
        # 重排落在哪儿：本地保险服务（127.0.0.1:8903）还是远程硅基流动。
        # 面板不写死这个判断 —— 口径变了（比如临时切远程做对照）要让眼睛能看出来。
        try:
            from urllib.parse import urlparse
            from .rerank import _api_base

            api_base = _api_base(meta) if meta.get("use_rerank") else ""
            host = (urlparse(api_base).hostname or "") if api_base else ""
        except Exception:
            api_base, host = (meta.get("rerank_api_base") or ""), ""
        where = ("关" if not meta.get("use_rerank")
                 else ("本地" if host in ("127.0.0.1", "localhost", "::1") else "远程"))
        payload = {
            "policy": {
                "source": ("data/active_policy.json" if active_path().exists() else "DEFAULT_POLICY（内置）"),
                "path": str(active_path()),
                "version": meta.get("version"),
                "weights": {k: meta.get(k, DEFAULT_POLICY.get(k))
                            for k in ("w_dense", "w_bm25", "w_act", "w_graph")},
                "arms": meta.get("arms") or DEFAULT_POLICY.get("arms"),
                "use_rerank": meta.get("use_rerank", False),
                # 重排链路：面板一眼要能看出「本地 8903 / 远程 / 关」，以及池大小与并发片数
                "rerank_backend": meta.get("rerank_backend") or "lexical",
                "rerank_api_base": api_base,
                "rerank_host": host,
                "rerank_where": where,
                "rerank_pool": meta.get("rerank_pool"),
                "api_parallel_n": meta.get("api_parallel_n"),
                "api_deadline_ms": meta.get("api_deadline_ms"),
                "cache_doc_chars": meta.get("cache_doc_chars"),
                "query_strip": bool(meta.get("query_strip", False)),
                "default_path_hint": str(ACTIVE_PATH),
            },
            "db": {"path": str(path), **counts},
            "baseline": _latest_baseline(),
            "decay": {"days": days, "series": series, "tiers": curves},
            "node": None,
            "hubs": [{"id": r["id"], "text": (r["text"] or "")[:120], "deg": r["deg"]}
                     for r in con.execute(
                         "SELECT m.id, m.text, count(e.a) deg FROM memories m "
                         "JOIN edges e ON e.a = m.id WHERE m.status='active' "
                         "GROUP BY m.id ORDER BY deg DESC LIMIT 8")],
        }
        if node:
            row = con.execute("SELECT id, scope, kind, text, importance, decay_tier, stability, "
                              "last_access FROM memories WHERE id=?", (node,)).fetchone()
            if row:
                nb = con.execute(
                    "SELECT e.weight, e.kind ek, m.id, m.scope, m.text, m.importance FROM edges e "
                    "JOIN memories m ON m.id = (CASE WHEN e.a=? THEN e.b ELSE e.a END) "
                    "WHERE (e.a=? OR e.b=?) AND m.status='active' "
                    "ORDER BY e.weight DESC LIMIT 12", (node, node, node)).fetchall()
                payload["node"] = {
                    **dict(row),
                    "neighbors": [{"id": r["id"], "scope": r["scope"],
                                   "text": (r["text"] or "")[:200], "weight": r["weight"],
                                   "edge_kind": r["ek"], "importance": r["importance"]} for r in nb],
                }
        return payload
    finally:
        con.close()


class Handler(BaseHTTPRequestHandler):
    server_version = "memex-http/1.0"
    protocol_version = "HTTP/1.1"

    # ---- 基础设施 ----
    def _guard(self) -> bool:
        host = (self.headers.get("Host") or "").split(":")[0].strip().lower()
        if host and host not in ALLOW_HOSTS:
            self._send(403, {"ok": False, "error": f"只接受本机访问，收到 Host={host}"})
            return False
        return True

    def _send(self, code: int, obj, close: bool = False) -> None:
        body = (obj if isinstance(obj, str) else json.dumps(obj, ensure_ascii=False)).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        if close:
            # body 没读完就回错误，必须断开，否则 keep-alive 里残留字节会被当成下一个请求
            self.send_header("Connection", "close")
            self.close_connection = True
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):  # 静音默认的 stderr 行，失败才记
        pass

    def _send_bytes(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    # ---- M7 面板（只读；不写库、不碰召回）----
    def _panel_static(self, path: str) -> None:
        # `/` 和 `/panel` 都是首页。只剥 `panel/` 前缀不够：`/panel` 剥完还是 `panel`，
        # 会被当成 panel/panel 这个文件 → 404，而 `/panel/data` 是 200，
        # 现场看起来就像「服务坏了、接口在但页面没了」。
        rel = "index.html" if path in ("/", "", "/panel") else path.lstrip("/")
        if rel.startswith("panel/"):
            rel = rel[len("panel/"):]
        if rel in ("", "panel"):          # `/panel/` 也归首页
            rel = "index.html"
        root = PANEL_DIR.resolve()
        target = (root / rel).resolve()
        if root not in target.parents and target != root:
            self._send(403, {"ok": False, "error": "只允许读取 panel/ 下的文件"})
            return
        if not target.is_file():
            self._send(404, {"ok": False, "error": f"面板文件不存在：{rel}"})
            return
        ctype = PANEL_TYPES.get(target.suffix.lower(), "application/octet-stream")
        self._send_bytes(200, target.read_bytes(), ctype)

    def _panel_data(self) -> None:
        qs = {}
        if "?" in (self.path or ""):
            from urllib.parse import parse_qs
            qs = {k: v[-1] for k, v in parse_qs(self.path.split("?", 1)[1]).items()}
        self._send(200, panel_payload(node=(qs.get("node") or "").strip(),
                                      days=int(qs.get("days") or 60)))

    def _dispatch(self, method: str) -> None:
        if not self._guard():
            return
        path = unquote((self.path or "/").split("?")[0]).rstrip("/") or "/"
        try:
            if method == "GET" and path == "/healthz":
                st = S.op_stats()
                self._send(200 if st.get("ok") else 503, {
                    "ok": bool(st.get("ok")),
                    "service": "memex",
                    "ts": int(time.time()),
                    "memories": (st.get("memories") or {}).get("active", 0),
                    "db": st.get("db", ""),
                })
                return
            if method in ("GET", "POST") and path == "/stats":
                self._send(200, S.op_stats())
                return
            if method == "POST" and path == "/recall":
                b = _json_body(self)
                self._send(200, S.op_recall(q=b.get("q", ""), k=int(b.get("k", 5) or 5),
                                            scope=b.get("scope", "")))
                return
            if method == "GET" and (path == "/" or path == "/panel" or path.startswith("/panel/")):
                if path == "/panel/data":
                    self._panel_data()
                else:
                    self._panel_static(path)
                return
            if method == "GET" and path == "/core":
                qs = {}
                if "?" in (self.path or ""):
                    from urllib.parse import parse_qs
                    qs = {k: v[-1] for k, v in parse_qs(self.path.split("?", 1)[1]).items()}
                self._send(200, S.op_core(budget=int(qs.get("budget") or 1200)))
                return
            if method == "POST" and path == "/core/pin":
                b = _json_body(self)
                self._send(200, S.op_core_pin(id=b.get("id", ""), slot=b.get("slot", "other")))
                return
            if method == "POST" and path == "/core/unpin":
                b = _json_body(self)
                self._send(200, S.op_core_unpin(id=b.get("id", "")))
                return
            if path.startswith("/memory/"):
                mid = path[len("/memory/"):].strip()
                if not mid:
                    self._send(400, {"ok": False, "error": "要带记忆 id：/memory/<id>"})
                    return
                def _st(resp: dict) -> int:
                    if resp.get("ok"):
                        return 200
                    return 404 if "没有 id" in (resp.get("error") or "") else 400

                if method == "GET":
                    r = S.op_memory_get(id=mid)
                    self._send(_st(r), r)
                    return
                if method == "PUT":
                    b = _json_body(self)
                    r = S.op_memory_update(id=mid, text=b.get("text", ""),
                                           importance=b.get("importance"))
                    self._send(_st(r), r)
                    return
                if method == "DELETE":
                    b = _json_body(self)
                    r = S.op_memory_forget(id=mid, reason=b.get("reason", "panel-delete"))
                    self._send(_st(r), r)
                    return
                self._send(405, {"ok": False, "error": f"/memory 只支持 GET/PUT/DELETE，不支持 {method}"})
                return
            if method == "GET" and path == "/traces":
                from urllib.parse import parse_qs
                qs = parse_qs((self.path or "").split("?", 1)[1]) if "?" in (self.path or "") else {}
                lim = int((qs.get("limit") or ["20"])[0])
                self._send(200, S.op_traces(limit=lim))
                return
            if method == "GET" and path.startswith("/trace/"):
                tid = path[len("/trace/"):].strip()
                r = S.op_trace(trace_id=tid)
                self._send(200 if r.get("ok") else 404, r)
                return
            if method == "POST" and path == "/feedback":
                b = _json_body(self)
                r = S.op_feedback(trace_id=b.get("trace_id", ""),
                                  hit_ids=b.get("hit_ids") or [],
                                  used=bool(b.get("used", True)))
                self._send(200 if r.get("ok") else 400, r)
                return
            if method == "POST" and path == "/remember":
                b = _json_body(self)
                self._send(200, S.op_remember(
                    text=b.get("text", ""), scope=b.get("scope", "facts"),
                    kind=b.get("kind", "fact"), importance=b.get("importance", ""),
                    source=b.get("source", "http"), meta=b.get("meta", "")))
                return
            self._send(404, {"ok": False, "error": f"没有 {method} {path}；"
                                                   "只有 /healthz /recall /remember /stats /core /core/pin /core/unpin /memory/<id>（GET/PUT/DELETE）"})
        except BodyTooLarge as e:
            self._send(413, {"ok": False, "error": str(e)}, close=True)
        except ValueError as e:
            self._send(400, {"ok": False, "error": str(e)})
        except Exception as e:
            self._send(500, {"ok": False, "error": f"{type(e).__name__}: {e}"})

    def do_GET(self):  # noqa: N802
        self._dispatch("GET")

    def do_POST(self):  # noqa: N802
        self._dispatch("POST")

    def do_PUT(self):  # noqa: N802
        self._dispatch("PUT")

    def do_DELETE(self):  # noqa: N802
        self._dispatch("DELETE")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="memex HTTP（只绑 127.0.0.1）")
    ap.add_argument("--port", type=int, default=PORT)
    ap.add_argument("--db", default=None)
    ap.add_argument("--no-warm", action="store_true", help="不预热（默认预热一次，首屏更快）")
    args = ap.parse_args(argv)

    if args.db:
        os.environ[S.DB_ENV] = str(Path(args.db).resolve())
    if not args.no_warm:
        try:
            S.ENGINE.boot()
            S.ENGINE.recall.load()  # type: ignore[union-attr]
            from .rerank import warmup as _rerank_warmup  # noqa: PLC0415
            _warm = _rerank_warmup(getattr(S.ENGINE.recall, "policy", None))
            print(f"[memex.http] 重排预热 {_warm}", flush=True)
        except Exception as e:
            print(f"[memex.http] 预热失败（不阻塞启动）：{type(e).__name__}: {e}", file=sys.stderr)

    srv = ThreadingHTTPServer((BIND, args.port), Handler)
    srv.daemon_threads = True
    print(f"[memex.http] listening http://{BIND}:{args.port}  db={S._db_path()}", flush=True)
    if (PANEL_DIR / "index.html").exists():
        print(f"[memex.http] 面板 http://{BIND}:{args.port}/  （只读）", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
