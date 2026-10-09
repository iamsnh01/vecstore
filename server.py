"""
vecstore REST API server.

A minimal HTTP server built on the standard library. No Flask, no FastAPI —
just http.server and json.

    python server.py                  # default port 8080
    python server.py --port 9000      # custom port

Endpoints
---------
POST   /collections                       Create a collection
GET    /collections                       List collections
POST   /collections/{name}/add            Add vectors
POST   /collections/{name}/query          Query vectors
DELETE /collections/{name}/{id}           Delete a vector
GET    /collections/{name}/count          Get collection size
"""

from __future__ import annotations

import argparse
import json
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Any, Dict, Optional

import numpy as np

from vecstore import VecStore


store = VecStore()


class VecStoreHandler(BaseHTTPRequestHandler):

    def _send_json(self, data: Any, status: int = 200) -> None:
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> Dict[str, Any]:
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length)
        return json.loads(raw) if raw else {}

    def _route(self) -> tuple[str, ...]:
        parts = [p for p in self.path.strip("/").split("/") if p]
        return tuple(parts)

    # -- GET ---------------------------------------------------------------

    def do_GET(self) -> None:  # noqa: N802
        route = self._route()
        try:
            if route == ("collections",):
                self._send_json({"collections": store.list_collections()})
            elif len(route) == 3 and route[0] == "collections" and route[2] == "count":
                col = store.get_collection(route[1])
                self._send_json({"count": col.count()})
            else:
                self._send_json({"error": "not found"}, 404)
        except KeyError as exc:
            self._send_json({"error": str(exc)}, 404)

    # -- POST --------------------------------------------------------------

    def do_POST(self) -> None:  # noqa: N802
        route = self._route()
        body = self._read_json()
        try:
            if route == ("collections",):
                col = store.create_collection(
                    name=body["name"],
                    dim=body["dim"],
                    metric=body.get("metric", "cosine"),
                )
                self._send_json({"created": col.name}, 201)

            elif (
                len(route) == 3
                and route[0] == "collections"
                and route[2] == "add"
            ):
                col = store.get_collection(route[1])
                ids = body["ids"]
                vectors = body["vectors"]
                metadatas = body.get("metadatas")
                col.add_batch(ids, vectors, metadatas)
                self._send_json({"added": len(ids)})

            elif (
                len(route) == 3
                and route[0] == "collections"
                and route[2] == "query"
            ):
                col = store.get_collection(route[1])
                results = col.query(
                    vector=body["vector"],
                    k=body.get("k", 10),
                    where=body.get("where"),
                )
                self._send_json({"results": results})
            else:
                self._send_json({"error": "not found"}, 404)
        except KeyError as exc:
            self._send_json({"error": str(exc)}, 404)
        except (ValueError, TypeError) as exc:
            self._send_json({"error": str(exc)}, 400)

    # -- DELETE ------------------------------------------------------------

    def do_DELETE(self) -> None:  # noqa: N802
        route = self._route()
        try:
            if (
                len(route) == 3
                and route[0] == "collections"
            ):
                col = store.get_collection(route[1])
                deleted = col.delete(route[2])
                self._send_json({"deleted": deleted})
            else:
                self._send_json({"error": "not found"}, 404)
        except KeyError as exc:
            self._send_json({"error": str(exc)}, 404)

    def log_message(self, format: str, *args: Any) -> None:
        print(f"[vecstore] {args[0]} {args[1]} {args[2]}")


def main() -> None:
    parser = argparse.ArgumentParser(description="vecstore REST API")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    args = parser.parse_args()

    server = HTTPServer((args.host, args.port), VecStoreHandler)
    print(f"vecstore server listening on {args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nshutting down")
        server.server_close()


if __name__ == "__main__":
    main()
