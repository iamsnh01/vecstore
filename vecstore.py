"""
vecstore — A vector database from scratch in Python.

HNSW (Hierarchical Navigable Small World) index with persistence,
metadata filtering, and batch operations. Only depends on numpy.
"""

from __future__ import annotations

import heapq
import json
import math
import os
import random
import struct
import threading
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import numpy as np


# ---------------------------------------------------------------------------
# Distance metrics
# ---------------------------------------------------------------------------

class Metric(Enum):
    COSINE = "cosine"
    EUCLIDEAN = "euclidean"
    DOT = "dot"


def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0.0 or norm_b == 0.0:
        return 1.0
    return 1.0 - float(np.dot(a, b) / (norm_a * norm_b))


def _euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def _dot_distance(a: np.ndarray, b: np.ndarray) -> float:
    # Negative dot product so that higher similarity = lower distance.
    return -float(np.dot(a, b))


_METRIC_FN: Dict[Metric, Callable[[np.ndarray, np.ndarray], float]] = {
    Metric.COSINE: _cosine_distance,
    Metric.EUCLIDEAN: _euclidean_distance,
    Metric.DOT: _dot_distance,
}


# ---------------------------------------------------------------------------
# HNSW graph node
# ---------------------------------------------------------------------------

@dataclass
class _Node:
    id: str
    vector: np.ndarray
    metadata: Dict[str, Any] = field(default_factory=dict)
    # neighbors[level] -> set of node ids
    neighbors: Dict[int, Set[str]] = field(default_factory=dict)
    level: int = 0


# ---------------------------------------------------------------------------
# HNSW index
# ---------------------------------------------------------------------------

class HNSWIndex:
    """Hierarchical Navigable Small World graph index.

    Parameters
    ----------
    dim : int
        Dimensionality of vectors.
    metric : Metric
        Distance metric (cosine, euclidean, dot).
    M : int
        Max number of neighbors per node per layer (default 16).
    ef_construction : int
        Size of the dynamic candidate list during construction (default 200).
    ef_search : int
        Size of the dynamic candidate list during search (default 50).
    ml : float
        Level generation factor (default 1/ln(M)).
    """

    def __init__(
        self,
        dim: int,
        metric: Metric = Metric.COSINE,
        M: int = 16,
        ef_construction: int = 200,
        ef_search: int = 50,
        ml: Optional[float] = None,
    ) -> None:
        self.dim = dim
        self.metric = metric
        self.M = M
        self.M_max0 = 2 * M  # max neighbors on layer 0
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self.ml = ml if ml is not None else 1.0 / math.log(M)

        self._dist_fn = _METRIC_FN[metric]
        self._nodes: Dict[str, _Node] = {}
        self._entry_point: Optional[str] = None
        self._max_level: int = -1
        self._lock = threading.Lock()

    # -- helpers ----------------------------------------------------------

    def _random_level(self) -> int:
        return int(-math.log(random.random()) * self.ml)

    def _distance(self, a: np.ndarray, b: np.ndarray) -> float:
        return self._dist_fn(a, b)

    # -- core HNSW operations --------------------------------------------

    def _search_layer(
        self,
        query: np.ndarray,
        entry_ids: List[str],
        ef: int,
        level: int,
    ) -> List[Tuple[float, str]]:
        """Greedy search on a single layer. Returns up to *ef* nearest
        neighbors as (distance, id) pairs sorted ascending by distance."""
        visited: Set[str] = set(entry_ids)
        candidates: List[Tuple[float, str]] = []
        result: List[Tuple[float, str]] = []

        for eid in entry_ids:
            d = self._distance(query, self._nodes[eid].vector)
            heapq.heappush(candidates, (d, eid))
            heapq.heappush(result, (-d, eid))  # max-heap for result bound

        while candidates:
            dist_c, c_id = heapq.heappop(candidates)
            # farthest in result
            furthest_dist = -result[0][0]
            if dist_c > furthest_dist:
                break

            node = self._nodes[c_id]
            neighbor_ids = node.neighbors.get(level, set())
            for n_id in neighbor_ids:
                if n_id in visited or n_id not in self._nodes:
                    continue
                visited.add(n_id)
                d = self._distance(query, self._nodes[n_id].vector)
                furthest_dist = -result[0][0]
                if d < furthest_dist or len(result) < ef:
                    heapq.heappush(candidates, (d, n_id))
                    heapq.heappush(result, (-d, n_id))
                    if len(result) > ef:
                        heapq.heappop(result)

        return sorted([(abs(d), nid) for d, nid in result])

    def _select_neighbors(
        self,
        query: np.ndarray,
        candidates: List[Tuple[float, str]],
        M: int,
    ) -> List[Tuple[float, str]]:
        """Simple neighbor selection — pick the M closest."""
        return sorted(candidates)[:M]

    def _connect(self, node_id: str, neighbor_id: str, level: int) -> None:
        self._nodes[node_id].neighbors.setdefault(level, set()).add(neighbor_id)
        self._nodes[neighbor_id].neighbors.setdefault(level, set()).add(node_id)

        # Prune neighbor's connections if they exceed the limit.
        max_conn = self.M_max0 if level == 0 else self.M
        neighbor = self._nodes[neighbor_id]
        if len(neighbor.neighbors[level]) > max_conn:
            # Keep the closest ones.
            scored = [
                (self._distance(neighbor.vector, self._nodes[nid].vector), nid)
                for nid in neighbor.neighbors[level]
                if nid in self._nodes
            ]
            scored.sort()
            neighbor.neighbors[level] = {nid for _, nid in scored[:max_conn]}

    # -- public API -------------------------------------------------------

    def insert(
        self,
        id: str,
        vector: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Insert a vector with the given id and optional metadata."""
        vector = np.asarray(vector, dtype=np.float32)
        if vector.shape != (self.dim,):
            raise ValueError(
                f"Expected vector of dim {self.dim}, got {vector.shape}"
            )
        if id in self._nodes:
            raise KeyError(f"ID '{id}' already exists. Delete it first.")

        level = self._random_level()
        node = _Node(id=id, vector=vector, metadata=metadata or {}, level=level)

        with self._lock:
            self._nodes[id] = node

            if self._entry_point is None:
                self._entry_point = id
                self._max_level = level
                return

            ep = [self._entry_point]

            # Traverse from top to the node's level + 1 using greedy search.
            for lc in range(self._max_level, level, -1):
                nearest = self._search_layer(vector, ep, ef=1, level=lc)
                ep = [nearest[0][1]]

            # Insert at layers [min(level, max_level) ... 0]
            for lc in range(min(level, self._max_level), -1, -1):
                candidates = self._search_layer(
                    vector, ep, ef=self.ef_construction, level=lc
                )
                max_conn = self.M_max0 if lc == 0 else self.M
                neighbors = self._select_neighbors(vector, candidates, max_conn)

                for _, n_id in neighbors:
                    self._connect(id, n_id, lc)

                ep = [n_id for _, n_id in candidates]

            if level > self._max_level:
                self._max_level = level
                self._entry_point = id

    def search(
        self,
        query: np.ndarray,
        k: int = 10,
        ef: Optional[int] = None,
        filter_fn: Optional[Callable[[Dict[str, Any]], bool]] = None,
        where: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Search for the k nearest neighbors.

        Parameters
        ----------
        query : array-like
            Query vector.
        k : int
            Number of results.
        ef : int, optional
            Override ef_search for this query.
        filter_fn : callable, optional
            Predicate on metadata dict; only matching vectors are returned.
        where : dict, optional
            Simple equality metadata filter (all keys must match).

        Returns
        -------
        list of (id, distance, metadata)
        """
        query = np.asarray(query, dtype=np.float32)
        if query.shape != (self.dim,):
            raise ValueError(
                f"Expected query of dim {self.dim}, got {query.shape}"
            )
        ef = ef or max(self.ef_search, k)

        if self._entry_point is None:
            return []

        if where is not None:
            def _where_filter(meta: Dict[str, Any]) -> bool:
                return all(meta.get(wk) == wv for wk, wv in where.items())
            if filter_fn is not None:
                orig = filter_fn
                filter_fn = lambda m: orig(m) and _where_filter(m)
            else:
                filter_fn = _where_filter

        ep = [self._entry_point]
        for lc in range(self._max_level, 0, -1):
            nearest = self._search_layer(query, ep, ef=1, level=lc)
            ep = [nearest[0][1]]

        candidates = self._search_layer(query, ep, ef=ef, level=0)

        results: List[Tuple[str, float, Dict[str, Any]]] = []
        for dist, nid in candidates:
            if nid not in self._nodes:
                continue
            node = self._nodes[nid]
            if filter_fn and not filter_fn(node.metadata):
                continue
            results.append((nid, dist, dict(node.metadata)))
            if len(results) >= k:
                break

        return results

    def delete(self, id: str) -> bool:
        """Remove a vector by id. Returns True if found and removed."""
        with self._lock:
            node = self._nodes.pop(id, None)
            if node is None:
                return False

            # Remove from all neighbor lists.
            for level, neighbor_ids in node.neighbors.items():
                for n_id in neighbor_ids:
                    peer = self._nodes.get(n_id)
                    if peer and level in peer.neighbors:
                        peer.neighbors[level].discard(id)

            # If the deleted node was the entry point, pick a new one.
            if self._entry_point == id:
                if self._nodes:
                    # Pick the node with the highest level.
                    self._entry_point = max(
                        self._nodes.values(), key=lambda n: n.level
                    ).id
                    self._max_level = self._nodes[self._entry_point].level
                else:
                    self._entry_point = None
                    self._max_level = -1
            return True

    def __len__(self) -> int:
        return len(self._nodes)

    def __contains__(self, id: str) -> bool:
        return id in self._nodes

    def ids(self) -> List[str]:
        return list(self._nodes.keys())


# ---------------------------------------------------------------------------
# Collection — wraps an index with a name and persistence
# ---------------------------------------------------------------------------

class Collection:
    """A named collection of vectors backed by an HNSW index.

    Parameters
    ----------
    name : str
        Collection name.
    dim : int
        Vector dimensionality.
    metric : str or Metric
        Distance metric: "cosine", "euclidean", or "dot".
    M : int
        HNSW M parameter.
    ef_construction : int
        HNSW construction beam width.
    ef_search : int
        HNSW search beam width.
    """

    def __init__(
        self,
        name: str,
        dim: int,
        metric: Union[str, Metric] = Metric.COSINE,
        M: int = 16,
        ef_construction: int = 200,
        ef_search: int = 50,
    ) -> None:
        self.name = name
        self.dim = dim
        if isinstance(metric, str):
            metric = Metric(metric)
        self.metric = metric
        self.index = HNSWIndex(
            dim=dim,
            metric=metric,
            M=M,
            ef_construction=ef_construction,
            ef_search=ef_search,
        )

    # -- convenience wrappers ---------------------------------------------

    def add(
        self,
        id: str,
        vector: Union[List[float], np.ndarray],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Add a single vector."""
        self.index.insert(id, np.asarray(vector, dtype=np.float32), metadata)

    def add_batch(
        self,
        ids: List[str],
        vectors: Union[List[List[float]], np.ndarray],
        metadatas: Optional[List[Optional[Dict[str, Any]]]] = None,
    ) -> None:
        """Add multiple vectors in one call."""
        vectors_arr = np.asarray(vectors, dtype=np.float32)
        if metadatas is None:
            metadatas = [None] * len(ids)
        if len(ids) != len(vectors_arr) or len(ids) != len(metadatas):
            raise ValueError("ids, vectors, and metadatas must have equal length")
        for vid, vec, meta in zip(ids, vectors_arr, metadatas):
            self.index.insert(vid, vec, meta)

    def query(
        self,
        vector: Union[List[float], np.ndarray],
        k: int = 10,
        where: Optional[Dict[str, Any]] = None,
        filter_fn: Optional[Callable[[Dict[str, Any]], bool]] = None,
    ) -> List[Dict[str, Any]]:
        """Query the collection. Returns list of dicts with id, distance,
        and metadata."""
        results = self.index.search(
            np.asarray(vector, dtype=np.float32),
            k=k,
            where=where,
            filter_fn=filter_fn,
        )
        return [
            {"id": rid, "distance": round(dist, 6), "metadata": meta}
            for rid, dist, meta in results
        ]

    def delete(self, id: str) -> bool:
        return self.index.delete(id)

    def count(self) -> int:
        return len(self.index)

    def ids(self) -> List[str]:
        return self.index.ids()

    # -- persistence ------------------------------------------------------

    def save(self, path: Union[str, Path]) -> None:
        """Save the collection to a directory.

        Creates two files:
          <path>/meta.json   — collection config
          <path>/data.bin    — binary vector data + graph structure
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        meta = {
            "name": self.name,
            "dim": self.dim,
            "metric": self.metric.value,
            "M": self.index.M,
            "ef_construction": self.index.ef_construction,
            "ef_search": self.index.ef_search,
            "entry_point": self.index._entry_point,
            "max_level": self.index._max_level,
            "count": len(self.index),
        }
        with open(path / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)

        with open(path / "data.bin", "wb") as f:
            for node in self.index._nodes.values():
                # id length (4 bytes) + id bytes
                id_bytes = node.id.encode("utf-8")
                f.write(struct.pack("<I", len(id_bytes)))
                f.write(id_bytes)

                # vector (dim * 4 bytes)
                f.write(node.vector.tobytes())

                # metadata as JSON
                meta_bytes = json.dumps(node.metadata).encode("utf-8")
                f.write(struct.pack("<I", len(meta_bytes)))
                f.write(meta_bytes)

                # level
                f.write(struct.pack("<I", node.level))

                # neighbors: num_levels, then for each level: level_id, count, ids
                levels = node.neighbors
                f.write(struct.pack("<I", len(levels)))
                for level_id, neighbor_ids in levels.items():
                    f.write(struct.pack("<I", level_id))
                    ids_list = list(neighbor_ids)
                    f.write(struct.pack("<I", len(ids_list)))
                    for nid in ids_list:
                        nid_bytes = nid.encode("utf-8")
                        f.write(struct.pack("<I", len(nid_bytes)))
                        f.write(nid_bytes)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "Collection":
        """Load a collection from a directory previously written by save()."""
        path = Path(path)

        with open(path / "meta.json", "r") as f:
            meta = json.load(f)

        col = cls(
            name=meta["name"],
            dim=meta["dim"],
            metric=meta["metric"],
            M=meta["M"],
            ef_construction=meta["ef_construction"],
            ef_search=meta["ef_search"],
        )
        col.index._entry_point = meta["entry_point"]
        col.index._max_level = meta["max_level"]

        with open(path / "data.bin", "rb") as f:
            for _ in range(meta["count"]):
                # id
                (id_len,) = struct.unpack("<I", f.read(4))
                node_id = f.read(id_len).decode("utf-8")

                # vector
                vec = np.frombuffer(f.read(meta["dim"] * 4), dtype=np.float32).copy()

                # metadata
                (meta_len,) = struct.unpack("<I", f.read(4))
                node_meta = json.loads(f.read(meta_len).decode("utf-8"))

                # level
                (level,) = struct.unpack("<I", f.read(4))

                # neighbors
                (num_levels,) = struct.unpack("<I", f.read(4))
                neighbors: Dict[int, Set[str]] = {}
                for _ in range(num_levels):
                    (level_id,) = struct.unpack("<I", f.read(4))
                    (count,) = struct.unpack("<I", f.read(4))
                    nids: Set[str] = set()
                    for _ in range(count):
                        (nid_len,) = struct.unpack("<I", f.read(4))
                        nids.add(f.read(nid_len).decode("utf-8"))
                    neighbors[level_id] = nids

                node = _Node(
                    id=node_id,
                    vector=vec,
                    metadata=node_meta,
                    neighbors=neighbors,
                    level=level,
                )
                col.index._nodes[node_id] = node

        return col


# ---------------------------------------------------------------------------
# VecStore — top-level store managing multiple collections
# ---------------------------------------------------------------------------

class VecStore:
    """In-process vector database.

    Usage
    -----
    >>> store = VecStore()
    >>> col = store.create_collection("docs", dim=384)
    >>> col.add("doc1", np.random.randn(384).astype("float32"), {"title": "hello"})
    >>> results = col.query(np.random.randn(384).astype("float32"), k=5)
    """

    def __init__(self, persist_dir: Optional[Union[str, Path]] = None) -> None:
        self._collections: Dict[str, Collection] = {}
        self.persist_dir = Path(persist_dir) if persist_dir else None

    def create_collection(
        self,
        name: str,
        dim: int,
        metric: Union[str, Metric] = Metric.COSINE,
        M: int = 16,
        ef_construction: int = 200,
        ef_search: int = 50,
    ) -> Collection:
        if name in self._collections:
            raise KeyError(f"Collection '{name}' already exists")
        col = Collection(
            name=name,
            dim=dim,
            metric=metric,
            M=M,
            ef_construction=ef_construction,
            ef_search=ef_search,
        )
        self._collections[name] = col
        return col

    def get_collection(self, name: str) -> Collection:
        if name not in self._collections:
            raise KeyError(f"Collection '{name}' not found")
        return self._collections[name]

    def delete_collection(self, name: str) -> bool:
        return self._collections.pop(name, None) is not None

    def list_collections(self) -> List[str]:
        return list(self._collections.keys())

    def save(self) -> None:
        """Persist all collections to disk."""
        if self.persist_dir is None:
            raise ValueError("No persist_dir configured")
        if not isinstance(self.persist_dir, Path):
            self.persist_dir = Path(self.persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        for name, col in self._collections.items():
            col.save(self.persist_dir / name)

        # Write a manifest.
        manifest = {
            "collections": [
                {"name": c.name, "dim": c.dim, "metric": c.metric.value}
                for c in self._collections.values()
            ]
        }
        with open(self.persist_dir / "manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)

    @classmethod
    def load(cls, persist_dir: Union[str, Path]) -> "VecStore":
        """Load all collections from a persisted directory."""
        persist_dir = Path(persist_dir)
        with open(persist_dir / "manifest.json", "r") as f:
            manifest = json.load(f)

        store = cls(persist_dir=persist_dir)
        for entry in manifest["collections"]:
            col = Collection.load(persist_dir / entry["name"])
            store._collections[col.name] = col
        return store
