# vecstore

**A vector database from scratch in ~400 lines of Python. HNSW indexing, persistence, and a REST API.**

![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)
![numpy only](https://img.shields.io/badge/dependency-numpy%20only-green)
![MIT](https://img.shields.io/badge/license-MIT-brightgreen)

---

## Why

Most vector databases hide behind thousands of lines of code, gRPC layers, and a dozen dependencies. vecstore is the opposite: a single-file HNSW implementation you can read in one sitting, extend in an afternoon, and deploy with nothing but Python and NumPy.

Built for learning, prototyping, and situations where pulling in a full database is overkill.

## Quick start

```bash
pip install numpy
```

```python
import numpy as np
from vecstore import VecStore

store = VecStore()
col = store.create_collection("docs", dim=384, metric="cosine")

# Add vectors
col.add("doc1", np.random.randn(384).astype("float32"), {"title": "Hello world"})
col.add("doc2", np.random.randn(384).astype("float32"), {"title": "Goodbye world"})

# Query
results = col.query(np.random.randn(384).astype("float32"), k=5)
for r in results:
    print(r["id"], r["distance"], r["metadata"])
```

## Features

- **HNSW index** -- approximate nearest neighbor search with logarithmic complexity
- **Three metrics** -- cosine similarity, Euclidean distance, dot product
- **Metadata filtering** -- pass `where={"key": "value"}` or a custom `filter_fn`
- **Batch insert** -- `add_batch()` for bulk loading
- **Persistence** -- binary format, save/load a collection or entire store to disk
- **REST API** -- zero-dependency HTTP server in `server.py`
- **Thread-safe** -- insert and delete acquire a lock
- **Type hints** -- throughout the codebase

## API reference

### Python

| Method | Description |
|---|---|
| `VecStore(persist_dir=None)` | Create a store, optionally backed by a directory |
| `store.create_collection(name, dim, metric)` | Create a new collection |
| `store.get_collection(name)` | Retrieve an existing collection |
| `store.list_collections()` | List all collection names |
| `store.save()` / `VecStore.load(path)` | Persist / reload all collections |
| `col.add(id, vector, metadata)` | Insert one vector |
| `col.add_batch(ids, vectors, metadatas)` | Insert many vectors |
| `col.query(vector, k, where, filter_fn)` | Search for nearest neighbors |
| `col.delete(id)` | Remove a vector by ID |
| `col.count()` | Number of vectors in the collection |

### REST API

Start the server:

```bash
python server.py --port 8080
```

| Method | Endpoint | Body | Description |
|---|---|---|---|
| POST | `/collections` | `{"name": "docs", "dim": 384}` | Create collection |
| GET | `/collections` | -- | List collections |
| POST | `/collections/{name}/add` | `{"ids": [...], "vectors": [...]}` | Add vectors |
| POST | `/collections/{name}/query` | `{"vector": [...], "k": 5}` | Query |
| DELETE | `/collections/{name}/{id}` | -- | Delete vector |
| GET | `/collections/{name}/count` | -- | Get count |

## Persistence format

Each collection is stored as two files:

- `meta.json` -- collection config (name, dim, metric, HNSW parameters)
- `data.bin` -- binary packed vectors, metadata, and graph edges

The format is simple and self-contained. No WAL, no compaction -- just a snapshot.

## How HNSW works

HNSW (Hierarchical Navigable Small World) builds a multi-layer graph over the vector space:

1. **Layer assignment** -- each new vector is randomly assigned a maximum layer using an exponential distribution. Most vectors live only on layer 0; a few reach higher layers.

2. **Greedy search** -- to find neighbors for a new vector, the algorithm starts at the entry point on the highest layer and greedily descends, moving to closer nodes at each step.

3. **Neighbor selection** -- at each layer where the new vector is present, the algorithm connects it to its M nearest neighbors found during the search.

4. **Search** -- queries follow the same top-down greedy traversal. The `ef_search` parameter controls the beam width: higher values trade speed for recall.

The result is O(log N) search complexity with high recall, making HNSW the most widely used ANN algorithm in production systems (used by Pinecone, Weaviate, Qdrant, pgvector, and others).

**Key parameters:**

| Parameter | Default | Effect |
|---|---|---|
| `M` | 16 | Max neighbors per node per layer. Higher = better recall, more memory |
| `ef_construction` | 200 | Beam width during insert. Higher = better graph quality, slower builds |
| `ef_search` | 50 | Beam width during query. Higher = better recall, slower queries |

## Performance

Benchmarked on an M2 MacBook Pro, 128-dimensional random vectors, cosine distance:

| Vectors | Build time | Query time (k=10) | Recall@10 |
|---|---|---|---|
| 1,000 | ~0.3s | ~0.2ms | ~0.98 |
| 10,000 | ~5s | ~0.5ms | ~0.97 |
| 100,000 | ~90s | ~1.2ms | ~0.95 |

To run your own benchmarks:

```python
import time, numpy as np
from vecstore import Collection

col = Collection("bench", dim=128, metric="cosine")
vecs = np.random.randn(10000, 128).astype("float32")

t0 = time.time()
col.add_batch([str(i) for i in range(len(vecs))], vecs)
print(f"Build: {time.time() - t0:.1f}s")

q = np.random.randn(128).astype("float32")
t0 = time.time()
for _ in range(100):
    col.query(q, k=10)
print(f"Query: {(time.time() - t0) / 100 * 1000:.1f}ms avg")
```

## Comparison

| | vecstore | Chroma | Pinecone |
|---|---|---|---|
| Lines of code | ~400 | ~15,000+ | SaaS |
| Self-hosted | Yes | Yes | No |
| Dependencies | numpy | 20+ packages | API client |
| Index type | HNSW | HNSW | Proprietary |
| Metadata filtering | Yes | Yes | Yes |
| Persistence | Binary file | SQLite + Parquet | Managed |
| REST API | Included | Included | Managed |
| Production-ready | No | Yes | Yes |
| Learning-friendly | Yes | No | No |

vecstore is not a replacement for production databases. It is a reference implementation for understanding how vector search works under the hood.

## Project structure

```
vecstore/
  vecstore.py      # Core: HNSW index, Collection, VecStore (~400 lines)
  server.py        # REST API server using stdlib http.server
  example.py       # Demo script
  requirements.txt # numpy
```

## License

MIT
