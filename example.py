"""
vecstore example — create, insert, query, persist, reload.

Run:
    python example.py
"""

from __future__ import annotations

import shutil
import tempfile

import numpy as np

from vecstore import VecStore


def main() -> None:
    dim = 128
    n_vectors = 500

    # --- 1. Create a store and a collection --------------------------------
    store = VecStore()
    col = store.create_collection("documents", dim=dim, metric="cosine")
    print(f"Created collection '{col.name}' (dim={dim}, metric=cosine)")

    # --- 2. Add vectors with metadata --------------------------------------
    rng = np.random.default_rng(42)
    ids = [f"doc-{i}" for i in range(n_vectors)]
    vectors = rng.standard_normal((n_vectors, dim)).astype(np.float32)
    categories = ["science", "history", "tech", "art", "sports"]
    metadatas = [
        {"category": categories[i % len(categories)], "index": i}
        for i in range(n_vectors)
    ]
    col.add_batch(ids, vectors, metadatas)
    print(f"Inserted {col.count()} vectors")

    # --- 3. Query -----------------------------------------------------------
    query = rng.standard_normal(dim).astype(np.float32)
    results = col.query(query, k=5)
    print("\nTop 5 nearest neighbors:")
    for r in results:
        print(f"  {r['id']:>10}  dist={r['distance']:.4f}  {r['metadata']}")

    # --- 4. Filtered query --------------------------------------------------
    results_filtered = col.query(query, k=3, where={"category": "tech"})
    print("\nTop 3 in category 'tech':")
    for r in results_filtered:
        print(f"  {r['id']:>10}  dist={r['distance']:.4f}  {r['metadata']}")

    # --- 5. Persistence: save and reload -----------------------------------
    tmpdir = tempfile.mkdtemp(prefix="vecstore_")
    store.persist_dir = tmpdir
    store.save()
    print(f"\nSaved to {tmpdir}")

    store2 = VecStore.load(tmpdir)
    col2 = store2.get_collection("documents")
    print(f"Reloaded collection '{col2.name}' with {col2.count()} vectors")

    # Verify the same query returns the same results.
    results2 = col2.query(query, k=5)
    print("\nTop 5 after reload (should match):")
    for r in results2:
        print(f"  {r['id']:>10}  dist={r['distance']:.4f}  {r['metadata']}")

    # Clean up.
    shutil.rmtree(tmpdir)
    print(f"\nCleaned up {tmpdir}")


if __name__ == "__main__":
    main()
