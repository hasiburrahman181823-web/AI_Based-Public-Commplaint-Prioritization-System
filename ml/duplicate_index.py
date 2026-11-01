"""Cached candidate retrieval and score-based duplicate groups.

The NumPy backend stores one float32 embedding per complaint and computes one
matrix-vector product per query. It never constructs a corpus-by-corpus matrix.
For learned rerankers, top-k retrieval happens *before* pair scoring: candidates
outside that retrieval set cannot be returned. This is an approximate candidate
pipeline, not proof of exhaustive duplicate recall at a given corpus size.
"""

from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import sqlite3
import threading
from collections.abc import Iterable, Sequence

import joblib
import numpy as np
from sklearn.preprocessing import normalize

REPO_ROOT = Path(__file__).resolve().parents[1]


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _normal_vectors(values, rows: int, dimension: int | None = None) -> np.ndarray:
    vectors = np.asarray(values, dtype=np.float32)
    if vectors.ndim != 2 or vectors.shape[0] != rows or vectors.shape[1] < 1:
        raise ValueError("Encoder must return a nonempty-dimension matrix aligned with input texts")
    if dimension is not None and vectors.shape[1] != dimension:
        raise ValueError("Embedding dimension changed within one cache namespace; use a new encoder fingerprint/cache")
    if not np.isfinite(vectors).all():
        raise ValueError("Encoder returned non-finite embedding vectors")
    lengths = np.linalg.norm(vectors.astype(np.float64), axis=1, keepdims=True)
    if not np.isfinite(lengths).all() or (lengths <= 0).any():
        raise ValueError("Encoder returned a zero-length embedding")
    return np.ascontiguousarray(vectors / lengths, dtype=np.float32)


class EmbeddingCache:
    """SQLite float32 cache namespaced by encoder and preprocessing identity.

    ``path`` is the database file, not a directory. Callers must use a namespace
    derived from the exact encoder weights/revision and normalization resources.
    Text keys use SHA-256; equal cleaned texts share one cached embedding. Each
    process uses independent short-lived SQLite connections, with transactional
    writes and a fixed dimension per namespace.
    """

    def __init__(self, path: str | Path, namespace: str):
        if not isinstance(namespace, str) or not namespace.strip():
            raise ValueError("Embedding cache namespace must identify the encoder and preprocessing resources")
        self.path = Path(path).expanduser().resolve()
        self.namespace = namespace
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.executescript("""
                CREATE TABLE IF NOT EXISTS namespaces (
                    namespace TEXT PRIMARY KEY,
                    dimension INTEGER NOT NULL CHECK(dimension > 0)
                );
                CREATE TABLE IF NOT EXISTS embeddings (
                    namespace TEXT NOT NULL,
                    text_key TEXT NOT NULL,
                    dimension INTEGER NOT NULL,
                    vector BLOB NOT NULL,
                    PRIMARY KEY(namespace, text_key),
                    FOREIGN KEY(namespace) REFERENCES namespaces(namespace)
                );
            """)

    @contextmanager
    def _connect(self):
        connection = sqlite3.connect(self.path, timeout=30)
        connection.execute("PRAGMA foreign_keys = ON")
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def get_or_encode(self, texts: Sequence[str], encoder, batch_size: int = 64) -> np.ndarray:
        if isinstance(texts, (str, bytes)):
            raise ValueError("texts must be a sequence, not one string")
        texts = list(texts)
        if any(not isinstance(text, str) or not text.strip() for text in texts):
            raise ValueError("Cache texts must be nonempty strings")
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        keys = [hashlib.sha256(text.encode("utf-8")).hexdigest() for text in texts]
        unique = dict(zip(keys, texts))
        found: dict[str, np.ndarray] = {}
        dimension = None
        with self._connect() as connection:
            metadata = connection.execute("SELECT dimension FROM namespaces WHERE namespace=?", (self.namespace,)).fetchone()
            if metadata:
                dimension = int(metadata[0])
            ordered_keys = list(unique)
            for start in range(0, len(ordered_keys), 800):
                chunk = ordered_keys[start:start + 800]
                placeholders = ",".join("?" for _ in chunk)
                rows = connection.execute(
                    f"SELECT text_key, dimension, vector FROM embeddings WHERE namespace=? AND text_key IN ({placeholders})",
                    [self.namespace, *chunk],
                ).fetchall()
                for key, stored_dimension, blob in rows:
                    if dimension is None or stored_dimension != dimension or len(blob) != 4 * dimension:
                        raise ValueError("Corrupt/inconsistent embedding cache; move this cache aside and rebuild it")
                    vector = np.frombuffer(blob, dtype="<f4").copy()
                    if not np.isfinite(vector).all() or not np.isclose(np.linalg.norm(vector), 1, atol=1e-3):
                        raise ValueError("Corrupt embedding cache vector; move this cache aside and rebuild it")
                    found[key] = vector
        if not texts:
            return np.empty((0, dimension or 0), dtype=np.float32)
        missing = [key for key in unique if key not in found]
        for start in range(0, len(missing), batch_size):
            chunk = missing[start:start + batch_size]
            vectors = _normal_vectors(encoder.encode([unique[key] for key in chunk]), len(chunk), dimension)
            dimension = vectors.shape[1]
            with self._connect() as connection:
                connection.execute("BEGIN IMMEDIATE")
                existing = connection.execute("SELECT dimension FROM namespaces WHERE namespace=?", (self.namespace,)).fetchone()
                if existing is not None and int(existing[0]) != dimension:
                    raise ValueError("Concurrent encoder dimension conflict in embedding cache; use distinct namespaces")
                connection.execute("INSERT OR IGNORE INTO namespaces VALUES (?, ?)", (self.namespace, dimension))
                connection.executemany(
                    "INSERT OR IGNORE INTO embeddings VALUES (?, ?, ?, ?)",
                    [(self.namespace, key, dimension, sqlite3.Binary(vector.astype("<f4", copy=False).tobytes()))
                     for key, vector in zip(chunk, vectors)],
                )
            found.update(zip(chunk, vectors))
        return np.stack([found[key] for key in keys]).astype(np.float32, copy=False)


def _top_k(scores: np.ndarray, identifiers: list[str], k: int) -> np.ndarray:
    """Top-k with deterministic lexical-ID tie breaking, including cutoff ties."""
    if len(scores) <= k:
        candidates = np.arange(len(scores))
    else:
        provisional = np.argpartition(-scores, k - 1)[:k]
        cutoff = scores[provisional].min()
        candidates = np.flatnonzero(scores >= cutoff)
    ordered = sorted(candidates.tolist(), key=lambda index: (-float(scores[index]), identifiers[index]))
    return np.asarray(ordered[:k], dtype=int)


class DuplicateFinder:
    """Load a trusted duplicate model and reuse an incremental candidate index.

    Supply complaints as dictionaries with unique ``id``, ``text`` and ``lang``;
    optional category/location/timestamp values feed the saved pair model.
    ``original_id`` is never used as a retrieval or scoring feature. Records are
    copied during indexing. Updating IDs, text or relevant metadata rebuilds the
    in-memory index; unchanged cleaned text is recovered from the persistent
    embedding cache without encoding again.
    """

    def __init__(self, model_path: str | Path, cache_dir: str | Path, encoder=None, top_k: int = 100):
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
            raise ValueError("top_k must be a positive integer")
        self.model_path = Path(model_path).expanduser().resolve()
        self.cache_dir = Path(cache_dir).expanduser().resolve()
        self.top_k = top_k
        self.bundle = joblib.load(self.model_path)  # Load only trusted, self-created artifacts.
        if (not isinstance(self.bundle, dict) or self.bundle.get("kind") != "banglapsg_duplicate_model"
                or self.bundle.get("schema_version") != 1):
            raise ValueError("Not a supported duplicate-detection model bundle")
        self.model_name = self.bundle.get("model_name")
        if self.model_name not in ("B1", "B2", "C0", "P1", "P2"):
            raise ValueError("Unsupported duplicate model name")
        self.threshold = float(self.bundle["threshold"])
        if not math.isfinite(self.threshold):
            raise ValueError("Saved duplicate threshold must be finite")
        self.artifact_root = (self.model_path.parent / self.bundle.get("artifact_root_relative", ".")).resolve()
        resources = self.bundle.get("resource_sha256", {})
        if not isinstance(resources, dict):
            raise ValueError("resource_sha256 must map relative resource paths to hashes")
        for relative, expected in resources.items():
            path = self._resource(relative)
            if _file_hash(path) != expected:
                raise ValueError(f"Saved duplicate resource fingerprint changed: {relative}")
        self.variants_path = self._resource(self.bundle["variants_file"])
        from ml.duplicate_features import preprocess_complaint
        self._preprocess = preprocess_complaint
        self._feature_matrix = None
        self.gazetteer = None
        self.feature_names = list(self.bundle.get("feature_names", []))
        if self.model_name in ("C0", "P1", "P2"):
            if not self.feature_names or len(set(self.feature_names)) != len(self.feature_names):
                raise ValueError("Learned duplicate models need unique saved feature_names")
            if self.bundle.get("estimator") is None:
                raise ValueError("Learned duplicate model has no fitted estimator")
            from ml.duplicate_features import feature_matrix
            from ml.location import LocationGazetteer
            self._feature_matrix = feature_matrix
            config = self.bundle.get("feature_config", {})
            self.gazetteer = LocationGazetteer(self._resource(self.bundle["gazetteer_file"]),
                                              fuzzy_threshold=config.get("fuzzy_threshold", 90))
        self.encoder = encoder
        self.cache = None
        if self.model_name == "B1":
            if self.bundle.get("vectorizer") is None:
                raise ValueError("B1 needs the saved training-only TF-IDF vectorizer")
        else:
            fingerprint = self.bundle.get("encoder_fingerprint")
            if not isinstance(fingerprint, str) or not fingerprint:
                raise ValueError("Embedding bundles must identify their encoder fingerprint")
            namespace = hashlib.sha256(json.dumps(
                {"encoder": fingerprint, "resources": resources}, sort_keys=True,
            ).encode("utf-8")).hexdigest()
            self.cache = EmbeddingCache(self.cache_dir / "embeddings.sqlite3", namespace)
            if self.encoder is None:
                from ml_training.modeling import SentenceEncoder
                self.encoder = SentenceEncoder(str(self._resource(self.bundle["encoder_dir"])), allow_download=False)
        self._signature = None
        self._records: list[dict] = []
        self._matrix = None
        self._built = False
        self._lock = threading.RLock()
        self.last_query_stats: dict = {}

    def _resource(self, relative: str) -> Path:
        if not isinstance(relative, str) or not relative:
            raise ValueError("Resource paths must be nonempty relative strings")
        path = (self.artifact_root / relative).resolve()
        if path != self.artifact_root and self.artifact_root not in path.parents:
            raise ValueError("Saved resource path escapes the artifact directory")
        return path

    @staticmethod
    def _validate_records(records) -> list[dict]:
        if isinstance(records, (str, bytes, dict)):
            raise ValueError("existing_complaints must be a sequence of complaint dictionaries")
        result = []
        identifiers = set()
        for record in records:
            if not isinstance(record, dict):
                raise ValueError("Each complaint must be a dictionary")
            for name in ("id", "text", "lang"):
                if not isinstance(record.get(name), str) or not record[name].strip():
                    raise ValueError(f"Complaint field {name!r} must be a nonempty string")
            identifier = record["id"].strip()
            if identifier in identifiers:
                raise ValueError(f"Duplicate complaint id in retrieval corpus: {identifier!r}")
            identifiers.add(identifier)
            result.append(dict(record, id=identifier))
        return result

    def build_index(self, existing_complaints: Sequence[dict]):
        """Initialize/update the corpus once, or reuse it via find_duplicates(..., None)."""
        with self._lock:
            raw = self._validate_records(existing_complaints)
            # Metadata changes must invalidate the stored feature context even
            # when the text/embedding is unchanged. Original IDs are excluded.
            signature = hashlib.sha256(json.dumps(
                [[row.get(key, "") for key in ("id", "text", "raw_text", "lang", "category", "location", "timestamp")]
                 for row in raw], ensure_ascii=False, separators=(",", ":"),
            ).encode("utf-8")).hexdigest()
            if self._built and signature == self._signature:
                return self
            records = [self._preprocess(row, variants_path=self.variants_path) for row in raw]
            if any(not row.get("text") for row in records):
                raise ValueError("Corpus complaint text became empty after preprocessing")
            if not records:
                matrix = None
            elif self.model_name == "B1":
                matrix = normalize(self.bundle["vectorizer"].transform([row["text"] for row in records]), norm="l2", copy=False)
            else:
                matrix = self.cache.get_or_encode([row["text"] for row in records], self.encoder)
            self._records, self._matrix, self._signature = records, matrix, signature
            self._built = True
            return self

    def find_duplicates(self, new_complaint: dict, existing_complaints: Sequence[dict] | None = None,
                        top_k: int | None = None) -> list[dict]:
        """Return threshold-passing top-k candidates, sorted by final pair score.

        B1 retrieves by TF-IDF cosine; all other models retrieve by MiniLM
        cosine. Learned C0/P1/P2 then rerank those same candidates. Top-k must be
        selected independently of test labels; measure retrieval recall on a
        representative labeled corpus before asserting production coverage.
        """
        k = self.top_k if top_k is None else top_k
        if isinstance(k, bool) or not isinstance(k, int) or k < 1:
            raise ValueError("top_k must be a positive integer")
        with self._lock:
            if existing_complaints is not None:
                self.build_index(existing_complaints)
            if not self._built:
                raise ValueError("Build an index or pass existing_complaints before querying")
            new = self._validate_records([new_complaint])[0]
            new = self._preprocess(new, variants_path=self.variants_path)
            if not new.get("text"):
                raise ValueError("New complaint text became empty after preprocessing")
            eligible = np.asarray([index for index, row in enumerate(self._records) if row["id"] != new["id"]], dtype=int)
            self.last_query_stats = {"corpus_size": len(self._records), "eligible_candidates": len(eligible),
                                     "retrieved_candidates": 0, "returned_duplicates": 0,
                                     "backend": "numpy_dot" if self.model_name != "B1" else "sparse_tfidf_dot",
                                     "exhaustive_candidates": len(eligible) <= k}
            if not len(eligible):
                return []
            if self.model_name == "B1":
                query = normalize(self.bundle["vectorizer"].transform([new["text"]]), norm="l2", copy=False)
                all_cosines = (self._matrix @ query.T).toarray().reshape(-1)
            else:
                query = self.cache.get_or_encode([new["text"]], self.encoder)[0]
                if self._matrix.shape[1] != len(query):
                    raise ValueError("Query and index embedding dimensions do not match")
                all_cosines = self._matrix @ query
            all_cosines = np.clip(np.asarray(all_cosines, dtype=float), -1, 1)
            chosen = _top_k(all_cosines[eligible], [self._records[index]["id"] for index in eligible], k)
            positions = eligible[chosen]
            records = [self._records[index] for index in positions]
            cosines = all_cosines[positions]
            features = None
            if self.model_name in ("B1", "B2"):
                scores = cosines
            else:
                config = self.bundle.get("feature_config", {})
                features = self._feature_matrix(
                    [new] * len(records), records, cosines, self.feature_names, self.gazetteer,
                    category_source=config.get("category_source", "provided"),
                )
                probabilities = self.bundle["estimator"].predict_proba(features)
                positive = self.bundle.get("positive_class", 1)
                classes = list(self.bundle["estimator"].classes_)
                indices = [i for i, label in enumerate(classes) if label == positive]
                if len(indices) != 1:
                    raise ValueError("Saved learned model is missing its positive class")
                scores = np.asarray(probabilities, dtype=float)[:, indices[0]]
                if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
                    raise ValueError("Learned duplicate model returned invalid probabilities")
            matches = []
            for rank, (record, score, cosine) in enumerate(zip(records, scores, cosines), 1):
                if float(score) < self.threshold:
                    continue
                result = {"id": record["id"], "score": float(score), "threshold": self.threshold,
                          "is_duplicate": True, "cosine": float(cosine),
                          "embedding_cosine": float(cosine) if self.model_name != "B1" else None,
                          "retrieval_rank": rank, "model_name": self.model_name}
                if features is not None:
                    result["features"] = dict(zip(self.feature_names, map(float, features[rank - 1])))
                matches.append(result)
            matches.sort(key=lambda item: (-item["score"], item["id"]))
            self.last_query_stats.update(retrieved_candidates=len(records), returned_duplicates=len(matches))
            return matches


_default_finder: DuplicateFinder | None = None


def configure_duplicate_finder(model_path: str | Path, cache_dir: str | Path | None = None,
                               encoder=None, top_k: int = 100) -> DuplicateFinder:
    """Configure one reusable finder at application startup."""
    global _default_finder
    _default_finder = DuplicateFinder(model_path, cache_dir or REPO_ROOT / ".cache" / "duplicate_retrieval",
                                      encoder=encoder, top_k=top_k)
    return _default_finder


def find_duplicates(new_complaint: dict, existing_complaints: Sequence[dict], *,
                    finder: DuplicateFinder | None = None, model_path: str | Path | None = None,
                    cache_dir: str | Path | None = None, top_k: int | None = None) -> list[dict]:
    """Production convenience API; configure once or set BANGLAPSG_DUPLICATE_MODEL.

    The default trusted model is ``artifacts/duplicates/P2.joblib``. The corpus
    is compared against the cached index signature on every two-argument call;
    repeated calls reuse encoded texts and the unchanged in-memory matrix.
    """
    global _default_finder
    if finder is None:
        selected = Path(model_path or os.getenv("BANGLAPSG_DUPLICATE_MODEL", str(REPO_ROOT / "artifacts" / "duplicates" / "P2.joblib"))).expanduser().resolve()
        if _default_finder is None or (model_path is not None and _default_finder.model_path != selected):
            if not selected.is_file():
                raise ValueError(f"Duplicate model not found: {selected}. Train it or call configure_duplicate_finder with your trusted model.")
            _default_finder = DuplicateFinder(selected, cache_dir or REPO_ROOT / ".cache" / "duplicate_retrieval")
        finder = _default_finder
    return finder.find_duplicates(new_complaint, existing_complaints, top_k=top_k)


def connected_components(edges: Iterable[tuple[str, str, float]], threshold: float,
                         all_ids: Iterable[str] | None = None) -> list[list[str]]:
    """Build deterministic connected components using only thresholded scores.

    Endpoints of below-threshold edges remain singleton vertices. Transitivity
    can merge A and C through B even if A-C was never scored or is below the
    threshold. These are candidate incident groups and require cluster review.
    ``original_id`` and true duplicate labels are never inspected.
    """
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not math.isfinite(threshold):
        raise ValueError("threshold must be finite")
    parents: dict[str, str] = {}

    def add(identifier: str) -> None:
        if not isinstance(identifier, str) or not identifier.strip():
            raise ValueError("Cluster complaint IDs must be nonempty strings")
        parents.setdefault(identifier, identifier)

    def root(identifier: str) -> str:
        while identifier != parents[identifier]:
            parents[identifier] = parents[parents[identifier]]
            identifier = parents[identifier]
        return identifier

    for identifier in all_ids if all_ids is not None else []:
        add(identifier)
    for left, right, score in edges:
        add(left)
        add(right)
        if isinstance(score, bool) or not isinstance(score, (int, float, np.number)) or not math.isfinite(score):
            raise ValueError("Cluster edge scores must be finite numbers")
        if score >= threshold:
            left_root, right_root = root(left), root(right)
            if left_root != right_root:
                lower, upper = sorted((left_root, right_root))
                parents[upper] = lower
    groups: dict[str, list[str]] = {}
    for identifier in sorted(parents):
        groups.setdefault(root(identifier), []).append(identifier)
    return sorted(groups.values(), key=lambda members: tuple(members))
