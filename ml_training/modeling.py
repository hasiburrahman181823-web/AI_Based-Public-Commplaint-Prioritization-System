"""Leakage-safe baseline estimators and a lazy, offline-first MiniLM encoder.

Text entering these helpers must already have received the shared preprocessing
used at inference. Vectorizers therefore do not lowercase a second time. The
word baseline uses whitespace tokens because sklearn's default regex splits
Bengali combining marks and can discard whole short Bengali words.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.exceptions import NotFittedError
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.frozen import FrozenEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.utils.validation import check_is_fitted

DEFAULT_SENTENCE_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def _linear_classifier(kind: str, seed: int):
    if kind == "svm":
        return LinearSVC(class_weight="balanced", random_state=seed, max_iter=10000)
    if kind == "lr":
        return LogisticRegression(class_weight="balanced", random_state=seed, max_iter=2000)
    raise ValueError(f"Unknown linear classifier kind: {kind!r}")


def build_text_estimator(name: str, seed: int = 42) -> Pipeline:
    """Return an unfitted char-wb (1–5) SVM or whitespace-word LR pipeline.

    Both estimators use all available features (no hidden vocabulary cap) and
    balanced class weights. Calling ``fit`` only on training text fits both the
    vocabulary/IDF and classifier exclusively on that partition.
    """
    if name == "char_svm":
        features = TfidfVectorizer(
            analyzer="char_wb", ngram_range=(1, 5), lowercase=False,
        )
        classifier = _linear_classifier("svm", seed)
    elif name == "word_lr":
        features = TfidfVectorizer(
            analyzer="word", token_pattern=r"(?u)\S+", ngram_range=(1, 1), lowercase=False,
        )
        classifier = _linear_classifier("lr", seed)
    else:
        raise ValueError(f"Unknown text model {name!r}; choose char_svm or word_lr.")
    return Pipeline([("tfidf", features), ("classifier", classifier)])


def build_embedding_estimator(name: str, seed: int = 42):
    """Return an unfitted classifier over frozen sentence embedding vectors."""
    if name == "minilm_lr":
        return _linear_classifier("lr", seed)
    if name == "minilm_svm":
        return _linear_classifier("svm", seed)
    raise ValueError(f"Unknown embedding model {name!r}; choose minilm_lr or minilm_svm.")


def calibrate_prefit(estimator, X_val, y_val) -> CalibratedClassifierCV:
    """Fit sigmoid calibration on a separate validation partition, without refit.

    The already fitted estimator is wrapped in ``FrozenEstimator`` (sklearn
    1.6+). In particular, the text vectorizer is never refitted on validation
    text. Callers must guarantee that no incident group crosses train/val.
    Validation must contain every training class and no unseen classes; silent
    calibration with a missing class is rejected instead of producing a weak or
    misleading probability model. Validation is reserved for calibration, not
    reported as held-out test performance.
    """
    try:
        check_is_fitted(estimator)
    except (NotFittedError, TypeError) as exc:
        raise ValueError("Fit the base estimator on training data before calibration.") from exc
    classes = getattr(estimator, "classes_", None)
    if classes is None or len(classes) < 2:
        raise ValueError("Calibration requires a fitted classifier with at least two classes.")
    labels = np.asarray(y_val)
    if labels.ndim != 1 or labels.size == 0:
        raise ValueError("Calibration labels must be a nonempty one-dimensional sequence.")
    rows = X_val.shape[0] if hasattr(X_val, "shape") else len(X_val)
    if rows != len(labels):
        raise ValueError("Calibration features and labels must have the same number of rows.")
    train_classes, val_classes = set(classes.tolist()), set(labels.tolist())
    missing, unseen = train_classes - val_classes, val_classes - train_classes
    if missing or unseen:
        raise ValueError(
            "Validation labels must contain exactly the training classes for calibration. "
            f"Missing: {sorted(map(str, missing))}; unseen: {sorted(map(str, unseen))}. "
            "Collect more validation incident groups or correct the upstream split."
        )
    # An explicit single fold makes this genuinely held-out calibration rather
    # than requiring sklearn's default five validation folds. The same indices
    # are supplied on each side because FrozenEstimator.fit is a no-op: base
    # learning has already happened on the disjoint training data, and all
    # validation predictions are used exactly once to fit the sigmoid only.
    indices = np.arange(len(labels))
    calibrated = CalibratedClassifierCV(
        estimator=FrozenEstimator(estimator), method="sigmoid", ensemble=False,
        cv=[(indices, indices)],
    )
    return calibrated.fit(X_val, labels)


class SentenceEncoder:
    """Load a pretrained multilingual encoder only when embeddings are requested.

    By default ``local_files_only=True`` prevents network access. Point
    ``model_name`` to an exported local encoder directory or pre-cache the model.
    Set ``allow_download=True`` explicitly for the first connected run. Encoder
    weights are saved separately with :meth:`save`; joblib artifacts should
    store the fitted lightweight classifier and relative encoder directory.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_SENTENCE_MODEL,
        device: str = "cpu",
        allow_download: bool = False,
        batch_size: int = 64,
        revision: str | None = None,
    ):
        if not isinstance(model_name, str) or not model_name.strip():
            raise ValueError("model_name must be a local encoder path or a nonempty model ID.")
        if not isinstance(device, str) or not device.strip():
            raise ValueError("device must be a nonempty device name such as cpu or cuda.")
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer.")
        if not isinstance(allow_download, bool):
            raise ValueError("allow_download must be True or False.")
        if revision is not None and (not isinstance(revision, str) or not revision.strip()):
            raise ValueError("revision must be None or a nonempty model revision.")
        self.model_name = model_name
        self.device = device
        self.allow_download = allow_download
        self.batch_size = batch_size
        self.revision = revision
        self._model = None

    def _get_model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise RuntimeError(
                    "MiniLM embeddings require the optional sentence-transformers dependency. "
                    "Install the embedding requirements or run: pip install sentence-transformers."
                ) from exc
            kwargs = {
                "device": self.device,
                "local_files_only": not self.allow_download,
                "trust_remote_code": False,
            }
            if self.revision is not None:
                kwargs["revision"] = self.revision
            try:
                self._model = SentenceTransformer(self.model_name, **kwargs)
            except Exception as exc:
                mode = (
                    "Offline loading is enabled: supply a local encoder directory or pre-cache "
                    "the model; explicitly enable downloads for an initial connected run."
                    if not self.allow_download else
                    "Model download/loading failed; check the model ID, revision and connection."
                )
                raise RuntimeError(f"Cannot load sentence encoder {self.model_name!r}. {mode}") from exc
        return self._model

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """Return finite, L2-normalized float32 vectors; no model training occurs."""
        if isinstance(texts, (str, bytes)):
            raise ValueError("encode expects a sequence of texts, not one text string.")
        texts = list(texts)
        if not texts or any(not isinstance(text, str) or not text.strip() for text in texts):
            raise ValueError("encode expects a nonempty sequence of nonempty text strings.")
        vectors = np.asarray(
            self._get_model().encode(
                texts, batch_size=self.batch_size, show_progress_bar=False,
                convert_to_numpy=True, normalize_embeddings=True,
            ),
            dtype=np.float32,
        )
        if vectors.ndim != 2 or vectors.shape[0] != len(texts) or vectors.shape[1] == 0 or not np.isfinite(vectors).all():
            raise RuntimeError("Sentence encoder returned invalid or non-finite embedding vectors.")
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        if (norms <= 0).any():
            raise RuntimeError("Sentence encoder returned zero-length embedding vectors.")
        return vectors / norms

    def save(self, path: str | Path) -> Path:
        """Export encoder configuration and weights for later local-only inference."""
        directory = Path(path)
        directory.mkdir(parents=True, exist_ok=True)
        self._get_model().save(str(directory))
        return directory
