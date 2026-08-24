from __future__ import annotations

import argparse
import json
import logging
import pickle
from pathlib import Path
from typing import Sequence

import lz4.frame
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder

from . import eval as rec_eval
from . import init_hierarchy as init_h
from . import utils
from .logistic_head import _load_split_embeddings


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_knn_head(
    train_samples: Sequence[dict],
    n_neighbors: int = 5,
    weights: str = "distance",
    metric: str = "cosine",
    n_jobs: int = 1,
    jl_transform=None,
):
    logging.info("Loading embeddings for %d training samples...", len(train_samples))
    X_train, y_train = _load_split_embeddings(train_samples, jl_transform=jl_transform)
    n_classes = len(np.unique(y_train))
    logging.info(
        "Training KNeighborsClassifier: X=%s, classes=%d, k=%d, weights=%s, metric=%s",
        X_train.shape,
        n_classes,
        n_neighbors,
        weights,
        metric,
    )

    le = LabelEncoder()
    y_enc = le.fit_transform(y_train)

    clf = KNeighborsClassifier(
        n_neighbors=n_neighbors,
        weights=weights,
        metric=metric,
        n_jobs=n_jobs,
    )

    import time
    t0 = time.time()
    clf.fit(X_train, y_enc)
    elapsed = time.time() - t0

    logging.info("Training complete: elapsed=%.1fs", elapsed)
    return clf, le


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_knn_head(clf: KNeighborsClassifier, le: LabelEncoder, test_samples: Sequence[dict], gt_tree=None, jl_transform=None):
    logging.info("Loading embeddings for %d test samples...", len(test_samples))
    X_test, y_test = _load_split_embeddings(test_samples, jl_transform=jl_transform)

    if X_test.shape[0] == 0:
        logging.warning("No test samples to evaluate")
        return rec_eval.compute_eval_metrics([], [], tree=None, gt_tree=gt_tree)

    y_pred_enc = clf.predict(X_test)
    y_pred = le.inverse_transform(y_pred_enc)

    logging.info("Evaluation complete")
    return rec_eval.compute_eval_metrics(y_test.tolist(), y_pred.tolist(), tree=None, gt_tree=gt_tree)


# ---------------------------------------------------------------------------
# Model persistence
# ---------------------------------------------------------------------------

def _save_model(clf: KNeighborsClassifier, le: LabelEncoder, path_like):
    out_path = Path(path_like)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"clf": clf, "le": le}
    with lz4.frame.open(str(out_path), mode="wb",
                        compression_level=lz4.frame.COMPRESSIONLEVEL_MINHC) as f:
        pickle.dump(payload, f)
    logging.info("Model saved to %s", out_path)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(cmdline):
    samples = init_h.build_samples_from_descriptor(cmdline.descriptor)
    logging.info("Loaded %d total samples from descriptor", len(samples))

    train_samples, test_samples = init_h.split_fixed_samples(
        samples,
        test_size=cmdline.test_size,
        train_size=cmdline.train_size,
        seed=cmdline.seed,
    )
    logging.info(
        "Split: %d train, %d test (seed=%d)",
        len(train_samples),
        len(test_samples),
        cmdline.seed,
    )

    if len(train_samples) == 0:
        raise ValueError("No train samples available after split")

    jl_transform = init_h.build_jl_transform(samples, cmdline.jl_dim, cmdline.jl_seed)

    clf, le = train_knn_head(
        train_samples,
        n_neighbors=cmdline.n_neighbors,
        weights=cmdline.weights,
        metric=cmdline.metric,
        n_jobs=cmdline.n_jobs,
        jl_transform=jl_transform,
    )

    summary = {
        "train_samples": len(train_samples),
        "test_samples": len(test_samples),
        "classes": int(len(le.classes_)),
        "n_neighbors": int(cmdline.n_neighbors),
        "weights": cmdline.weights,
        "metric": cmdline.metric,
        "seed": int(cmdline.seed),
        "jl_dim": cmdline.jl_dim,
        "jl_seed": int(cmdline.jl_seed),
    }

    gt_tree = None
    try:
        _desc = utils.load_descriptor(cmdline.descriptor)
        _hier = utils.hierarchy_from_descriptor(_desc)
        gt_tree = utils.tree_from_list(_hier)
        logging.info("Loaded GT hierarchy for geodesic distance (%d nodes)", len(gt_tree.nodes))
    except Exception as exc:
        logging.warning("Could not load GT hierarchy for geodesic distance: %s", exc)

    if cmdline.eval_test:
        metrics = evaluate_knn_head(clf, le, test_samples, gt_tree=gt_tree, jl_transform=jl_transform)
        summary["test_metrics"] = metrics
        logging.info("Test metrics: %s", metrics)

    if cmdline.output is not None:
        out_path = Path(cmdline.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w") as ofile:
            json.dump(summary, ofile, indent=1)
        logging.info("Summary saved to %s", out_path)

    if cmdline.model_output is not None:
        _save_model(clf, le, cmdline.model_output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a KNN classification head on pre-computed DINO embeddings."
    )
    parser.add_argument("--descriptor", required=True, type=str,
                        help="descriptor JSON path (must point to pre-embedded paths)")
    parser.add_argument("--output", default=None, type=str,
                        help="optional output JSON summary path")
    parser.add_argument("--model-output", default=None, type=str,
                        help="optional serialized model output (.lz4)")
    parser.add_argument("--test-size", default=0.15, type=float,
                        help="number of test samples (>=1) or fraction of total (0<v<1, e.g. 0.15 for 15%%)")
    parser.add_argument("--train-size", default=None, type=int,
                        help="fixed number of train samples after test split (default: all remaining)")
    parser.add_argument("--seed", default=0, type=int,
                        help="seed for deterministic train/test split")
    parser.add_argument("--n-neighbors", default=5, type=int,
                        help="number of neighbors for KNeighborsClassifier (default: 5)")
    parser.add_argument("--weights", default="distance", type=str, choices=["uniform", "distance"],
                        help="weight function used in prediction (default: distance)")
    parser.add_argument("--metric", default="cosine", type=str,
                        help="distance metric for KNeighborsClassifier (default: cosine)")
    parser.add_argument("--n-jobs", default=1, type=int,
                        help="number of parallel jobs for neighbor search")
    parser.add_argument("--jl-dim", default=None, type=int,
                        help="if set, apply a Gaussian Johnson–Lindenstrauss projection to reduce "
                             "embeddings to this many dimensions before training/eval")
    parser.add_argument("--jl-seed", default=0, type=int,
                        help="random seed for the JL projection matrix (default: 0)")
    parser.add_argument("--eval-test", action="store_true",
                        help="evaluate on test split and report metrics")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="enable debug logging")
    parser.add_argument("-q", "--quite", action="store_true",
                        help="disable warnings")
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    elif args.quite:
        logging.basicConfig(level=logging.ERROR)
    else:
        logging.basicConfig(level=logging.INFO)

    main(args)
