from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import os
import platform
import re
import time
from concurrent.futures import ProcessPoolExecutor
from collections import Counter
from importlib.metadata import version
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib"))

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from lime.lime_text import LimeTextExplainer
from matplotlib.patches import Patch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    confusion_matrix,
    matthews_corrcoef,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from scipy.sparse import csr_matrix
from threadpoolctl import threadpool_limits

SEED = 42
TOP_FEATURES = 15
FIXED_COMPARISON_THRESHOLD = 0.50
CODE_TOKEN_PATTERN = re.compile(
    r"\$?[A-Za-z_][A-Za-z0-9_]*|\d+(?:\.\d+)?|"
    r"===|!==|==|!=|<=|>=|=>|->|::|&&|\|\||"
    r"[{}()\[\];,.?:+*/%<>=!&|~-]"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def code_tokenise(code: str) -> list[str]:

    return CODE_TOKEN_PATTERN.findall(str(code))


def length_features(frame: pd.DataFrame) -> np.ndarray:

    char_len = frame["code"].str.len().to_numpy(dtype=np.float64)
    token_len = frame["token_len"].to_numpy(dtype=np.float64)
    values = np.column_stack([np.log1p(char_len), np.log1p(token_len)])
    return values


def metric_row(name: str, y_true: np.ndarray, probabilities: np.ndarray, n_features: int, seconds: float) -> dict[str, Any]:
    prediction = (probabilities >= FIXED_COMPARISON_THRESHOLD).astype(int)
    tn, fp, fn, tp = (int(x) for x in confusion_matrix(y_true, prediction, labels=[0, 1]).ravel())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    return {
        "model": name,
        "comparison_threshold": FIXED_COMPARISON_THRESHOLD,
        "n_validation": int(len(y_true)),
        "n_features": int(n_features),
        "accuracy": (tp + tn) / len(y_true),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "specificity": specificity,
        "false_positive_rate": 1 - specificity,
        "mcc": float(matthews_corrcoef(y_true, prediction)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "pr_auc_average_precision": float(average_precision_score(y_true, probabilities)),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "fit_and_predict_seconds": seconds,
    }


def build_word_model(max_features: int) -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            tokenizer=code_tokenise,
            token_pattern=None,
            preprocessor=None,
            lowercase=False,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
            max_features=max_features,
            sublinear_tf=True,
            dtype=np.float32,
        )),
        ("classifier", LogisticRegression(C=1.0, max_iter=2_000, random_state=SEED)),
    ])


def build_char_model(max_features: int) -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            min_df=2,
            max_df=0.95,
            max_features=max_features,
            sublinear_tf=True,
            dtype=np.float32,
        )),
        ("classifier", LogisticRegression(C=1.0, max_iter=2_000, random_state=SEED)),
    ])


def counts_and_metrics(y_true: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict[str, Any]:
    predicted = probabilities >= threshold
    actual = y_true == 1
    tp = int(np.sum(predicted & actual))
    fp = int(np.sum(predicted & ~actual))
    fn = int(np.sum(~predicted & actual))
    tn = int(np.sum(~predicted & ~actual))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    return {
        "threshold": float(threshold),
        "predicted_vulnerable": int(predicted.sum()),
        "accuracy": (tp + tn) / len(y_true),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mcc": float(matthews_corrcoef(y_true, predicted)),
        "specificity": specificity,
        "false_positive_rate": 1 - specificity,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def threshold_sweep(y_true: np.ndarray, probabilities: np.ndarray) -> pd.DataFrame:
    """Evaluate every distinct validation score plus all-positive/all-negative ends."""

    scores = np.unique(probabilities)
    candidates = np.unique(np.concatenate(([0.0], scores, [np.nextafter(scores.max(), np.inf)])))
    rows = [counts_and_metrics(y_true, probabilities, float(threshold)) for threshold in candidates]
    return pd.DataFrame(rows).sort_values("threshold").reset_index(drop=True)


def select_row(sweep: pd.DataFrame, objective: str) -> pd.Series:
    return sweep.sort_values([objective, "threshold"], ascending=[False, False], kind="mergesort").iloc[0]


def metric_bundle(y_true: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict[str, Any]:
    predicted = probabilities >= threshold
    actual = y_true == 1
    tp = int(np.sum(predicted & actual))
    fp = int(np.sum(predicted & ~actual))
    fn = int(np.sum(~predicted & actual))
    tn = int(np.sum(~predicted & ~actual))
    precision = tp / (tp + fp) if tp + fp else 0.0  # TP / (TP + FP)
    recall = tp / (tp + fn) if tp + fn else 0.0  # TP / (TP + FN)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    return {
        "threshold": float(threshold),
        "records": int(len(y_true)),
        "accuracy": (tp + tn) / len(y_true),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mcc": (tp * tn - fp * fn) / np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) if all((tp + fp, tp + fn, tn + fp, tn + fn)) else 0.0,
        "specificity": specificity,
        "false_positive_rate": 1 - specificity,
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "pr_auc_average_precision": float(average_precision_score(y_true, probabilities)),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def configure_matplotlib() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "legend.fontsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def save_figure(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def choose_representative_cases(predictions: pd.DataFrame) -> pd.DataFrame:

    chosen: list[pd.Series] = []
    for outcome in ("TP", "FP", "FN", "TN"):
        subset = predictions.loc[predictions["outcome"] == outcome].copy()
        if subset.empty:
            continue
        median = subset["vulnerability_probability"].median()
        row = subset.assign(_distance=(subset["vulnerability_probability"] - median).abs()).sort_values(
            ["_distance", "record_id"], kind="mergesort"
        ).iloc[0].drop(labels="_distance")
        row["selection_rule"] = "probability closest to the median within the outcome group"
        chosen.append(row)
    return pd.DataFrame(chosen)


def shap_values_for_case(explainer: Any, x_case: Any) -> np.ndarray:
    values = explainer.shap_values(x_case)
    if isinstance(values, list):
        values = values[1]
    values = np.asarray(values)
    if values.ndim == 3:
        values = values[0, :, 1]
    elif values.ndim == 2:
        values = values[0]
    return values


def transform_neighborhood(vectorizer: Any, texts: list[str]) -> Any:
    if vectorizer.analyzer != "char_wb":
        return vectorizer.transform(texts)
    preprocess = vectorizer.build_preprocessor()
    analyzer = vectorizer.build_analyzer()
    vocabulary = vectorizer.vocabulary_
    words = {}
    doc_rows, word_columns, word_counts = [], [], []
    for row, text in enumerate(texts):
        counts = Counter(preprocess(text).split())
        for word, count in counts.items():
            column = words.setdefault(word, len(words))
            doc_rows.append(row)
            word_columns.append(column)
            word_counts.append(count)
    word_rows, feature_columns, feature_counts = [], [], []
    for word, row in words.items():
        counts = Counter(vocabulary[gram] for gram in analyzer(word) if gram in vocabulary)
        for column, count in counts.items():
            word_rows.append(row)
            feature_columns.append(column)
            feature_counts.append(count)
    documents = csr_matrix((word_counts, (doc_rows, word_columns)),
                           shape=(len(texts), len(words)), dtype=vectorizer.dtype)
    features = csr_matrix((feature_counts, (word_rows, feature_columns)),
                          shape=(len(words), len(vocabulary)), dtype=vectorizer.dtype)
    counts = (documents @ features).tocsr()
    counts.sort_indices()
    return vectorizer._tfidf.transform(counts, copy=False)


def lime_explanation(vectorizer: Any, classifier: Any, code: str, samples: int) -> tuple[pd.DataFrame, dict[str, float]]:
    def predict_probability(texts: list[str]) -> np.ndarray:
        return classifier.predict_proba(transform_neighborhood(vectorizer, texts))

    explainer = LimeTextExplainer(
        class_names=["safe", "vulnerable"],
        bow=True,
        split_expression=r"[^A-Za-z0-9_$]+",
        random_state=SEED,
    )
    explanation = explainer.explain_instance(
        code,
        classifier_fn=predict_probability,
        labels=(1,),
        num_features=24,
        num_samples=samples,
    )
    table = pd.DataFrame(explanation.as_list(label=1), columns=["feature", "lime_weight"])
    table["direction"] = np.where(table["lime_weight"] >= 0, "toward_vulnerable", "toward_safe")
    details = {
        "local_surrogate_r2": float(explanation.score),
        "local_surrogate_probability": float(np.asarray(explanation.local_pred).ravel()[0]),
    }
    return table, details


def global_shap_scores(x_sample: Any, background: Any, classifier: Any) -> tuple[np.ndarray, np.ndarray]:

    baseline = np.asarray(background.mean(axis=0)).ravel()
    coefficients = np.asarray(classifier.coef_).ravel()
    x_coo = x_sample.tocoo()
    absolute_sum = x_sample.shape[0] * np.abs(baseline)
    absolute_sum += np.bincount(
        x_coo.col,
        weights=np.abs(x_coo.data - baseline[x_coo.col]) - np.abs(baseline[x_coo.col]),
        minlength=x_sample.shape[1],
    )
    return np.abs(coefficients) * absolute_sum / x_sample.shape[0], baseline


def draw_confusion(metrics: pd.Series, path: Path) -> None:
    matrix = np.array([[int(metrics.tn), int(metrics.fp)], [int(metrics.fn), int(metrics.tp)]])
    fig, ax = plt.subplots(figsize=(6.4, 5.5))
    display = ConfusionMatrixDisplay(matrix, display_labels=["Safe", "Vulnerable"])
    display.plot(cmap="Blues", values_format="d", colorbar=True, ax=ax)
    ax.grid(False)
    display.im_.colorbar.set_label("Number of fragments")
    ax.set_title("Confusion matrix — final test set")
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    save_figure(path)


def draw_precision_recall(pr: pd.DataFrame, metrics: pd.Series, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.4, 5.4))
    ax.step(pr["recall"], pr["precision"], where="post", color="#1f77b4", linewidth=2, label=f"Model (PR-AUC = {metrics.pr_auc_average_precision:.3f})")
    prevalence = (int(metrics.tp) + int(metrics.fn)) / (int(metrics.tp) + int(metrics.fp) + int(metrics.fn) + int(metrics.tn))
    ax.axhline(prevalence, color="#777777", linestyle="--", linewidth=1.3, label=f"No-skill baseline ({prevalence:.3f})")
    ax.set(xlim=(0, 1), ylim=(0, 1.02), xlabel="Recall = TP / (TP + FN)", ylabel="Precision = TP / (TP + FP)", title="Precision–recall curve — final test set")
    ax.legend(loc="lower left", frameon=True)
    save_figure(path)


def draw_lime_bar(table: pd.DataFrame, row: pd.Series, path: Path) -> None:
    values = table.reindex(table["lime_weight"].abs().sort_values(ascending=True).index).tail(12)
    colours = np.where(values["lime_weight"] >= 0, "#1f77b4", "#d62728")
    fig, ax = plt.subplots(figsize=(8.6, 5.8))
    ax.barh(values["feature"], values["lime_weight"], color=colours, edgecolor="none")
    ax.axvline(0, color="black", linewidth=0.9)
    ax.set_xlabel("LIME weight in local surrogate model")
    ax.set_ylabel("Token in the code fragment")
    ax.set_title(f"LIME local explanation — {row.outcome}, record {int(row.record_id)}")
    ax.legend(handles=[Patch(color="#1f77b4", label="Supports vulnerable"), Patch(color="#d62728", label="Supports safe")], loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2, frameon=False)
    save_figure(path)


def draw_local_shap_bar(table: pd.DataFrame, row: pd.Series, path: Path) -> None:
    values = table.reindex(table["shap_value"].abs().sort_values(ascending=True).index).tail(12)
    colours = np.where(values["shap_value"] >= 0, "#e24a33", "#348abd")
    fig, ax = plt.subplots(figsize=(8.6, 5.8))
    ax.barh(values["feature"], values["shap_value"], color=colours, edgecolor="none")
    ax.axvline(0, color="black", linewidth=0.9)
    ax.set_xlabel("SHAP value (contribution to vulnerable log-odds)")
    ax.set_ylabel("Character TF-IDF n-gram")
    ax.set_title(f"SHAP local feature contributions — {row.outcome}, record {int(row.record_id)}")
    ax.legend(handles=[Patch(color="#e24a33", label="Pushes toward vulnerable"), Patch(color="#348abd", label="Pushes toward safe")], loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2, frameon=False)
    save_figure(path)


def draw_local_waterfall(values: np.ndarray, expected_value: float, x_case: Any, feature_names: np.ndarray, row: pd.Series, path: Path) -> None:
    import shap

    explanation = shap.Explanation(
        values=values,
        base_values=expected_value,
        data=x_case.toarray().ravel(),
        feature_names=feature_names,
    )
    plt.figure(figsize=(9.3, 6.2))
    shap.plots.waterfall(explanation, max_display=12, show=False)
    plt.title(f"SHAP waterfall — {row.outcome}, record {int(row.record_id)}", pad=16)
    save_figure(path)


def draw_global_shap(x_global: Any, baseline: np.ndarray, classifier: Any, feature_names: np.ndarray, ranking: np.ndarray, figure_dir: Path) -> pd.DataFrame:
    import shap

    indices = ranking[:min(TOP_FEATURES, len(ranking))]
    names = feature_names[indices]
    x_values = x_global[:, indices].toarray()
    coefficients = np.asarray(classifier.coef_).ravel()[indices]
    shap_values = (x_values - baseline[indices]) * coefficients
    global_table = pd.DataFrame({
        "rank": np.arange(1, len(indices) + 1),
        "feature": names,
        "mean_abs_shap": np.mean(np.abs(shap_values), axis=0),
        "coefficient": coefficients,
        "background_mean_tfidf": baseline[indices],
    })

    plt.figure(figsize=(8.4, 6.6))
    shap.summary_plot(shap_values, x_values, feature_names=names, max_display=TOP_FEATURES, show=False, cmap="coolwarm", rng=np.random.default_rng(SEED))
    plt.title(f"SHAP summary plot (beeswarm) — {x_global.shape[0]:,} test fragments", pad=16)
    save_figure(figure_dir / "shap_summary_beeswarm.png")

    plt.figure(figsize=(8.4, 6.6))
    shap.summary_plot(shap_values, x_values, feature_names=names, plot_type="bar", max_display=TOP_FEATURES, show=False)
    plt.xlabel("Mean |SHAP value| (global importance)")
    plt.title(f"SHAP global bar plot — {x_global.shape[0]:,} test fragments", pad=16)
    save_figure(figure_dir / "shap_global_bar.png")

    plt.figure(figsize=(8.2, 5.8))
    shap.dependence_plot(0, shap_values, x_values, feature_names=names, interaction_index=1, show=False)
    plt.title(f"SHAP dependence plot — {names[0]}", pad=14)
    save_figure(figure_dir / "shap_dependence_top_feature.png")
    return global_table


def read_split(data_dir: Path, name: str) -> pd.DataFrame:
    frame = pd.read_csv(data_dir / f"wordpress_xai_{name}.csv")
    required = {"record_id", "code", "label_binary", "plugin_slug", "token_len", "code_hash", "clone_family_id"}
    if missing := required - set(frame.columns):
        raise ValueError(f"{name}: отсутствуют поля {sorted(missing)}")
    frame["code"] = frame["code"].fillna("").astype(str)
    labels = pd.to_numeric(frame["label_binary"], errors="raise")
    if set(labels.unique()) != {0, 1}:
        raise ValueError(f"{name}: нужны оба бинарных класса")
    frame["label_binary"] = labels.astype(int)
    if frame["record_id"].duplicated().any() or frame["code"].str.strip().eq("").any():
        raise ValueError(f"{name}: повторяющиеся идентификаторы или пустой код")
    if not frame["split"].eq(name).all():
        raise ValueError(f"{name}: неверные значения поля split")
    return frame


def check_overlap(left: pd.DataFrame, right: pd.DataFrame) -> None:
    for field in ("record_id", "plugin_slug", "code_hash", "clone_family_id"):
        if set(left[field].dropna()) & set(right[field].dropna()):
            raise ValueError(f"Выборки пересекаются по {field}")


def model_input(name: str, frame: pd.DataFrame) -> Any:
    return length_features(frame) if name == "A_Length_LR" else frame["code"]


def build_models() -> dict[str, Pipeline]:
    return {
        "A_Length_LR": Pipeline([
            ("scale", StandardScaler()),
            ("classifier", LogisticRegression(C=1.0, max_iter=2000, random_state=SEED)),
        ]),
        "B_Word_TFIDF_LR": build_word_model(75_000),
        "C_Char_TFIDF_LR": build_char_model(75_000),
    }


def init_explanation_worker(model_path: str, background: Any) -> None:
    global WORKER_MODEL, WORKER_EXPLAINER, WORKER_THREADS
    WORKER_THREADS = threadpool_limits(limits=1)
    WORKER_MODEL = joblib.load(model_path)
    masker = shap.maskers.Independent(background, max_samples=background.shape[0])
    WORKER_EXPLAINER = shap.LinearExplainer(WORKER_MODEL.named_steps["classifier"], masker)


def explain_record(task: tuple) -> tuple[dict, dict, list]:
    record, lime_samples = task
    case = pd.Series(record)
    vectorizer = WORKER_MODEL.named_steps["tfidf"]
    classifier = WORKER_MODEL.named_steps["classifier"]
    x_case = vectorizer.transform([case.code])
    names = vectorizer.get_feature_names_out()
    started = time.perf_counter()
    values = shap_values_for_case(WORKER_EXPLAINER, x_case)
    shap_seconds = time.perf_counter() - started
    expected = float(np.asarray(WORKER_EXPLAINER.expected_value).ravel()[-1])
    # Отдельно берём по пять крупнейших вкладов каждого знака.
    positive = np.flatnonzero(values > 0)
    negative = np.flatnonzero(values < 0)
    indices = np.concatenate((positive[np.argsort(values[positive])[-5:]],
                              negative[np.argsort(values[negative])[:5]]))
    local_shap = pd.DataFrame({"feature": names[indices],
                               "tfidf_value": x_case[:, indices].toarray().ravel(),
                               "shap_value": values[indices]})
    local_shap.attrs["sum_shap"] = float(values.sum())
    started = time.perf_counter()
    local_lime, lime_details = lime_explanation(vectorizer, classifier, case.code, lime_samples)
    lime_seconds = time.perf_counter() - started
    log_odds = float(classifier.decision_function(x_case)[0])
    row = make_explanation_row(case, local_shap, local_lime, expected, log_odds, lime_details)
    comparison = {
        "record_id": int(case.record_id), "outcome": case.outcome,
        "model_probability": float(case.vulnerability_probability),
        "shap_additivity_abs_error": abs(log_odds - expected - float(values.sum())),
        "lime_local_surrogate_r2": lime_details["local_surrogate_r2"],
        "lime_local_surrogate_probability": lime_details["local_surrogate_probability"],
        "shap_seconds": shap_seconds, "lime_seconds": lime_seconds,
    }
    return row, comparison, local_lime.to_dict("records")


def explain_model(model: Pipeline, train: pd.DataFrame, test: pd.DataFrame,
                  predictions: pd.DataFrame, output_dir: Path, lime_samples: int,
                  workers: int = 4) -> None:
    figure_dir = output_dir / "figures"
    explanation_dir = output_dir / "explanations"
    explanation_dir.mkdir(exist_ok=True)
    vectorizer = model.named_steps["tfidf"]
    classifier = model.named_steps["classifier"]
    x_train = vectorizer.transform(train["code"])
    rng = np.random.default_rng(SEED)
    background = x_train[rng.choice(len(train), size=min(200, len(train)), replace=False)]
    masker = shap.maskers.Independent(background, max_samples=background.shape[0])
    explainer = shap.LinearExplainer(classifier, masker)
    expected = float(np.asarray(explainer.expected_value).ravel()[-1])
    names = vectorizer.get_feature_names_out()
    # repr сохраняет видимыми пробелы внутри символьных n-грамм.
    display_names = np.array([repr(name) for name in names])
    merged = test.merge(predictions[["record_id", "vulnerability_probability", "frozen_threshold", "prediction", "outcome"]],
                        on="record_id", validate="one_to_one")
    cases = choose_representative_cases(merged)
    plot_ids = set(cases.record_id)
    comparisons = []
    records = merged.to_dict("records")
    columns = ["record_id", "cve_id", "plugin_slug", "file_path", "function_name",
               "vulnerability_type", "code", "label_binary", "model_prediction",
               "shap_explanation", "lime_explanation"]
    with (explanation_dir / "explanations.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        with ProcessPoolExecutor(max_workers=workers, initializer=init_explanation_worker,
                                 initargs=(str(output_dir / "model.joblib"), background)) as pool:
            results = pool.map(explain_record, ((record, lime_samples) for record in records))
            for position, (row, comparison, lime_records) in enumerate(results):
                writer.writerow(row)
                stream.flush()
                comparisons.append(comparison)
                if row["record_id"] in plot_ids:
                    case = merged.iloc[position]
                    key = f"{case.outcome.lower()}_record_{int(case.record_id)}"
                    x_case = vectorizer.transform([case.code])
                    values = shap_values_for_case(explainer, x_case)
                    plot_shap = pd.DataFrame({"feature": display_names, "shap_value": values})
                    draw_local_shap_bar(plot_shap, case, figure_dir / f"shap_local_bar_{key}.png")
                    draw_local_waterfall(values, expected, x_case, display_names, case, figure_dir / f"shap_waterfall_{key}.png")
                    draw_lime_bar(pd.DataFrame(lime_records), case, figure_dir / f"lime_local_{key}.png")
    pd.DataFrame(comparisons).to_csv(output_dir / "metrics" / "method_comparison.csv", index=False)
    x_test = vectorizer.transform(test["code"])
    subset = rng.choice(len(test), size=min(1000, len(test)), replace=False)
    x_global = x_test[subset]
    scores, baseline = global_shap_scores(x_global, background, classifier)
    ranking = np.argsort(scores)[::-1]
    table = draw_global_shap(x_global, baseline, classifier, display_names, ranking, figure_dir)
    table["feature"] = names[ranking[:len(table)]]
    table.to_csv(output_dir / "metrics" / "shap_global_importance.csv", index=False)


def describe_contributions(table: pd.DataFrame, value_column: str) -> str:
    descriptions = []
    for positive, description in ((True, "Поддерживают класс «уязвимо»"),
                                  (False, "Поддерживают класс «безопасно»")):
        subset = table.loc[table[value_column] > 0] if positive else table.loc[table[value_column] < 0]
        subset = subset.reindex(subset[value_column].abs().sort_values(ascending=False).index).head(5)
        contributions = []
        for _, item in subset.iterrows():
            text = f"{item['feature']!r} ({item[value_column]:+.5f})"
            if "tfidf_value" in table and item["tfidf_value"] == 0:
                text += " — отсутствует во фрагменте"
            contributions.append(text)
        descriptions.append(description + ": " + ("; ".join(contributions) if contributions else "нет среди рассчитанных признаков") + ".")
    return " ".join(descriptions)


def make_explanation_row(case: pd.Series, shap_table: pd.DataFrame, lime_table: pd.DataFrame,
                         expected: float, log_odds: float, lime_details: dict) -> dict:
    columns = ("record_id", "cve_id", "plugin_slug", "file_path", "function_name",
               "vulnerability_type", "code", "label_binary")
    row = {column: "" if pd.isna(case.get(column, "")) else case.get(column, "") for column in columns}
    label = "уязвимо" if int(case.prediction) else "безопасно"
    decision = (f"Решение модели: «{label}». Вероятность уязвимости {case.vulnerability_probability:.4f}, "
                f"рабочий порог {case.frozen_threshold:.4f}. ")
    row["model_prediction"] = int(case.prediction)
    row["shap_explanation"] = (
        decision + describe_contributions(shap_table, "shap_value") + " "
        f"Базовое значение log-odds {expected:.5f}; сумма всех вкладов "
        f"{shap_table.attrs.get('sum_shap', shap_table.shap_value.sum()):+.5f}; итоговое значение модели {log_odds:.5f}. "
        "Вклады относятся к символьным TF-IDF-признакам и объясняют прогноз классификатора."
    )
    row["lime_explanation"] = (
        decision + describe_contributions(lime_table, "lime_weight") + " "
        f"Локальный суррогат оценивает вероятность как {lime_details['local_surrogate_probability']:.4f}, "
        f"R² на возмущённых текстах {lime_details['local_surrogate_r2']:.3f}. "
        "Веса показывают влияние токенов в локальном приближении классификатора."
    )
    return row


def draw_roc(y_true: np.ndarray, probabilities: np.ndarray, path: Path) -> None:
    fpr, tpr, thresholds = roc_curve(y_true, probabilities)
    _, ax = plt.subplots(figsize=(7.2, 5.4))
    ax.plot(fpr, tpr, label=f"ROC-AUC = {roc_auc_score(y_true, probabilities):.3f}")
    ax.plot([0, 1], [0, 1], "--", color="grey", label="Random classifier")
    ax.set(xlabel="False positive rate", ylabel="True positive rate (recall)",
           xlim=(0, 1), ylim=(0, 1.02), title="ROC curve — final test set")
    ax.legend(loc="lower right")
    save_figure(path)




def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--lime-samples", type=int, default=1200)
    parser.add_argument("--xai-workers", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--explain-only", action="store_true",
                        help="Пересчитать объяснения сохранённой модели без повторного обучения")
    args = parser.parse_args()
    if args.lime_samples < 100:
        parser.error("--lime-samples должно быть не меньше 100")
    if args.xai_workers < 1:
        parser.error("--xai-workers должно быть положительным")
    output_dir = args.output_dir.resolve()
    if args.explain_only:
        configure_matplotlib()
        train = read_split(args.data_dir.resolve(), "train")
        test = read_split(args.data_dir.resolve(), "test")
        model = joblib.load(output_dir / "model.joblib")
        if "tfidf" not in model.named_steps:
            parser.error("Текстовые объяснения требуют модели TF-IDF")
        predictions = pd.read_csv(output_dir / "metrics" / "test_predictions.csv")
        matched = test[["record_id"]].merge(predictions, on="record_id", validate="one_to_one")
        if len(matched) != len(test) or not np.allclose(
            model.predict_proba(test["code"])[:, 1], matched["vulnerability_probability"], atol=1e-12, rtol=0
        ):
            raise ValueError("Данные и сохранённые прогнозы не соответствуют модели")
        print("Обучение...", flush=True)
        explain_model(model, train, test, predictions, output_dir, args.lime_samples, args.xai_workers)
        manifest_path = output_dir / "run_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest.update({"explained_records": len(test), "lime_samples": args.lime_samples,
                         "xai_workers": args.xai_workers})
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Результаты: {output_dir}")
        return
    if output_dir.exists() and any(output_dir.iterdir()):
        parser.error("Папка результатов уже содержит файлы. Укажите новую через --output-dir")
    output_dir.mkdir(parents=True, exist_ok=True)
    for directory in ("metrics", "figures"):
        (output_dir / directory).mkdir()
    data_dir = args.data_dir.resolve()
    configure_matplotlib()
    train = read_split(data_dir, "train")
    validation = read_split(data_dir, "validation")
    check_overlap(train, validation)
    models = build_models()
    rows, validation_scores = [], {}
    y_validation = validation["label_binary"].to_numpy()
    print("Обучение...", flush=True)
    for name, model in models.items():
        started = time.perf_counter()
        model.fit(model_input(name, train), train["label_binary"].to_numpy())
        probabilities = model.predict_proba(model_input(name, validation))[:, 1]
        validation_scores[name] = probabilities
        count = 2 if name == "A_Length_LR" else len(model.named_steps["tfidf"].vocabulary_)
        rows.append(metric_row(name, y_validation, probabilities, count, time.perf_counter() - started))
    ranked = pd.DataFrame(rows).sort_values(
        ["pr_auc_average_precision", "mcc", "f1", "model"], ascending=[False, False, False, True])
    selected_name = str(ranked.iloc[0]["model"])
    model = models[selected_name]
    sweep = threshold_sweep(y_validation, validation_scores[selected_name])
    primary = select_row(sweep, "f1")
    alternative = select_row(sweep, "mcc")
    threshold = float(primary["threshold"])
    joblib.dump(model, output_dir / "model.joblib")
    (output_dir / "threshold.json").write_text(json.dumps({
        "model": selected_name, "threshold": threshold, "selection": "validation_f1",
        "alternative_validation_mcc_threshold": float(alternative["threshold"]),
    }, indent=2), encoding="utf-8")
    test = read_split(data_dir, "test")
    check_overlap(train, test)
    check_overlap(validation, test)
    probabilities = model.predict_proba(model_input(selected_name, test))[:, 1]
    y_test = test["label_binary"].to_numpy()
    metrics = metric_bundle(y_test, probabilities, threshold)
    predicted = (probabilities >= threshold).astype(int)
    predictions = test.drop(columns="code").copy()
    predictions["vulnerability_probability"] = probabilities
    predictions["frozen_threshold"] = threshold
    predictions["prediction"] = predicted
    predictions["outcome"] = np.select([
        (y_test == 1) & (predicted == 1), (y_test == 0) & (predicted == 1),
        (y_test == 1) & (predicted == 0)], ["TP", "FP", "FN"], default="TN")
    predictions.to_csv(output_dir / "metrics" / "test_predictions.csv", index=False)
    precision, recall, thresholds = precision_recall_curve(y_test, probabilities)
    pr = pd.DataFrame({"threshold": np.append(thresholds, np.nan), "precision": precision, "recall": recall})
    draw_confusion(pd.Series(metrics), output_dir / "figures" / "confusion_matrix_test.png")
    draw_precision_recall(pr, pd.Series(metrics), output_dir / "figures" / "precision_recall_test.png")
    draw_roc(y_test, probabilities, output_dir / "figures" / "roc_test.png")
    if "tfidf" in model.named_steps:
        explain_model(model, train, test, predictions, output_dir, args.lime_samples, args.xai_workers)
    manifest = {
        "selected_model": selected_name, "threshold": threshold, "random_seed": SEED,
        "protocol": "train -> validation selection -> frozen test evaluation -> XAI",
        "test_used_for_selection": False, "alternative_threshold_applied_to_test": False,
        "split_sizes": {"train": len(train), "validation": len(validation), "test": len(test)},
        "dataset_sha256": {name: sha256(data_dir / f"wordpress_xai_{name}.csv") for name in ("train", "validation", "test")},
        "environment": {package: version(package) for package in ("numpy", "pandas", "scikit-learn", "matplotlib", "shap", "lime", "joblib")},
        "python": platform.python_version(), "lime_samples": args.lime_samples,
        "shap_background_records": min(200, len(train)), "global_shap_records": min(1000, len(test)),
        "model_sha256": sha256(output_dir / "model.joblib"),
        "test_metrics": metrics,
        "explained_records": len(test) if "tfidf" in model.named_steps else 0,
        "xai_workers": args.xai_workers,
    }
    (output_dir / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(pd.DataFrame([metrics]).to_string(index=False))
    print(f"Результаты: {output_dir}")


if __name__ == "__main__":
    main()
