import ipaddress
from pathlib import Path
from urllib.parse import urlparse

import joblib
import numpy as np
import pandas as pd
import tldextract

from sklearn.model_selection import GroupShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    brier_score_loss,
    log_loss,
    roc_auc_score,
    confusion_matrix,
)

from src.feature_extractor import extract_dataset_features


def get_domain(url):
    hostname = (urlparse(str(url)).hostname or "").lower()

    if not hostname:
        raise ValueError(f"Missing hostname: {url}")

    try:
        ipaddress.ip_address(hostname)
        return hostname
    except ValueError:
        pass

    parts = tldextract.extract(hostname)
    return parts.top_domain_under_public_suffix or hostname


def get_scores(model, df):
    X = extract_dataset_features(df["url"])
    X = X[list(model.feature_names_in_)]
    y = df["label"].to_numpy()
    scores = model.predict_proba(X)[:, 1]
    return y, scores


def ece(y, scores, n_bins=10):
    edges = np.linspace(0, 1, n_bins + 1)
    bin_ids = np.minimum(
        np.searchsorted(edges[1:], scores, side="right"),
        n_bins - 1
    )

    result = 0.0

    for i in range(n_bins):
        mask = bin_ids == i

        if mask.any():
            result += mask.mean() * abs(
                y[mask].mean() - scores[mask].mean()
            )

    return result


def evaluate(dataset, method, y, scores):
    pred = (scores >= 0.5).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y, pred, labels=[0, 1]
    ).ravel()

    return {
        "Dataset": dataset,
        "Method": method,
        "N": len(y),
        "Brier": brier_score_loss(y, scores),
        "LogLoss": log_loss(y, scores, labels=[0, 1]),
        "ECE": ece(y, scores),
        "AUC": roc_auc_score(y, scores),
        "FP": int(fp),
        "FN": int(fn),
    }


def main():
    model = joblib.load(
        "models/diverse/random_forest_diverse.pkl"
    )
    assert list(model.classes_) == [0, 1]

    comp = pd.read_csv("data/compphish/validation.csv")
    phi = pd.read_csv("data/diverse/phi_eval.csv")

    comp["domain"] = comp["url"].map(get_domain)
    phi["domain"] = phi["url"].map(get_domain)

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.40,
        random_state=42
    )

    fit_idx, eval_idx = next(
        splitter.split(
            comp,
            comp["label"],
            groups=comp["domain"]
        )
    )

    fit_df = comp.iloc[fit_idx]
    eval_df = comp.iloc[eval_idx]

    assert set(fit_df["domain"]).isdisjoint(
        set(eval_df["domain"])
    )

    assert set(fit_df["domain"]).isdisjoint(
        set(phi["domain"])
    )

    assert set(eval_df["domain"]).isdisjoint(
        set(phi["domain"])
    )

    for name, df in [
        ("Calibration fit", fit_df),
        ("CompPhish evaluation", eval_df),
        ("PhiUSIIL evaluation", phi),
    ]:
        print(
            f"{name}: {len(df)} URLs, "
            f"{df['domain'].nunique()} domains, "
            f"phishing prevalence={df['label'].mean():.4f}"
        )
        assert set(df["label"]) == {0, 1}, (
            f"{name} must contain both classes"
        )

    y_fit, raw_fit = get_scores(model, fit_df)
    y_comp, raw_comp = get_scores(model, eval_df)
    y_phi, raw_phi = get_scores(model, phi)

    sigmoid = LogisticRegression(
        random_state=42,
        max_iter=1000
    )
    sigmoid.fit(raw_fit.reshape(-1, 1), y_fit)

    isotonic = IsotonicRegression(
        y_min=0,
        y_max=1,
        out_of_bounds="clip"
    )
    isotonic.fit(raw_fit, y_fit)

    rows = []

    for dataset, y, raw in [
        ("CompPhish held-out domains", y_comp, raw_comp),
        ("PhiUSIIL transfer", y_phi, raw_phi),
    ]:
        sigmoid_scores = sigmoid.predict_proba(
            raw.reshape(-1, 1)
        )[:, 1]

        isotonic_scores = isotonic.predict(raw)

        for method, scores in [
            ("Original", raw),
            ("Sigmoid", sigmoid_scores),
            ("Isotonic", isotonic_scores),
        ]:
            rows.append(
                evaluate(dataset, method, y, scores)
            )

    results = pd.DataFrame(rows)

    print("\n=== DOMAIN-DISJOINT CALIBRATION RESULTS ===")
    print(results.round(5).to_string(index=False))

    Path("results").mkdir(exist_ok=True)
    results.to_csv(
        "results/domain_disjoint_calibration.csv",
        index=False
    )

    print(
        "\nSaved: results/domain_disjoint_calibration.csv"
    )


if __name__ == "__main__":
    main()
