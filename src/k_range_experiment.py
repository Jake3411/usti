from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from usti_pipeline import (
    NUMERIC_FEATURES,
    RANDOM_STATE,
    build_preprocessor,
    build_cluster_profiles,
    clean_and_select,
    load_dataset,
)


def run_k_range_experiment(data_path: Path, k_values: range = range(6, 10)) -> None:
    """Run K-means for a range of K and print compact summary stats for each K.

    For each K, we print:
    - silhouette score
    - cluster size distribution and minimum proportion
    - per-cluster key signals and a few core numeric feature means
    """

    raw = load_dataset(data_path)
    feature_df = clean_and_select(raw)

    preprocessor = build_preprocessor(feature_df)
    processed = preprocessor.fit_transform(feature_df)
    processed_dense = processed.toarray() if hasattr(processed, "toarray") else processed

    n_samples = processed_dense.shape[0]

    print(f"Total samples: {n_samples}")
    print(f"K range: {k_values.start}..{k_values.stop - 1}")

    for k in k_values:
        print("\n" + "=" * 50)
        print(f"K = {k}")

        kmeans = KMeans(n_clusters=k, n_init=20, random_state=RANDOM_STATE)
        labels = kmeans.fit_predict(processed_dense)

        # silhouette
        sil = float("nan")
        if k > 1:
            try:
                sil = float(silhouette_score(processed_dense, labels))
            except Exception as e:  # pragma: no cover - debug output only
                print(f"[WARN] silhouette_score failed for K={k}: {e}")

        # cluster size stats
        counts = Counter(labels)
        sizes = [counts[cid] for cid in sorted(counts)]
        props = [s / n_samples for s in sizes]
        min_prop = min(props) if props else float("nan")

        print(f"silhouette_score = {sil:.4f}" if not np.isnan(sil) else "silhouette_score = nan")
        print(f"cluster_sizes = {sizes}")
        print("cluster_props = [" + ", ".join(f"{p:.3f}" for p in props) + "]")
        print(f"min_cluster_prop = {min_prop:.3f}")

        # cluster profiles and typical numeric means
        profiles = build_cluster_profiles(feature_df, np.array(labels))
        df_with_labels = feature_df.copy()
        df_with_labels["cluster"] = labels

        print("cluster_summaries:")
        for p in sorted(profiles, key=lambda x: x["cluster"]):
            cid = p["cluster"]
            group = df_with_labels[df_with_labels["cluster"] == cid]
            means = group[NUMERIC_FEATURES].mean()
            print(
                "  "
                + f"cluster {cid}: "
                + f"signals={p['signals']}, "
                + f"Hours={means['Hours_Studied']:.2f}, "
                + f"GPA={means['Previous_GPA']:.2f}, "
                + f"Sleep={means['Sleep_Hours']:.2f}, "
                + f"Screen={means['Screen_Time']:.2f}, "
                + f"Stress={means['Stress_Level']:.2f}"
            )


if __name__ == "__main__":
    data_path = Path(__file__).resolve().parent.parent / "student_performance_grade.csv"
    run_k_range_experiment(data_path)
