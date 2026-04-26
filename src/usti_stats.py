from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import pandas as pd

from .usti_pipeline import FEATURE_NAMES, QUESTIONNAIRE


RESULT_LOG_PATH = Path(__file__).resolve().parent.parent / "usti_results.csv"


def get_results_path() -> Path:
    """返回用户结果日志 CSV 的路径。"""

    return RESULT_LOG_PATH


def append_result_record(user_id: str, answers: Dict[str, Any], result: Dict[str, Any], lang: str) -> None:
    """将一次答卷结果追加写入 CSV。

    user_id 可以来自用户输入，也可以是自动生成的匿名 ID。
    同一 user_id 多次作答会记录多行，后续统计时按时间排序仅保留首条记录。
    """

    path = get_results_path()

    row: Dict[str, Any] = {
        "user_id": user_id,
        "lang": lang,
        "cluster": result.get("cluster"),
        "kmeans_cluster": result.get("kmeans_cluster"),
        "rule_based": result.get("rule_based", False),
        "timestamp": pd.Timestamp.utcnow().isoformat(),
    }

    # 记录每个特征（对应每道题）的原始回答文本，便于后续统计选项占比
    for feat in FEATURE_NAMES:
        row[feat] = answers.get(feat)

    df = pd.DataFrame([row])
    path.parent.mkdir(parents=True, exist_ok=True)
    header = not path.exists()
    df.to_csv(path, mode="a", header=header, index=False)


def _load_results() -> pd.DataFrame:
    """加载全部作答结果，如果文件不存在则返回空表。"""

    path = get_results_path()
    if not path.exists():
        return pd.DataFrame()

    df = pd.read_csv(path)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    return df


def _first_attempt_only(df: pd.DataFrame) -> pd.DataFrame:
    """按 user_id 仅保留首条作答记录。"""

    if df.empty or "user_id" not in df.columns:
        return df

    df = df.dropna(subset=["user_id"]).copy()
    if "timestamp" in df.columns:
        df = df.sort_values("timestamp")
    return df.drop_duplicates(subset=["user_id"], keep="first")


# 建立 feature -> 问题文案 的映射，方便展示
_FEATURE_TO_QUESTION: Dict[str, Dict[str, Any]] = {q["feature"]: q for q in QUESTIONNAIRE}


def compute_question_option_stats() -> pd.DataFrame:
    """计算每道题各选项占比（仅按每位用户首答记录统计）。"""

    df = _load_results()
    if df.empty:
        return pd.DataFrame()

    df = _first_attempt_only(df)

    rows = []
    for feat in FEATURE_NAMES:
        if feat not in df.columns:
            continue
        series = df[feat].dropna().astype(str)
        if series.empty:
            continue

        counts = series.value_counts()
        total = counts.sum()

        q_meta = _FEATURE_TO_QUESTION.get(feat, {})
        q_id = q_meta.get("id", "")
        q_text = q_meta.get("question", feat)

        for option_text, count in counts.items():
            rows.append(
                {
                    "question_id": q_id,
                    "question": q_text,
                    "option": option_text,
                    "count": int(count),
                    "ratio": float(count) / float(total) if total else 0.0,
                }
            )

    if not rows:
        return pd.DataFrame()

    result_df = pd.DataFrame(rows)
    result_df = result_df.sort_values(["question_id", "option"]).reset_index(drop=True)
    result_df["ratio"] = result_df["ratio"].apply(lambda x: f"{x:.1%}")
    return result_df


def compute_type_distribution() -> pd.DataFrame:
    """计算人格类型占比（cluster 维度，仅按每位用户首答记录统计）。"""

    df = _load_results()
    if df.empty or "cluster" not in df.columns:
        return pd.DataFrame()

    df = _first_attempt_only(df)
    series = df["cluster"].dropna()
    if series.empty:
        return pd.DataFrame()

    series = series.astype(int)
    counts = series.value_counts().sort_index()
    total = counts.sum()

    rows = []
    for cid, count in counts.items():
        rows.append(
            {
                "cluster": int(cid),
                "type": f"C{int(cid)}",
                "count": int(count),
                "ratio": float(count) / float(total) if total else 0.0,
            }
        )

    type_df = pd.DataFrame(rows)
    type_df["ratio"] = type_df["ratio"].apply(lambda x: f"{x:.1%}")
    return type_df


def get_stats() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """返回（每题选项占比表，人格类型占比表）。"""

    question_stats = compute_question_option_stats()
    type_stats = compute_type_distribution()
    return question_stats, type_stats
