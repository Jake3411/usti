from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_STATE = 42

# 选用的 10 个关键行为特征及维度说明
FEATURES: List[Dict[str, str]] = [
    {"name": "Hours_Studied", "dimension": "学习驱动力", "meaning": "平均每日学习时长（小时）"},
    {"name": "Attendance", "dimension": "学习驱动力", "meaning": "课程出勤率（%）"},
    {"name": "Previous_GPA", "dimension": "学习驱动力", "meaning": "过往 GPA，反映长期学业投入"},
    {"name": "Exam_Anxiety_Score", "dimension": "学习驱动力", "meaning": "考试焦虑评分（0-10），高值可能导致应试型冲刺"},
    {"name": "Study_Method", "dimension": "学习方式", "meaning": "偏好的学习模式：线上/线下/混合"},
    {"name": "Sleep_Hours", "dimension": "学习方式", "meaning": "平均睡眠时长，反映节奏与恢复"},
    {"name": "Screen_Time", "dimension": "学习方式", "meaning": "每日屏幕使用时长（小时），高值可能挤占学习与休息"},
    {"name": "Extracurricular", "dimension": "校园参与", "meaning": "是否参加课外/社团活动"},
    {"name": "Part_Time_Job", "dimension": "校园参与", "meaning": "是否兼职，体现时间分配与责任"},
    {"name": "Stress_Level", "dimension": "校园参与", "meaning": "总体压力感受（0-10），与校园负荷和自我管理相关"},
]

FEATURE_NAMES = [f["name"] for f in FEATURES]

NUMERIC_FEATURES = [
    "Hours_Studied",
    "Attendance",
    "Previous_GPA",
    "Exam_Anxiety_Score",
    "Sleep_Hours",
    "Screen_Time",
    "Stress_Level",
]
CATEGORICAL_FEATURES = ["Study_Method", "Extracurricular", "Part_Time_Job"]

QUESTIONNAIRE: List[Dict[str, Any]] = [
    {
        "id": 1,
        "feature": "Hours_Studied",
        "dimension": "学习驱动力",
        "question": "还有一小时就到晚饭时间了，除了上课，这一个白天你一般是怎么度过的？",
        "type": "select",
        "options": [
            "A. 人已经在图书馆待了很久，今天学习时间大概会有 4–6 小时以上。",
            "B. 白天会抽出一两段时间看课/写作业，加起来差不多 2–3 小时。",
            "C. 只有作业/考试前才会突击一下，大部分时候很少坐下来系统学习。",
        ],
    },
    {
        "id": 2,
        "feature": "Attendance",
        "dimension": "学习驱动力",
        "question": "这个学期，你跟课堂的关系更像是哪种状态？",
        "type": "select",
        "options": [
            "A. 老师在我就在，老师不在我还在。",
            "B. 大部分课都会去，但偶尔会因为赖床/事情太多缺一两节。",
            "C. 经常因为各种原因不去上课，Canvas被设计出来是有它的道理的。",
        ],
    },
    {
        "id": 3,
        "feature": "Previous_GPA",
        "dimension": "学习驱动力",
        "question": "回想上个学期，如果把你的成绩单大概是？",
        "type": "select",
        "options": [
            "A. 整体稳定高分，偶有小翻车，但拿捏啦。",
            "B. 有高有低，偶尔有几门还不错，这已经很棒了。",
            "C. 经常在及格线附近徘徊，有几门甚至触碰到了fail的边缘。",
        ],
    },
    {
        "id": 4,
        "feature": "Exam_Anxiety_Score",
        "dimension": "学习驱动力",
        "question": "还有一周就要期末考了，这时候你往往是？",
        "type": "select",
        "options": [
            "A. 吃不下饭，睡不着觉，脑子里全是考试，我已成仙。",
            "B. 时不时会感到一点紧张，但这种紧张感让我保持兴奋和专注。",
            "C. 该吃吃，该睡睡，平常咋样就咋样。",
        ],
    },
    {
        "id": 5,
        "feature": "Study_Method",
        "dimension": "学习方式",
        "question": "到了真的要投入学习的时候，你更常见的操作是？",
        "type": "select",
        "options": [
            "A. 打开网课/电子教材/ppt，在电脑或平板前完成大部分学习（偏线上）。",
            "B. 既会去教室，也会用各种线上资源，哪种方便用哪种（线上线下混合）。",
            "C. 更喜欢坐在教室、图书馆或纸质书前学习，线上资源只是辅助（偏线下）。",
        ],
    },
    {
        "id": 6,
        "feature": "Sleep_Hours",
        "dimension": "学习方式",
        "question": "不算考试突击周，你平时的睡觉节奏更像是？",
        "type": "select",
        "options": [
            "A. 一般能保证 7–8 小时甚至更多睡眠，属于睡饱再战型。",
            "B. 大多时候能睡到 6–7 小时，偶尔会少一点，但整体还算够用。",
            "C. 经常熬到很晚，睡眠不足 6 小时，但我还活着啊。",
        ],
    },
    {
        "id": 7,
        "feature": "Screen_Time",
        "dimension": "学习方式",
        "question": "在不算学习的时间里，你每天刷手机/电脑娱乐的大致状态是？",
        "type": "select",
        "options": [
            "A. 刷视频、刷社交、打游戏是日常大头，一刷就是好几个小时。",
            "B. 会时不时玩玩，但会有意识地控制，别超过学习时长了。",
            "C. 偶尔看看消息，大部分时间不会长时间盯着屏幕。",
        ],
    },
    {
        "id": 8,
        "feature": "Extracurricular",
        "dimension": "校园参与",
        "question": "关于社团、志愿者、比赛这类课外活动，你更像是哪种角色？",
        "type": "select",
        "options": [
            "A. 经常参加，每周都会有一两次活动，是“活动名单里常见的名字”。",
            "B. 偶尔参加，兴趣来了会报一两个项目，但不会一直很忙。",
            "C. 很少参与，更多把时间留给自己或课内任务。",
        ],
    },
    {
        "id": 9,
        "feature": "Part_Time_Job",
        "dimension": "校园参与",
        "question": "如果把你这学期的时间表画成一张饼图，关于兼职这一块大概是？",
        "type": "select",
        "options": [
            "A. 没有兼职或只偶尔帮忙，不会长期占用固定时间。",
            "B. 有短期/不太稳定的兼职项目，某些周会比较忙。",
            "C. 有稳定的兼职/打工安排，每周固定会占用一部分时间。",
        ],
    },
    {
        "id": 10,
        "feature": "Stress_Level",
        "dimension": "校园参与",
        "question": "回想过去两周，你对学业/生活的整体压力感更接近？",
        "type": "select",
        "options": [
            "A. 事情一件接一件，常觉得被推着走，压力值经常爆表。",
            "B. 有任务也有压力，但基本在可控范围内，忙完能明显放松下来。",
            "C. 节奏相对平稳，偶有小波动，但整体觉得还挺能hold住。",
        ],
    },
]


@dataclass
class USTIArtifacts:
    preprocessor: ColumnTransformer
    kmeans: KMeans
    pca: PCA
    best_k: int
    k_values: List[int]
    elbow: List[float]
    silhouettes: List[float]
    processed_features: np.ndarray
    labels: np.ndarray
    feature_frame: pd.DataFrame
    cluster_profiles: List[Dict[str, Any]]


def load_dataset(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    return df


def clean_and_select(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # 仅保留需要的列
    missing_cols = [c for c in FEATURE_NAMES if c not in df.columns]
    if missing_cols:
        raise ValueError(f"数据集中缺少列: {missing_cols}")

    feature_df = df[FEATURE_NAMES].copy()

    # 数值列处理
    for col in NUMERIC_FEATURES:
        feature_df[col] = pd.to_numeric(feature_df[col], errors="coerce")
        feature_df[col] = feature_df[col].fillna(feature_df[col].median())

    # 分类列处理
    for col in CATEGORICAL_FEATURES:
        feature_df[col] = feature_df[col].astype(str).str.strip()
        mode = feature_df[col].mode().iloc[0]
        feature_df[col] = feature_df[col].replace({"nan": mode, "": mode}).fillna(mode)

    return feature_df


def build_preprocessor(feature_df: pd.DataFrame) -> ColumnTransformer:
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")
    numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ]
    )


def evaluate_kmeans(processed: np.ndarray, k_values: List[int]) -> Tuple[int, List[float], List[float]]:
    elbow: List[float] = []
    silhouettes: List[float] = []
    best_k = k_values[0]
    best_score = -1.0
    for k in k_values:
        model = KMeans(n_clusters=k, n_init=15, random_state=RANDOM_STATE)
        labels = model.fit_predict(processed)
        elbow.append(model.inertia_)
        score = silhouette_score(processed, labels) if k > 1 else float("nan")
        silhouettes.append(score)
        if score > best_score:
            best_score = score
            best_k = k
    return best_k, elbow, silhouettes


def train_usti(data_path: Path, k_values: List[int] | None = None) -> USTIArtifacts:
    raw = load_dataset(data_path)
    feature_df = clean_and_select(raw)

    preprocessor = build_preprocessor(feature_df)
    processed = preprocessor.fit_transform(feature_df)

    k_values = k_values or list(range(6, 9))  # 固定在 6-8 类之间
    best_k, elbow, silhouettes = evaluate_kmeans(processed, k_values)

    kmeans = KMeans(n_clusters=best_k, n_init=20, random_state=RANDOM_STATE)
    labels = kmeans.fit_predict(processed)

    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    processed_dense = processed.toarray() if hasattr(processed, "toarray") else processed
    pca_coords = pca.fit_transform(processed_dense)

    cluster_profiles = build_cluster_profiles(feature_df, labels)

    return USTIArtifacts(
        preprocessor=preprocessor,
        kmeans=kmeans,
        pca=pca,
        best_k=best_k,
        k_values=k_values,
        elbow=elbow,
        silhouettes=silhouettes,
        processed_features=processed_dense,
        labels=labels,
        feature_frame=feature_df.assign(cluster=labels, PC1=pca_coords[:, 0], PC2=pca_coords[:, 1]),
        cluster_profiles=cluster_profiles,
    )


def build_cluster_profiles(feature_df: pd.DataFrame, labels: np.ndarray) -> List[Dict[str, Any]]:
    df = feature_df.copy()
    df["cluster"] = labels
    global_mean = df[NUMERIC_FEATURES].mean()
    global_std = df[NUMERIC_FEATURES].std(ddof=0).replace(0, 1)

    profiles: List[Dict[str, Any]] = []

    for cid, group in df.groupby("cluster"):
        num_means = group[NUMERIC_FEATURES].mean()
        cat_modes = {
            c: group[c].value_counts(normalize=True).to_dict() for c in CATEGORICAL_FEATURES
        }

        signals = []
        if num_means["Hours_Studied"] > global_mean["Hours_Studied"] + 0.5 * global_std["Hours_Studied"]:
            signals.append("高投入学习")
        if num_means["Attendance"] > global_mean["Attendance"] + 0.5 * global_std["Attendance"]:
            signals.append("高出勤")
        if num_means["Screen_Time"] > global_mean["Screen_Time"] + 0.5 * global_std["Screen_Time"]:
            signals.append("高屏幕")
        if num_means["Sleep_Hours"] < global_mean["Sleep_Hours"] - 0.5 * global_std["Sleep_Hours"]:
            signals.append("睡眠不足")
        if num_means["Stress_Level"] > global_mean["Stress_Level"] + 0.5 * global_std["Stress_Level"]:
            signals.append("高压力")
        if num_means["Exam_Anxiety_Score"] > global_mean["Exam_Anxiety_Score"] + 0.5 * global_std["Exam_Anxiety_Score"]:
            signals.append("应试焦虑")
        if cat_modes["Extracurricular"].get("Yes", 0) >= 0.6:
            signals.append("社交/活动活跃")
        if cat_modes["Part_Time_Job"].get("Yes", 0) >= 0.5:
            signals.append("兼职压力")

        # 若特征信号不足，使用Z分数最高/最低的特征补充差异化
        z_scores = (num_means - global_mean) / global_std
        z_sorted = z_scores.sort_values(ascending=False)
        if len(signals) < 2:
            top_positive = [(feat, score) for feat, score in z_sorted.items() if score > 0.25][:2]
            top_negative = [(feat, score) for feat, score in z_sorted[::-1].items() if score < -0.25][:1]
            for feat, _ in top_positive:
                signals.append(f"{feat}偏高")
            for feat, _ in top_negative:
                signals.append(f"{feat}偏低")

        name = craft_cluster_name(signals)
        behavior = describe_behavior(num_means, cat_modes)
        risks = list(set(describe_risks(signals)))
        advice = list(set(describe_advice(signals)))

        profiles.append(
            {
                "cluster": int(cid),
                "name": name,
                "signals": signals,
                "behavior": behavior,
                "risks": risks,
                "advice": advice,
            }
        )

    return profiles


def summarize_clusters(artifacts: USTIArtifacts) -> pd.DataFrame:
    df = pd.DataFrame(artifacts.cluster_profiles)
    # 添加样本数量信息
    counts = pd.Series(artifacts.labels).value_counts().rename("count")
    df = df.merge(counts, left_on="cluster", right_index=True, how="left")
    df = df.rename(columns={"signals": "key_signals", "behavior": "behavior_summary"})
    df["key_signals"] = df["key_signals"].apply(lambda s: "；".join(s))
    return df[["cluster", "name", "key_signals", "behavior_summary", "count"]].sort_values(
        "cluster"
    )


def craft_cluster_name(signals: List[str]) -> str:
    if "高投入学习" in signals and "高出勤" in signals and "高压力" not in signals:
        return "结构型学霸🦉"
    if "高投入学习" in signals and "应试焦虑" in signals:
        return "冲刺派燃烧彗星🚀"
    if "高屏幕" in signals and "睡眠不足" in signals:
        return "夜猫型拖延者🌙"
    if "社交/活动活跃" in signals and "高屏幕" not in signals:
        return "社交探索者🐬"
    if "兼职压力" in signals:
        return "多线平衡者🧭"
    if signals:
        base = signals[0].replace("偏高", "型").replace("偏低", "型")
        return f"{base}探索者✨"
    return "平衡探索者✨"


def describe_behavior(num_means: pd.Series, cat_modes: Dict[str, Dict[str, float]]) -> str:
    top_study_mode = max(cat_modes["Study_Method"], key=cat_modes["Study_Method"].get)
    extracurricular_ratio = cat_modes["Extracurricular"].get("Yes", 0)
    job_ratio = cat_modes["Part_Time_Job"].get("Yes", 0)
    return (
        f"每日学习≈{num_means['Hours_Studied']:.1f}h，出勤{num_means['Attendance']:.0f}%；"
        f"睡眠{num_means['Sleep_Hours']:.1f}h，屏幕{num_means['Screen_Time']:.1f}h；"
        f"偏好{top_study_mode}学习；课外活跃度{extracurricular_ratio:.0%}，兼职占比{job_ratio:.0%}。"
    )


def describe_risks(signals: List[str]) -> List[str]:
    risks = []
    if "高压力" in signals or "应试焦虑" in signals:
        risks.append("压力管理不足，易在考试周透支")
    if "高屏幕" in signals:
        risks.append("娱乐时间侵占学习与睡眠")
    if "睡眠不足" in signals:
        risks.append("睡眠短缺可能影响记忆巩固")
    if "兼职压力" in signals:
        risks.append("兼职/时间碎片化造成注意力分散")
    if not risks:
        risks.append("保持平衡，警惕突发任务导致失衡")
    return risks


def describe_advice(signals: List[str]) -> List[str]:
    advice = []
    if "高压力" in signals or "应试焦虑" in signals:
        advice.append("为考试周提前 2 周做节奏拉长的复习计划，避免一次性突击")
        advice.append("使用 25-5 番茄钟结合夜间放松呼吸，控制交感激活")
    if "高屏幕" in signals:
        advice.append("设定每日娱乐上限（如 2h），将刷屏集中在学习后奖励时段")
    if "睡眠不足" in signals:
        advice.append("保证至少 7h 睡眠，避免 23:30 后继续高强度学习")
    if "兼职压力" in signals:
        advice.append("用时间块管理兼职/学业，周初锁定必做任务，预留缓冲")
    if "社交/活动活跃" in signals:
        advice.append("将社团/比赛反向纳入学习目标，输出型活动可替代部分被动复习")
    if not advice:
        advice.append("保持当前节奏，每周做一次小复盘，检查学习与休息配比")
    return advice


def questionnaire_mapping(answer: Any, feature: str) -> Any:
    def map_level(option_map: Dict[str, int], default_level: int = 3) -> int:
        if isinstance(answer, str):
            return option_map.get(answer, default_level)
        try:
            return int(answer)
        except (TypeError, ValueError):
            return default_level

    if feature == "Hours_Studied":
        level = map_level(
            {
                "A. 人已经在图书馆待了很久，今天学习时间大概会有 4–6 小时以上。": 5,
                "B. 白天会抽出一两段时间看课/写作业，加起来差不多 2–3 小时。": 3,
                "C. 只有作业/考试前才会突击一下，大部分时候很少坐下来系统学习。": 1,
            }
        )
        return {1: 0.8, 3: 4.0, 5: 8.5}.get(level, 4.0)

    if feature == "Attendance":
        level = map_level(
            {
                "A. 老师在我就在，老师不在我还在。": 5,
                "B. 大部分课都会去，但偶尔会因为赖床/事情太多缺一两节。": 3,
                "C. 经常因为各种原因不去上课，Canvas被设计出来是有它的道理的。": 1,
            }
        )
        return {1: 60, 3: 82, 5: 97}.get(level, 82)

    if feature == "Previous_GPA":
        level = map_level(
            {
                "A. 整体稳定高分，偶有小翻车，但拿捏啦。": 5,
                "B. 有高有低，偶尔有几门还不错，这已经很棒了。": 3,
                "C. 经常在及格线附近徘徊，有几门甚至触碰到了fail的边缘。": 1,
            }
        )
        return {1: 1.9, 3: 2.9, 5: 3.8}.get(level, 3.0)

    if feature == "Exam_Anxiety_Score":
        level = map_level(
            {
                "A. 吃不下饭，睡不着觉，脑子里全是考试，我已成仙。": 5,
                "B. 时不时会感到一点紧张，但这种紧张感让我保持兴奋和专注。": 3,
                "C. 该吃吃，该睡睡，平常咋样就咋样。": 1,
            }
        )
        return {1: 2.0, 3: 6.0, 5: 9.5}.get(level, 6.0)

    if feature == "Study_Method":
        mapping = {
            "A. 打开网课/电子教材/ppt，在电脑或平板前完成大部分学习（偏线上）。": "Online",
            "B. 既会去教室，也会用各种线上资源，哪种方便用哪种（线上线下混合）。": "Hybrid",
            "C. 更喜欢坐在教室、图书馆或纸质书前学习，线上资源只是辅助（偏线下）。": "Offline",
        }
        return mapping.get(str(answer), "Hybrid")

    if feature == "Sleep_Hours":
        level = map_level(
            {
                "A. 一般能保证 7–8 小时甚至更多睡眠，属于睡饱再战型。": 5,
                "B. 大多时候能睡到 6–7 小时，偶尔会少一点，但整体还算够用。": 3,
                "C. 经常熬到很晚，睡眠不足 6 小时，但我还活着啊。": 1,
            }
        )
        return {1: 5.0, 3: 6.5, 5: 8.2}.get(level, 6.5)

    if feature == "Screen_Time":
        level = map_level(
            {
                "A. 刷视频、刷社交、打游戏是日常大头，一刷就是好几个小时。": 5,
                "B. 会时不时玩玩，但会有意识地控制，别超过学习时长了。": 3,
                "C. 偶尔看看消息，大部分时间不会长时间盯着屏幕。": 1,
            }
        )
        return {1: 1.0, 3: 3.0, 5: 6.0}.get(level, 3.0)

    if feature == "Extracurricular":
        mapping = {
            "A. 经常参加，每周都会有一两次活动，是“活动名单里常见的名字”。": "Yes",
            "B. 偶尔参加，兴趣来了会报一两个项目，但不会一直很忙。": "Yes",
            "C. 很少参与，更多把时间留给自己或课内任务。": "No",
        }
        return mapping.get(str(answer), "Yes")

    if feature == "Part_Time_Job":
        mapping = {
            "A. 没有兼职或只偶尔帮忙，不会长期占用固定时间。": "No",
            "B. 有短期/不太稳定的兼职项目，某些周会比较忙。": "Yes",
            "C. 有稳定的兼职/打工安排，每周固定会占用一部分时间。": "Yes",
        }
        return mapping.get(str(answer), "No")

    if feature == "Stress_Level":
        level = map_level(
            {
                "A. 事情一件接一件，常觉得被推着走，压力值经常爆表。": 5,
                "B. 有任务也有压力，但基本在可控范围内，忙完能明显放松下来。": 3,
                "C. 节奏相对平稳，偶有小波动，但整体觉得还挺能hold住。": 1,
            }
        )
        return {1: 2.0, 3: 6.0, 5: 9.5}.get(level, 6.0)

    return answer


def answers_to_feature_row(answers: Dict[str, Any]) -> pd.DataFrame:
    mapped = {feat: questionnaire_mapping(ans, feat) for feat, ans in answers.items()}
    # 确保列顺序一致
    return pd.DataFrame([{col: mapped.get(col) for col in FEATURE_NAMES}])


def predict_usti_type(answers: Dict[str, Any], artifacts: USTIArtifacts) -> Dict[str, Any]:
    row = answers_to_feature_row(answers)
    processed = artifacts.preprocessor.transform(row)
    processed_dense = processed.toarray() if hasattr(processed, "toarray") else processed
    cluster_id = int(artifacts.kmeans.predict(processed_dense)[0])

    profile = next((p for p in artifacts.cluster_profiles if p["cluster"] == cluster_id), None)
    return {
        "cluster": cluster_id,
        "profile": profile,
    }


def plot_pca_scatter(artifacts: USTIArtifacts) -> plt.Figure:
    df_plot = artifacts.feature_frame.copy()
    fig, ax = plt.subplots(figsize=(7, 5))
    palette = sns.color_palette("tab10", n_colors=artifacts.best_k)
    sns.scatterplot(data=df_plot, x="PC1", y="PC2", hue="cluster", palette=palette, ax=ax)
    ax.set_title("PCA 2D of clusters")
    ax.grid(True, alpha=0.2)
    return fig


def format_elbow_silhouette(artifacts: USTIArtifacts) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "k": artifacts.k_values,
            "inertia": artifacts.elbow,
            "silhouette": artifacts.silhouettes,
        }
    )


def get_questionnaire() -> List[Dict[str, Any]]:
    return QUESTIONNAIRE


def run_cli_demo(data_path: Path) -> None:
    artifacts = train_usti(data_path)
    print(f"最佳 K: {artifacts.best_k}")
    print(format_elbow_silhouette(artifacts))
    print("\nCluster 解释：")
    for p in artifacts.cluster_profiles:
        print(f"- C{p['cluster']} {p['name']}: {p['behavior']}")
        print(f"  风险: {', '.join(p['risks'])}")
        print(f"  建议: {', '.join(p['advice'])}")


if __name__ == "__main__":
    data_path = Path(__file__).resolve().parent.parent / "student_performance_grade.csv"
    run_cli_demo(data_path)
