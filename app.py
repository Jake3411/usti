from pathlib import Path
from typing import Any, Dict

import pandas as pd
import streamlit as st

from src.usti_pipeline import (
    FEATURES,
    get_questionnaire,
    predict_usti_type,
    train_usti,
    answers_to_feature_row,
    format_elbow_silhouette,
    plot_pca_scatter,
    summarize_clusters,
    sample_cluster_examples,
)

DATA_PATH = Path(__file__).resolve().parent / "student_performance_grade.csv"

LANG_OPTIONS = {"zh": "中文", "en": "English"}

# UI 文案
TEXT: Dict[str, Dict[str, str]] = {
    "zh": {
        "page_title": "USTI · HKUST",
        "main_title": "USTI · University Student Type Indicator",
        "main_caption": "聚类固定为 6 类（C0–C5 数据驱动），另有规则型 C6/C7 由阈值判定",
        "feature_expander": "查看 10 个关键特征含义",
        "questionnaire_header": "10 题行为问卷",
        "submit_btn": "提交并查看 USTI 类型",
        "result_header": "你的 USTI 类型",
        "rule_caption": "（规则命中：本次类型由阈值直接判定为 C6/C7，模型概率未展示）",
        "kmeans_caption": "（对比：若直接按聚类最近中心，你更接近）",
        "type_intro": "类型介绍",
        "type_risks": "潜在问题",
        "type_advice": "改进建议",
        "prob_title_tree": "类型匹配度（基于决策树分类器）",
        "prob_empty": "暂无概率信息",
        "prob_title_kmeans": "类型匹配度（基于聚类距离/KMeans）",
        "prob_kmeans_empty": "暂无基于聚类的匹配度信息",
        "importances_title": "区分度最高的特征（模型重要性）",
        "importances_empty": "暂无特征重要性信息",
        "samples_title": "典型样本（同类型 3 个例子）",
        "samples_empty": "暂时没有找到同类型样本，可尝试调整数据或重新训练。",
        "summary_title": "聚类类型特征概览（C0–C5 数据驱动）",
        "answers_title": "你的回答对应的标准化特征",
        "process_expander_title": "查看聚类/KMeans 过程（完成问卷后可见）",
        "process_metric_title": "最佳 K (Silhouette)",
        "process_metric_caption": "K 固定为 6，依据轮廓系数验证聚类效果",
        "hint_modify": "提示：你可以修改答案，观察类型如何变化。",
        "lang_label": "界面语言 / Language",
    },
    "en": {
        "page_title": "USTI · HKUST",
        "main_title": "USTI · University Student Type Indicator",
        "main_caption": "Clustering fixed to 6 types (C0–C5 data-driven); rule-based C6/C7 decided by thresholds",
        "feature_expander": "See meanings of 10 key features",
        "questionnaire_header": "10-question behavior survey",
        "submit_btn": "Submit to view USTI type",
        "result_header": "Your USTI type",
        "rule_caption": "(Rule hit: this result is rule-based C6/C7; model probabilities hidden)",
        "kmeans_caption": "(Reference: nearest KMeans center would be)",
        "type_intro": "Type intro",
        "type_risks": "Potential risks",
        "type_advice": "Suggestions",
        "prob_title_tree": "Type match (Decision Tree)",
        "prob_empty": "No probability available",
        "prob_title_kmeans": "Type match (KMeans distance)",
        "prob_kmeans_empty": "No KMeans match available",
        "importances_title": "Top separating features (importance)",
        "importances_empty": "No feature importance available",
        "samples_title": "Typical samples (3 examples)",
        "samples_empty": "No samples found for this type. Try adjusting data or retraining.",
        "summary_title": "Cluster overview (C0–C5 data-driven)",
        "answers_title": "Your standardized features",
        "process_expander_title": "See clustering/KMeans process (visible after submission)",
        "process_metric_title": "Best K (Silhouette)",
        "process_metric_caption": "K fixed at 6; silhouette to validate clustering",
        "hint_modify": "Tip: tweak your answers to see how the type changes.",
        "lang_label": "界面语言 / Language",
    },
}

# 问卷英文文案（保留原始选项作为真实取值，便于映射）
QUESTION_I18N: Dict[int, Dict[str, Any]] = {
    1: {
        "question_en": "It's one hour before dinner. Besides classes, how did you spend the day?",
        "options_en": [
            "A. Spent most of the day in the library; likely 4–6+ hours of study.",
            "B. Found one or two study blocks, about 2–3 hours in total.",
            "C. Rarely sit down to study unless a deadline/exam is near.",
        ],
    },
    2: {
        "question_en": "This semester, which best describes your attendance?",
        "options_en": [
            "A. I'm there whenever the instructor is (and often even if not).",
            "B. I go to most classes, occasionally miss a couple.",
            "C. Frequently skip; Canvas was invented for a reason.",
        ],
    },
    3: {
        "question_en": "Thinking of last term, your transcript looks like?",
        "options_en": [
            "A. Consistently high grades with minor hiccups.",
            "B. Mixed; some highs and lows, a few good courses.",
            "C. Hovering near pass line; some courses almost failed.",
        ],
    },
    4: {
        "question_en": "One week before finals, you are usually?",
        "options_en": [
            "A. Can't eat or sleep; mind is all exams.",
            "B. A bit nervous, which keeps me focused.",
            "C. Business as usual; eat and sleep fine.",
        ],
    },
    5: {
        "question_en": "When you actually start studying, you usually?",
        "options_en": [
            "A. Use online courses/materials; mostly study on computer/tablet.",
            "B. Mix of in-person and online, whichever works.",
            "C. Prefer in-person/printed materials; online is just support.",
        ],
    },
    6: {
        "question_en": "On regular weeks (not cram), your sleep rhythm is?",
        "options_en": [
            "A. Usually 7–8+ hours; well-rested.",
            "B. Mostly 6–7 hours; occasionally less but generally ok.",
            "C. Often sleep <6 hours, staying up late.",
        ],
    },
    7: {
        "question_en": "Daily non-study screen time for entertainment is?",
        "options_en": [
            "A. Major part of the day; hours of videos/social/games.",
            "B. Use it but try to control it, not exceeding study time.",
            "C. Occasionally check messages; rarely long screen time.",
        ],
    },
    8: {
        "question_en": "Regarding clubs/volunteering/competitions, you're more like?",
        "options_en": [
            "A. Join frequently; on the activity list weekly.",
            "B. Join occasionally when interested; not always busy.",
            "C. Seldom participate; keep time for self or coursework.",
        ],
    },
    9: {
        "question_en": "If your weekly schedule is a pie chart, the slice for part-time job is?",
        "options_en": [
            "A. None or very occasional help; no fixed time.",
            "B. Short-term/irregular gigs; some weeks busy.",
            "C. A steady job; fixed hours every week.",
        ],
    },
    10: {
        "question_en": "Past two weeks, your overall stress about study/life is?",
        "options_en": [
            "A. Tasks keep coming; stress often maxed.",
            "B. Stress exists but manageable; can relax after tasks.",
            "C. Fairly steady rhythm; mostly able to hold it.",
        ],
    },
}

FEATURE_MEANING_EN = {
    "Hours_Studied": "Average daily study hours",
    "Attendance": "Course attendance rate (%)",
    "Previous_GPA": "Previous GPA, reflecting long-term academic engagement",
    "Exam_Anxiety_Score": "Exam anxiety score (0-10)",
    "Study_Method": "Preferred study mode: online/offline/mixed",
    "Sleep_Hours": "Average sleep duration",
    "Screen_Time": "Daily screen time (entertainment)",
    "Extracurricular": "Participation in clubs/activities",
    "Part_Time_Job": "Whether having a part-time job",
    "Stress_Level": "Overall perceived stress (0-10)",
}

FEATURE_DIMENSION_EN = {
    "学习驱动力": "Study drive",
    "学习方式": "Study mode",
    "校园参与": "Campus engagement",
}

PROFILE_EN: Dict[int, Dict[str, str]] = {
    0: {
        "title": "C0 · genius",
        "intro": "You grasp new concepts quickly and often excel with seemingly little effort.",
        "risks_text": "Relying on intuition may hide gaps; advanced courses can become challenging.",
        "advice_text": "Teach others, build your own notes, and add deliberate practice to your talent.",
    },
    1: {
        "title": "C1 · rat king",
        "intro": "You live in the library and keep a tightly packed study schedule.",
        "risks_text": "High self-demand may bring anxiety and burnout when results fluctuate.",
        "advice_text": "Set off-hours, focus on quality over hours, and recharge regularly.",
    },
    2: {
        "title": "C2 · worrier",
        "intro": "Many plans stay in your head; action often lags behind ideas.",
        "risks_text": "Overthinking prevents starting; last-minute rush reinforces self-doubt.",
        "advice_text": "Start tiny tasks (15 minutes) and value completion over perfection.",
    },
    3: {
        "title": "C3 · empty soul",
        "intro": "You attend but often disengage; classes feel like a checklist.",
        "risks_text": "Missed basics may hurt later; sudden panic can emerge near key milestones.",
        "advice_text": "Find small interesting hooks and set short, focused goals to rebuild engagement.",
    },
    4: {
        "title": "C4 · Xiao Ming",
        "intro": "You work hard but results feel average, leading to self-doubt.",
        "risks_text": "Effort without reflection can become exhausting and discouraging.",
        "advice_text": "Treat yourself as a tunable system; use review and structured notes to improve.",
    },
    5: {
        "title": "C5 · Princess Aurora",
        "intro": "Sleep patterns dominate your rhythm; rest is precious yet conflicts with tasks.",
        "risks_text": "Irregular sleep hurts focus and may cause a cycle of cramming and fatigue.",
        "advice_text": "Anchor a latest bedtime, protect 7h sleep, and align hard tasks with awake hours.",
    },
    6: {
        "title": "C6 · lier",
        "intro": "You keep expectations low and run on minimal effort to avoid stress.",
        "risks_text": "Long-term low investment limits options and reinforces self-restriction.",
        "advice_text": "Add small anchors weekly to preserve future flexibility without over-pressuring now.",
    },
    7: {
        "title": "C7 · social bro",
        "intro": "You thrive in activities, projects, and people over classroom routines.",
        "risks_text": "Ignoring academic basics can hurt at GPA/requirement checkpoints.",
        "advice_text": "Translate real projects into academic language and set a GPA/required-course baseline.",
    },
}

SIGNAL_EN_MAP = {
    "高投入学习": "High study input",
    "高出勤": "High attendance",
    "高屏幕": "High screen time",
    "睡眠不足": "Insufficient sleep",
    "高压力": "High stress",
    "应试焦虑": "Exam anxiety",
    "社交/活动活跃": "Active in social/activities",
    "兼职压力": "Part-time workload",
}


def translate_signal(sig: str) -> str:
    if sig in SIGNAL_EN_MAP:
        return SIGNAL_EN_MAP[sig]
    if sig.endswith("偏高"):
        return f"{sig[:-2]} high"
    if sig.endswith("偏低"):
        return f"{sig[:-2]} low"
    return sig


def translate_signals_text(text: str) -> str:
    parts = [p for p in text.split("；") if p]
    return "; ".join(translate_signal(p) for p in parts)


@st.cache_resource(show_spinner=True)
def load_artifacts():
    return train_usti(DATA_PATH)


def get_query_params() -> Dict[str, Any]:
    if hasattr(st, "query_params"):
        try:
            return dict(st.query_params)
        except Exception:
            return {}
    try:
        return st.experimental_get_query_params()
    except Exception:
        return {}


def set_query_params(**kwargs: Any) -> None:
    if hasattr(st, "query_params"):
        for k, v in kwargs.items():
            st.query_params[k] = v
    else:
        st.experimental_set_query_params(**kwargs)


def init_language() -> str:
    params = get_query_params()
    default = params.get("lang", [st.session_state.get("lang", "zh")])[0] if isinstance(params.get("lang"), list) else params.get("lang", st.session_state.get("lang", "zh"))
    if default not in LANG_OPTIONS:
        default = "zh"
    st.session_state.setdefault("lang", default)
    return st.session_state["lang"]


def t(key: str, lang: str) -> str:
    return TEXT.get(lang, TEXT["zh"]).get(key, TEXT["zh"].get(key, key))


def render_language_switch(lang: str) -> str:
    st.session_state.setdefault("lang", lang)
    options = list(LANG_OPTIONS.keys())
    chosen = st.sidebar.selectbox(
        t("lang_label", lang),
        options=options,
        format_func=lambda k: LANG_OPTIONS[k],
        key="lang",
    )
    if chosen != lang:
        set_query_params(lang=chosen)
    return chosen


def render_feature_meanings(lang: str):
    with st.expander(t("feature_expander", lang)):
        for feat in FEATURES:
            dimension = feat.get("dimension") or ""
            if lang == "en":
                dimension = FEATURE_DIMENSION_EN.get(dimension, dimension)
            meaning = feat.get("meaning", "")
            if lang == "en":
                meaning = FEATURE_MEANING_EN.get(feat["name"], meaning)
            st.markdown(f"- **{feat['name']} ({dimension})**：{meaning}")


def render_questionnaire(lang: str) -> Dict[str, Any] | None:
    questionnaire = get_questionnaire()
    answers: Dict[str, Any] = {}
    with st.form("usti_form"):
        for q in questionnaire:
            meta = QUESTION_I18N.get(q["id"], {})
            question_text = q["question"] if lang == "zh" else meta.get("question_en", q["question"])
            if q["type"] == "select":
                base_options = q["options"]
                display_options = base_options if lang == "zh" else meta.get("options_en", base_options)
                option_indices = list(range(len(base_options)))
                selected_index = st.selectbox(
                    label=f"Q{q['id']} · {question_text}",
                    options=option_indices,
                    format_func=lambda idx: display_options[idx],
                )
                answers[q["feature"]] = base_options[selected_index]
            elif q["type"] == "scale":
                anchors = q.get("anchors", [])
                anchors_text = " / ".join(anchors) if anchors else None
                answers[q["feature"]] = st.slider(
                    label=f"Q{q['id']} · {question_text}",
                    min_value=int(q["range"][0]),
                    max_value=int(q["range"][1]),
                    value=int((q["range"][0] + q["range"][1]) / 2),
                    help=anchors_text,
                )
        submitted = st.form_submit_button(t("submit_btn", lang))
    return answers if submitted else None


def localize_profile(profile: Dict[str, Any] | None, cluster_id: int | None, lang: str) -> Dict[str, Any]:
    if lang == "zh" or cluster_id is None:
        return profile or {}
    en_profile = PROFILE_EN.get(cluster_id)
    if not en_profile:
        return profile or {}
    return {
        **(profile or {}),
        "title": en_profile.get("title", profile.get("title") if profile else None),
        "intro": en_profile.get("intro", profile.get("intro") if profile else None),
        "risks_text": en_profile.get("risks_text", profile.get("risks_text") if profile else None),
        "advice_text": en_profile.get("advice_text", profile.get("advice_text") if profile else None),
    }


def main() -> None:
    lang = init_language()
    st.set_page_config(page_title=t("page_title", lang), layout="wide")
    lang = render_language_switch(lang)

    st.title(t("main_title", lang))
    st.caption(t("main_caption", lang))

    artifacts = load_artifacts()

    render_feature_meanings(lang)

    st.markdown("---")
    st.header(t("questionnaire_header", lang))
    answers = render_questionnaire(lang)

    if answers is not None:
        result = predict_usti_type(answers, artifacts)
        cluster_id = result["cluster"]
        kmeans_cluster_id = result.get("kmeans_cluster")
        profile_raw = result.get("profile") or {}
        profile = localize_profile(profile_raw, cluster_id, lang)
        title = profile.get("title") or profile.get("name") or f"C{cluster_id}"
        probs = result.get("type_probabilities") or {}
        cluster_probs = result.get("cluster_probabilities") or {}
        importances = result.get("feature_importances") or {}
        rule_based = result.get("rule_based", False)

        st.subheader(f"{t('result_header', lang)}：{title}")
        if rule_based:
            st.caption(t("rule_caption", lang))
        if kmeans_cluster_id is not None and kmeans_cluster_id != cluster_id:
            st.caption(f"{t('kmeans_caption', lang)} C{kmeans_cluster_id}")

        intro_text = profile.get("intro") or profile.get("behavior") or ""
        risks_text = profile.get("risks_text") or "；".join(profile.get("risks", [])) or ("暂未识别" if lang == "zh" else "Not identified")
        advice_text = profile.get("advice_text") or "；".join(profile.get("advice", [])) or ("保持当前节奏，继续小步复盘。" if lang == "zh" else "Keep your pace and review regularly.")

        st.markdown(f"**{t('type_intro', lang)}**：{intro_text}")
        st.markdown(f"**{t('type_risks', lang)}**：{risks_text}")
        st.markdown(f"**{t('type_advice', lang)}**：{advice_text}")

        st.markdown(f"### {t('prob_title_tree', lang)}")
        if probs:
            prob_df = pd.DataFrame([probs]).T.reset_index()
            prob_df.columns = ["Type", "Probability"]
            prob_df["Type"] = prob_df["Type"].apply(lambda x: f"C{x}")
            prob_df["Probability"] = prob_df["Probability"].apply(lambda x: f"{x:.1%}")
            prob_df = prob_df.rename(columns={"Type": "类型" if lang == "zh" else "Type", "Probability": "概率" if lang == "zh" else "Probability"})
            st.dataframe(prob_df)
        else:
            st.info(t("prob_empty", lang))

        st.markdown(f"### {t('prob_title_kmeans', lang)}")
        if cluster_probs:
            kmeans_prob_df = pd.DataFrame([cluster_probs]).T.reset_index()
            kmeans_prob_df.columns = ["Type", "Probability"]
            kmeans_prob_df["Type"] = kmeans_prob_df["Type"].apply(lambda x: f"C{x}")
            kmeans_prob_df["Probability"] = kmeans_prob_df["Probability"].apply(lambda x: f"{x:.1%}")
            kmeans_prob_df = kmeans_prob_df.rename(columns={"Type": "类型" if lang == "zh" else "Type", "Probability": "概率" if lang == "zh" else "Probability"})
            st.dataframe(kmeans_prob_df)
        else:
            st.info(t("prob_kmeans_empty", lang))

        st.markdown(f"### {t('importances_title', lang)}")
        if importances:
            imp_df = pd.DataFrame(
                [
                    {"特征" if lang == "zh" else "Feature": feat, "重要性" if lang == "zh" else "Importance": score}
                    for feat, score in sorted(importances.items(), key=lambda kv: kv[1], reverse=True)
                ]
            )
            imp_col = "重要性" if lang == "zh" else "Importance"
            imp_df[imp_col] = imp_df[imp_col].apply(lambda x: f"{x:.2%}")
            st.dataframe(imp_df)
        else:
            st.info(t("importances_empty", lang))

        st.markdown(f"### {t('samples_title', lang)}")
        samples = sample_cluster_examples(artifacts, cluster_id)
        if samples.empty:
            st.info(t("samples_empty", lang))
        else:
            st.dataframe(samples)

        st.markdown(f"### {t('summary_title', lang)}")
        summary_df = summarize_clusters(artifacts)
        if lang == "en":
            summary_df["name"] = summary_df["cluster"].apply(
                lambda cid: PROFILE_EN.get(int(cid), {}).get("title", f"C{cid}")
            )
            summary_df["behavior_summary"] = summary_df["cluster"].apply(
                lambda cid: PROFILE_EN.get(int(cid), {}).get("intro", summary_df.loc[summary_df["cluster"] == cid, "behavior_summary"].values[0])
            )
            summary_df["key_signals"] = summary_df["key_signals"].apply(translate_signals_text)
            summary_df = summary_df.rename(
                columns={
                    "cluster": "Cluster",
                    "name": "Type",
                    "key_signals": "Key signals",
                    "behavior_summary": "Behavior summary",
                    "count": "Samples",
                }
            )
        st.dataframe(summary_df)

        st.markdown(f"### {t('answers_title', lang)}")
        row = answers_to_feature_row(answers, artifacts.quantiles)
        st.dataframe(row)

        with st.expander(t("process_expander_title", lang), expanded=False):
            st.metric(t("process_metric_title", lang), artifacts.best_k)
            st.caption(t("process_metric_caption", lang))
            col_left, col_right = st.columns([2, 1])
            with col_right:
                st.dataframe(format_elbow_silhouette(artifacts))
            with col_left:
                st.pyplot(plot_pca_scatter(artifacts))

        st.success(t("hint_modify", lang))


if __name__ == "__main__":
    main()
