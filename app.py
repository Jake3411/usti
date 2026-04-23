from pathlib import Path
from typing import Dict, Any

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


@st.cache_resource(show_spinner=True)
def load_artifacts():
    return train_usti(DATA_PATH)


def render_feature_meanings():
    with st.expander("查看 10 个关键特征含义"):
        for feat in FEATURES:
            st.markdown(f"- **{feat['name']} ({feat['dimension']})**：{feat['meaning']}")


def render_questionnaire() -> Dict[str, Any] | None:
    questionnaire = get_questionnaire()
    answers: Dict[str, Any] = {}
    with st.form("usti_form"):
        for q in questionnaire:
            if q["type"] == "scale":
                answers[q["feature"]] = st.slider(
                    label=f"Q{q['id']} · {q['question']}",
                    min_value=int(q["range"][0]),
                    max_value=int(q["range"][1]),
                    value=int((q["range"][0] + q["range"][1]) / 2),
                    help=" / ".join(q.get("anchors", [])),
                )
            elif q["type"] == "select":
                answers[q["feature"]] = st.selectbox(
                    label=f"Q{q['id']} · {q['question']}",
                    options=q["options"],
                )
        submitted = st.form_submit_button("提交并查看 USTI 类型")
    return answers if submitted else None


def main() -> None:
    st.set_page_config(page_title="USTI · HKUST", layout="wide")
    st.title("USTI · University Student Type Indicator")
    st.caption("聚类固定为 6 类，完成问卷后可查看训练过程")

    artifacts = load_artifacts()

    render_feature_meanings()

    st.markdown("---")
    st.header("10 题行为问卷")
    answers = render_questionnaire()

    if answers is not None:
        result = predict_usti_type(answers, artifacts)
        cluster_id = result["cluster"]
        profile = result.get("profile") or {}
        title = profile.get("title") or profile.get("name") or f"C{cluster_id}"

        st.subheader(f"你的 USTI 类型：{title}")
        intro_text = profile.get("intro") or profile.get("behavior") or ""
        risks_text = profile.get("risks_text") or "；".join(profile.get("risks", [])) or "暂未识别"
        advice_text = profile.get("advice_text") or "；".join(profile.get("advice", [])) or "保持当前节奏，继续小步复盘。"

        st.markdown("**类型介绍**：" + intro_text)
        st.markdown("**潜在问题**：" + risks_text)
        st.markdown("**改进建议**：" + advice_text)

        st.markdown("### 典型样本（同类型 3 个例子）")
        samples = sample_cluster_examples(artifacts, cluster_id)
        if samples.empty:
            st.info("暂时没有找到同类型样本，可尝试调整数据或重新训练。")
        else:
            st.dataframe(samples)

        st.markdown("### 聚类类型特征概览（6 类）")
        summary_df = summarize_clusters(artifacts)
        st.dataframe(summary_df)

        st.markdown("### 你的回答对应的标准化特征")
        row = answers_to_feature_row(answers)
        st.dataframe(row)

        with st.expander("查看聚类/KMeans 过程（完成问卷后可见）", expanded=False):
            st.metric("最佳 K (Silhouette)", artifacts.best_k)
            st.caption("K 固定为 6，依据轮廓系数验证聚类效果")
            col_left, col_right = st.columns([2, 1])
            with col_right:
                st.dataframe(format_elbow_silhouette(artifacts))
            with col_left:
                st.pyplot(plot_pca_scatter(artifacts))

        st.success("提示：你可以修改答案，观察类型如何变化。")


if __name__ == "__main__":
    main()
