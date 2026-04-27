## USTI (University Student Type Indicator for HKUST)

一个面向课堂展示的可解释性聚类+问卷系统。核心流程：清洗/标准化学生行为数据 → KMeans 聚类 → 反向设计 10 题行为问卷 → Streamlit 网页展示并给出类型及建议。

### 项目结构
- `data/`（可选）：存放 CSV 数据。此仓库根目录已有 `Student Attitude and Behavior.csv` 与 `student_performance_grade.csv`
- `src/usti_pipeline.py`：数据处理、特征构造、聚类、类型解释、问卷映射与预测函数
- `models/`（可选）：后续可存放持久化的 scaler/encoder/kmeans
- `app.py`：Streamlit 网页应用，加载管道并提供问卷交互
- `requirements.txt`：依赖

### 快速开始
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

### 数据
默认使用 `student_performance_grade.csv`，包含学习投入、出勤、睡眠、压力、GPA 等字段，适合映射到行为问卷。

### 结果
- 自动搜索 K=2..6，基于 Elbow + Silhouette 选最佳 K
- PCA 2D 可视化不同 cluster
- 为每个 cluster 生成趣味命名、行为特点、潜在问题和改进建议
- 10 题行为问卷输入后可预测对应 USTI 类型并展示解释
